"""Learned reranker over the top-K candidate plans (round 3, change 4).

The slot model's top-8 holds the right answer far more often than its greedy pick (oracle@8 80%
vs greedy 69% on held-out databases), and most wrong picks run and return plausible rows, so
execution checks alone can't tell them apart. The reranker reads the whole candidate:

    [schema text] ──backbone + rerank LoRA (once per database)──► cache
    [question] "Candidate SQL:" [sql] "Does it answer the question?" ──over the cache──► h_last
    score = w · h_last

It is a second LoRA adapter ("rerank") on the same OpenJev backbone, so the slot model's adapter
is untouched and the two can be switched per pass (peft set_adapter). Trained listwise: each
correct candidate against the question's wrong ones.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..sq.schema_card import SchemaCard
from .prefix_cache import expand_cache, release_cache
from .serialize import question_text, schema_text

ADAPTER = "rerank"
SQL_MAX = 700


def candidate_text(question: str, context: str, sql: str, result: str | None = None) -> str:
    """With result (round 7): a short preview of what the SQL returned (shape, types, first row), so the
    validator can see a list answering "how many", or a single number answering "list the ..."."""
    q = question_text(question, context).text.removesuffix("\nPlan:")
    res = f"\nResult: {result}" if result else ""
    return f"{q}\nCandidate SQL: {sql[:SQL_MAX]}{res}\nDoes this SQL answer the question exactly?"


def result_preview(rows: list) -> str:
    """'3 rows x 2 cols (text, integer); first: ('Monaco', 1950)' -- short and database-agnostic."""
    if not rows:
        return "0 rows"
    first = rows[0]
    kinds = ", ".join("null" if v is None else "integer" if isinstance(v, int) else "real" if isinstance(v, float)
                      else "text" for v in first)
    shown = "(" + ", ".join(repr(v)[:40] for v in first) + ")"
    return f"{len(rows)} row{'s' if len(rows) != 1 else ''} x {len(first)} col{'s' if len(first) != 1 else ''} ({kinds}); first: {shown}"


class Reranker(nn.Module):
    """adapter: the LoRA it scores with. "rerank" (default) is its own second adapter; "default" shares the slot
    model's adapter (single-LoRA System 1, docs/paper_notes/single_lora_2026-10-16.md), so no adapter is added."""

    def __init__(self, backbone, lora_r: int = 16, adapter: str = ADAPTER, head: str = "linear", head_hidden: int = 1024):
        super().__init__()
        from peft import LoraConfig
        self.backbone = backbone
        self.adapter = adapter
        body = backbone.model
        if adapter not in body.peft_config:
            if adapter != ADAPTER:
                raise ValueError(f"adapter {adapter!r} not on the backbone (has {list(body.peft_config)})")
            body.add_adapter(ADAPTER, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                 target_modules="all-linear"))
        if head == "linear":
            self.score = nn.Linear(backbone.hidden, 1)
            last = self.score
        elif head == "mlp":  # 2-layer scoring head (single-LoRA option 3); the output layer starts at zero, as the linear one
            self.score = nn.Sequential(nn.Linear(backbone.hidden, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1))
            last = self.score[2]
        else:
            raise ValueError(f"head {head!r}: linear or mlp")
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @property
    def body(self):
        return self.backbone.model

    @property
    def device(self):
        return next(self.score.parameters()).device

    def activate(self, train: bool = False):
        """Make the rerank adapter the active (and only trainable) one."""
        self.body.set_adapter(self.adapter)
        for n, p in self.body.named_parameters():
            p.requires_grad = train and f".{self.adapter}." in n

    def scores(self, card: SchemaCard, items: list[tuple], tables: list[str] | None = None,
               max_tok: int = 384) -> torch.Tensor:
        """items: (question, context, sql[, result preview]) rows on one database -> (n,) scores. One schema pass, then
        all rows continue it in one batch (right padding; causal, so each row's last token sees no pad)."""
        st = schema_text(card, tables)
        tok = self.backbone.tok
        ids = tok(st.text, add_special_tokens=False)["input_ids"]
        rows = [tok(candidate_text(*it), add_special_tokens=False)["input_ids"][-max_tok:] for it in items]
        return Reranker.scores_ids(self, ids, rows)  # unbound: callers pass duck-typed rerankers

    def scores_ids(self, ids: list[int], rows: list[list[int]]) -> torch.Tensor:
        """Token-level scores(): one pass over the prefix ids, then every row continues it (joint scoring: the prefix is
        schema + question + results table, the rows are the candidates)."""
        tok = self.backbone.tok
        out = self.body(input_ids=torch.tensor([ids], device=self.device), use_cache=True)
        n, T = len(rows), max(len(r) for r in rows)
        x = torch.full((n, T), tok.pad_token_id, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.tensor(r, device=self.device)
        cache = expand_cache(out.past_key_values, n)
        pos = torch.arange(len(ids), len(ids) + T, device=self.device)[None].expand(n, -1)
        try:
            H = self.body(input_ids=x, past_key_values=cache, position_ids=pos, use_cache=True).last_hidden_state
        finally:  # also on out-of-memory: otherwise the cycle keeps the cache and its graph alive
            release_cache(cache)
        last = torch.tensor([len(r) - 1 for r in rows], device=self.device)
        h = H[torch.arange(n, device=self.device), last].float()
        return self.score(h).squeeze(-1)

    def head_state(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}


def listwise_loss(s: torch.Tensor, ok: list[bool]) -> torch.Tensor:
    """Each correct candidate against all wrong ones (correct ones don't compete)."""
    pos = [i for i, v in enumerate(ok) if v]
    neg = s[torch.tensor([not v for v in ok], device=s.device)]
    return torch.stack([torch.logsumexp(torch.cat([s[p:p + 1], neg]), 0) - s[p] for p in pos]).mean()


def multipos_loss(s: torch.Tensor, ok: list[bool], smooth: float = 0.0) -> torch.Tensor:
    """-log sum_{pos} softmax(s): the list's probability mass on any right candidate (right ones do not compete with
    each other). With smooth > 0, mixed with the uniform cross-entropy over the whole list (label smoothing):
    (1 - smooth) * that + smooth * mean_j(-log softmax(s)_j). With one positive and smooth = 0 it equals listwise_loss."""
    pos = torch.tensor(ok, device=s.device)
    lse = torch.logsumexp(s, 0)
    loss = lse - torch.logsumexp(s[pos], 0)
    if smooth:
        loss = (1 - smooth) * loss + smooth * (lse - s.mean())
    return loss


def group_loss(s: torch.Tensor, ok: list[bool], gid: list, smooth: float = 0.0) -> torch.Tensor:
    """multipos_loss over result groups (VALHARD): candidates with the same gid (result signature) form one group
    scored log-mean-exp of its members (size-neutral: a class with more SQL texts gets no extra mass), a group is right
    when any member is; label smoothing is uniform over groups. With one candidate per group it equals multipos_loss."""
    order, idx = [], {}
    for k, g in enumerate(gid):
        if g not in idx:
            idx[g] = []
            order.append(g)
        idx[g].append(k)
    gs = torch.stack([torch.logsumexp(s[idx[g]], 0) - math.log(len(idx[g])) for g in order])
    return multipos_loss(gs, [any(ok[k] for k in idx[g]) for g in order], smooth)


# ---- gradient cache: exact gradients for a list that does not fit in one pass ---------------------------------------
# Pass 1 scores every candidate without a graph (chunk by chunk over one shared schema prefix), the listwise loss and
# g = dL/ds come from that detached score vector, pass 2 re-runs each chunk with a graph and backpropagates sum(g * s_c).
# The prefix: its key/value states are computed ONCE with a graph and detached into leaves; every chunk (both passes)
# reads the leaves, pass 2 accumulates dL/dP into them, and one final backward pushes that through the prefix graph.
# dL/dθ = Σ_j g_j ∂s_j/∂θ (direct) + (Σ_j g_j ∂s_j/∂P) · dP/dθ, so the sum over chunks is the un-chunked gradient.
# Dropout: the RNG state before each chunk is saved in pass 1 and restored in pass 2, so a chunk sees the same masks in
# both passes (the gradient is exact for the masks pass 1 drew); the prefix runs once, so all chunks share its masks.

def _rng_get():
    return torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _rng_set(st):
    torch.set_rng_state(st[0])
    if st[1] is not None:
        torch.cuda.set_rng_state_all(st[1])


def gradcache_backward(prefix_fn, chunk_fn, n_chunks: int, loss_fn, replay_prefix: bool = False):
    """prefix_fn() -> list of tensors computed WITH a graph (or [] when chunk_fn recomputes the prefix itself);
    chunk_fn(leaves, c) -> scores of chunk c; loss_fn(scores of all chunks, concatenated in chunk order) -> scalar.
    replay_prefix: chunk_fn recomputes the prefix each call (backbones whose cache cannot be detached into leaves);
    then every chunk starts from the same RNG state, so the recomputed prefix is identical in every chunk and pass.
    Runs the backward (gradients accumulate into .grad); returns (loss value, detached scores)."""
    P = list(prefix_fn() or [])
    leaves = [t.detach().requires_grad_(t.requires_grad) for t in P]
    acc = [None] * len(P)
    states, parts = [], []
    s0 = _rng_get()
    with torch.no_grad():
        for c in range(n_chunks):
            if replay_prefix:
                _rng_set(s0)
            states.append(_rng_get())
            parts.append(chunk_fn(leaves, c).float())
    end = _rng_get()
    s = torch.cat(parts).detach().requires_grad_(True)
    loss = loss_fn(s)
    loss.backward()
    g = s.grad
    parts = None
    off = 0
    for c in range(n_chunks):
        _rng_set(states[c])
        sc = chunk_fn(leaves, c).float()
        (sc * g[off:off + len(sc)]).sum().backward()
        off += len(sc)
        sc = None
        for i, l in enumerate(leaves):  # accumulate dL/dP in fp32 (the leaves are bf16 on the GPU)
            if l.grad is not None:
                acc[i] = l.grad.float() if acc[i] is None else acc[i].add_(l.grad.float())
                l.grad = None
    _rng_set(end)
    pairs = [(t, a.to(t.dtype)) for t, a in zip(P, acc) if a is not None]
    if pairs:
        torch.autograd.backward([t for t, _ in pairs], [a for _, a in pairs])
    return float(loss.detach()), s.detach()


def full_attention(body) -> bool:
    """True when every decoder layer keeps a plain key/value cache (OmniSQL / Qwen2.5): the prefix can be detached."""
    cfg = body.config.get_text_config() if hasattr(body.config, "get_text_config") else body.config
    kinds = set(getattr(cfg, "layer_types", None) or ["full_attention"])
    return not getattr(cfg, "use_sliding_window", False) and not (kinds - {"full_attention"})


def row_ids(tok, items: list[tuple], max_tok: int = 384) -> list[list[int]]:
    return [tok(candidate_text(*it), add_special_tokens=False)["input_ids"][-max_tok:] for it in items]


def prefix_kv(rr, card: SchemaCard, tables: list[str] | None) -> tuple[list[torch.Tensor], int]:
    """The schema pass (with whatever grad mode is active) -> flat [k0, v0, k1, v1, ...] and its token count."""
    return prefix_kv_ids(rr, rr.backbone.tok(schema_text(card, tables).text, add_special_tokens=False)["input_ids"])


def prefix_kv_ids(rr, ids: list[int]) -> tuple[list[torch.Tensor], int]:
    """prefix_kv for given prefix token ids (joint scoring: schema + question + results table)."""
    cache = rr.body(input_ids=torch.tensor([ids], device=rr.device), use_cache=True).past_key_values
    kv = []
    for layer in cache.layers:
        kv += [layer.keys, layer.values]
    return kv, len(ids)


def rows_over_prefix(rr, kv: list[torch.Tensor], n_prefix: int, rows: list[list[int]]) -> torch.Tensor:
    """Scores of token rows continuing a prefix given as key/value tensors (prefix_kv); same numbers as scores()."""
    from types import SimpleNamespace

    from .prefix_cache import question_cache
    tok = rr.backbone.tok
    n, T = len(rows), max(len(r) for r in rows)
    x = torch.full((n, T), tok.pad_token_id, dtype=torch.long, device=rr.device)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r, device=rr.device)
    prefix = SimpleNamespace(layers=[SimpleNamespace(keys=kv[i], values=kv[i + 1]) for i in range(0, len(kv), 2)])
    cache = question_cache(prefix, n, rr.body.config)  # read-only: every chunk continues the same prefix
    pos = torch.arange(n_prefix, n_prefix + T, device=rr.device)[None].expand(n, -1)
    H = rr.body(input_ids=x, past_key_values=cache, position_ids=pos, use_cache=True).last_hidden_state
    last = torch.tensor([len(r) - 1 for r in rows], device=rr.device)
    return rr.score(H[torch.arange(n, device=rr.device), last].float()).squeeze(-1)


# ---- joint scoring (--joint-results; docs/paper_notes/research/validator/01_literature.md T1a, val3_trainer.md) -------
# Pointwise (default): prefix = schema, row = question + candidate SQL. Joint: the prefix also holds the question and a
# compact table of ALL distinct live results of the list, and each row is the candidate SQL plus the id of its result:
#     [schema][question + evidence][Candidate results: R1: <shape, types, first row> ... Rk: ...]
#         └─ row j: Candidate SQL: <sql> / Result: R3 / Does this SQL answer the question exactly?
# so each candidate is scored with every alternative answer in view, in one forward pass (rows_over_prefix + the
# gradient cache unchanged; only the prefix differs, one prefix per question). Token limits: at most
# JOINT_MAX_RESULTS results in the table (a candidate whose result was cut reads "Result: not shown"), each result line
# cut to JOINT_RES_CHARS characters (one-row cells to JOINT_CELL_CHARS, at most JOINT_COLS columns and JOINT_ROWS rows
# when a structured 'rows' field is present), question + evidence cut to JOINT_Q_CHARS. Which results are shown and in
# what order is the caller's choice (rerank_sq2.py: random subset + random order in training, because prebuilt lists put
# the right candidate first; at score time the first JOINT_MAX_RESULTS in pool order, averaged over --joint-orders
# seeded permutations, the second being the reverse of the first).

JOINT_MAX_RESULTS = 16
JOINT_RES_CHARS = 160
JOINT_CELL_CHARS = 30
JOINT_COLS = 6
JOINT_ROWS = 3
JOINT_Q_CHARS = 2000
JOINT_HEAD = "\nCandidate results (every distinct result the candidates return; id: shape, column types, first row):\n"


def _cell(v) -> str:
    return repr(v)[:JOINT_CELL_CHARS]


def result_line(c: dict) -> str | None:
    """One result's table text: a structured 'rows' field (first rows; with 'n_rows' / 'cols' when known) if the list
    has one, else the result_preview string in 'res'; None when the candidate carries neither."""
    rows = c.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], (list, tuple)):
        n, m = int(c.get("n_rows", len(rows))), len(rows[0])
        kinds = ", ".join("null" if v is None else "integer" if isinstance(v, int) else "real" if isinstance(v, float)
                          else "text" for v in rows[0][:JOINT_COLS])
        shown = "; ".join("(" + ", ".join(_cell(v) for v in r[:JOINT_COLS]) + (", ..." if len(r) > JOINT_COLS else "") + ")"
                          for r in rows[:JOINT_ROWS])
        line = f"{n} row{'s' if n != 1 else ''} x {m} col{'s' if m != 1 else ''} ({kinds}); rows: {shown}"
    elif isinstance(rows, list) and not rows and c.get("runs"):
        line = "0 rows"
    else:
        line = c.get("res")
    return line[:JOINT_RES_CHARS] if line else None


def result_classes(cands: list[dict]) -> tuple[list[str], list[str], list[str | None]]:
    """Distinct live results of a list, in first-appearance order -> (keys, table lines, each candidate's key or None).
    Live = runs (or a prebuilt-list mutant); same 'sig' = same result. A class's line is its first member's that has
    one (pair members carry no 'res' of their own), else "result not available"."""
    keys, lines, of = [], {}, []
    for k, c in enumerate(cands):
        if not (c.get("runs") or c.get("mutant")):
            of.append(None)
            continue
        key = c.get("sig") or f"_row{k}"
        if key not in lines:
            keys.append(key)
            lines[key] = None
        if lines[key] is None:
            lines[key] = result_line(c)
        of.append(key)
    return keys, [lines[k] or "result not available" for k in keys], of


def joint_texts(question: str, context: str, cands: list[dict], shown: list[int]) -> tuple[str, list[str]]:
    """shown: indices into result_classes' keys, in table order (R1 = shown[0]). -> (question + table text, one row
    text per candidate). The prefix is schema_text(...).text + this head (joint_prefix_ids)."""
    keys, lines, of = result_classes(cands)
    rid = {keys[j]: f"R{i + 1}" for i, j in enumerate(shown)}
    q = question_text(question, context).text.removesuffix("\nPlan:")[:JOINT_Q_CHARS]
    head = q + JOINT_HEAD + "".join(f"R{i + 1}: {lines[j]}\n" for i, j in enumerate(shown))
    rows = [joint_row_text(c["sql"], "error" if k is None else rid.get(k, "not shown")) for c, k in zip(cands, of)]
    return head, rows


def joint_row_text(sql: str, rid: str) -> str:
    return f"\nCandidate SQL: {sql[:SQL_MAX]}\nResult: {rid}\nDoes this SQL answer the question exactly?"


def joint_prefix_ids(tok, card: SchemaCard, tables: list[str] | None, head: str) -> list[int]:
    return tok(schema_text(card, tables).text + head, add_special_tokens=False)["input_ids"]


def average_states(states: list[dict]) -> dict:
    """Uniform average of parameter dicts (same keys), in fp32, cast back to each tensor's dtype (--avg-last: the tail
    checkpoints of one run -> one LoRA + head)."""
    if not states:
        raise ValueError("average_states: no checkpoints")
    out = {}
    for n, t in states[0].items():
        acc = torch.zeros_like(t, dtype=torch.float32)
        for s in states:
            acc += s[n].float()
        out[n] = (acc / len(states)).to(t.dtype)
    return out
