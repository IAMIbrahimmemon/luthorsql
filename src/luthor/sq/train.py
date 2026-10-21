"""One training step, and the plan decoder used for evaluation and inference.

Per step: for every database in the batch, encode its option texts and value-candidate
texts (with grad), encode its questions once (token states + pooled), score, and sum the
losses. Twins in the same database group add the minimal-pair margin.
"""
from __future__ import annotations

import torch

from .candidates import Candidate
from .data import DBBatch, span_token_ranges, value_candidate_texts
from .losses import WEIGHTS, example_loss, twin_loss
from .model import CHOICES, ExampleOut, LuthorSQ
from .plan import AGGS, OPS, PARTS, ROWS, Col, Filter, Plan, Proj

import os
import time as _time

PROFILE = {} if os.environ.get("LUTHOR_PROFILE") else None


def _tick(name, t0):
    """Accumulate wall time per phase (GPU-synchronised) when LUTHOR_PROFILE is set."""
    if PROFILE is None:
        return t0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = _time.perf_counter()
    PROFILE[name] = PROFILE.get(name, 0.0) + (t - t0)
    return t


class TextCache:
    """Option / value vectors encoded WITHOUT gradient and reused across steps, re-encoded
    after `refresh` steps. Used in the late phase of training: the question side keeps
    learning with gradient while the schema side costs one cheap forward per refresh."""

    def __init__(self, refresh: int = 400):
        self.refresh = refresh
        self.vecs: dict[str, tuple[int, torch.Tensor]] = {}

    @torch.no_grad()
    def get(self, model: LuthorSQ, texts: list[str], step: int) -> torch.Tensor:
        stale = [t for t in dict.fromkeys(texts) if t not in self.vecs or step - self.vecs[t][0] >= self.refresh]
        if stale:
            V = model.encode_texts(stale)
            for t, v in zip(stale, V):
                self.vecs[t] = (step, v.detach())
        if len(self.vecs) > 400_000:  # bound memory: drop the oldest
            for t in sorted(self.vecs, key=lambda k: self.vecs[k][0])[:100_000]:
                del self.vecs[t]
        return torch.stack([self.vecs[t][1] for t in texts]) if texts else None


def forward_db(model: LuthorSQ, b: DBBatch, option_vecs: torch.Tensor | None = None,
               cache: TextCache | None = None, step: int = 0, partial: bool = False):
    """Score every question of one database batch. Returns [(ExampleOut, Example)], opt_index.
    With `cache`, option and value texts come from the no-gradient cache."""
    t = _time.perf_counter()
    if option_vecs is not None:
        O = option_vecs
    elif partial and cache is not None and b.grad_opts is not None:
        # gold options + hard negatives with gradient, the rest from the no-gradient cache
        gi = [i for i, g in enumerate(b.grad_opts) if g]
        ni = [i for i, g in enumerate(b.grad_opts) if not g]
        G = model.encode_texts([b.option_texts[i] for i in gi]) if gi else None
        N = cache.get(model, [b.option_texts[i] for i in ni], step) if ni else None
        ref = G if G is not None else N
        O = ref.new_zeros(len(b.option_texts), ref.shape[-1])
        if gi:
            O = O.index_put((torch.tensor(gi, device=O.device),), G)
        if ni:
            O = O.index_put((torch.tensor(ni, device=O.device),), N.to(O.dtype))
    elif cache is not None:
        O = cache.get(model, b.option_texts, step)
    else:
        O = model.encode_texts(b.option_texts)
    t = _tick("encode_options", t)
    opt_index = {oid: i for i, oid in enumerate(b.option_ids)}
    H, h, offsets = model.encode_questions([e.text for e in b.examples])
    t = _tick("encode_questions", t)
    # value texts repeat across the questions of one database: encode each once
    per_ex = [value_candidate_texts(e) for e in b.examples]
    uniq = list(dict.fromkeys(t for ts in per_ex for t in ts))
    pos = {t: i for i, t in enumerate(uniq)}
    pre = getattr(b, "_value_vecs", None)
    if pre is not None and option_vecs is not None:
        pv, PV = pre
        pmap = {t: i for i, t in enumerate(pv)}
        V = PV[torch.tensor([pmap[t] for t in uniq], device=PV.device)] if uniq else None
    else:
        V = (cache.get(model, uniq, step) if cache is not None else model.encode_texts(uniq)) if uniq else None
    t = _tick("encode_values", t)
    outs = []
    for k, e in enumerate(b.examples):
        ts = per_ex[k]
        vvecs = V[torch.tensor([pos[t] for t in ts], device=V.device)] if ts else H.new_zeros(0, H.shape[-1])
        spans = span_token_ranges(e, offsets[k])
        kinds = [c.kind == "value" for c in e.cands]
        cols = [opt_index.get(c.col, -1) if c.kind == "value" else -1 for c in e.cands]
        outs.append((model.score(h[k], H[k], O, vvecs, spans, kinds, cols), e))
    _tick("score_heads", t)
    return outs, opt_index


def encode_batch(model: LuthorSQ, batch: list[DBBatch]) -> list[torch.Tensor]:
    """All option + value texts of every database in the batch in ONE length-sorted encode
    (fewer, fuller kernels). Returns per-database option vectors; value vectors are stashed."""
    texts, spans = [], []
    for b in batch:
        vals = list(dict.fromkeys(t for e in b.examples for t in value_candidate_texts(e)))
        spans.append((len(texts), len(b.option_texts), len(vals), vals))
        texts += b.option_texts + vals
    V = model.encode_texts(texts)
    out = []
    for (start, n_opt, n_val, vals), b in zip(spans, batch):
        b._value_vecs = (vals, V[start + n_opt:start + n_opt + n_val])
        out.append(V[start:start + n_opt])
    return out


def train_step(model: LuthorSQ, batch: list[DBBatch], gamma: float = 1.0, cache: TextCache | None = None,
               step: int = 0, backward: bool = False, fused: bool = False,
               partial: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss over a batch of database groups. With backward=True the gradient is taken per
    database (scaled to the batch mean), so only one database's activations are alive at a time."""
    totals = {k: 0.0 for k in WEIGHTS}
    loss = None
    n = 0
    n_total = sum(len(b.examples) for b in batch)
    tf = _time.perf_counter()
    fused_O = encode_batch(model, batch) if fused and cache is None and not partial else None
    if fused_O is not None:
        _tick("encode_options+values(fused)", tf)
    for bi, b in enumerate(batch):
        db_loss = None
        outs, opt_index = forward_db(model, b, option_vecs=fused_O[bi] if fused_O else None, cache=cache, step=step,
                                     partial=partial)
        by_id = {e.id: (o, e) for o, e in outs}
        tl = _time.perf_counter()
        for o, e in outs:
            parts = example_loss(model, o, e.target, opt_index, gamma)
            if e.twin_of and e.twin_of in by_id:
                oa, ea = by_id[e.twin_of]
                parts["twin"] = twin_loss(model, (oa, ea.target, opt_index), (o, e.target, opt_index), e.twin_slot, gamma)
            l = sum(WEIGHTS[k] * v for k, v in parts.items())
            db_loss = l if db_loss is None else db_loss + l
            n += 1
            for k, v in parts.items():
                totals[k] += float(v.detach())
        tl = _tick("losses", tl)
        if db_loss is None:
            continue
        if backward and not fused:
            (db_loss / max(n_total, 1)).backward()
            db_loss = db_loss.detach()
            _tick("backward", tl)
        loss = db_loss if loss is None else loss + db_loss
    loss = loss / max(n, 1)
    if backward and fused:
        tb = _time.perf_counter()
        loss.backward()
        loss = loss.detach()
        _tick("backward", tb)
    return loss, {k: v / max(n, 1) for k, v in totals.items()}


# ---- decoding ---------------------------------------------------------------------------
def _choice(out: ExampleOut, head: str) -> str:
    return CHOICES[head][int(out.choice[head].argmax())]


def decode(model: LuthorSQ, out: ExampleOut, option_ids: list[str], cands: list[Candidate],
           filter_threshold: float = 0.0, unsupported_threshold: float = 0.5,
           via_threshold: float = 1.0) -> Plan | None:
    """ExampleOut -> Plan (None = the model abstains: outside the plan grammar)."""
    p_unsup = torch.softmax(out.choice["outcome"], -1)[1]
    if float(p_unsup) >= unsupported_threshold:
        return None
    col = lambda i: Col(*option_ids[i].split(".", 1))

    def values_for(i: int, op: str, query=None) -> list:
        if op in ("is_null", "not_null"):
            return []
        s = model.value_scores(out, out.ov[i], i, query=query)
        if s.numel() == 0:
            return []
        order = s.argsort(descending=True).tolist()
        if op == "between":
            picked = sorted(order[:2])
        elif op in ("in", "not_in"):
            picked = [j for j in order if s[j] > 0][:8] or order[:1]
        else:
            picked = order[:1]
        return [cands[j].value for j in picked]

    n_proj = int(_choice(out, "n_proj"))
    cond_mask = _choice(out, "cond_mask")
    projections = []
    for k in range(n_proj):
        i = int(out.slots[f"proj{k}"].argmax())
        agg = AGGS[int(out.agg[f"proj{k}"][i].argmax())]
        c = col(i)
        if c.column == ROWS:
            agg = "count"
        cond = None
        if (k == 0 and cond_mask in ("p0", "both")) or (k == 1 and cond_mask in ("p1", "both")):
            r = f"cond{k}"
            j = int(out.slots[r].argmax())
            op = OPS[int(out.op[r][j].argmax())]
            cond = Filter(col(j), op, values_for(j, op), PARTS[int(out.part[r][j].argmax())])
        projections.append(Proj(agg, c, cond))
    plan = Plan(projections=projections, compose=_choice(out, "compose"), distinct=_choice(out, "distinct") == "yes")

    n_f = int(_choice(out, "n_filter"))
    fs = out.slots["filter"]
    for i in fs.argsort(descending=True)[:n_f].tolist():
        op = OPS[int(out.op["filter"][i].argmax())]
        plan.filters.append(Filter(col(i), op, values_for(i, op), PARTS[int(out.part["filter"][i].argmax())]))

    n_g = int(_choice(out, "n_group"))
    plan.group_by = [col(i) for i in out.slots["group"].argsort(descending=True)[:n_g].tolist()]
    plan.grain = _choice(out, "grain")
    if plan.grain != "none":
        plan.time_col = col(int(out.slots["time"].argmax()))
    plan.order_dir = _choice(out, "order_dir")
    if plan.order_dir != "none":
        i = int(out.slots["order"].argmax())
        c = col(i)
        agg = "count" if c.column == ROWS else AGGS[int(out.agg["order"][i].argmax())]
        plan.order = Proj(agg, c)
    if len(plan.tables()) > 1:  # join hints only matter when there is a join
        rows_opts = [i for i, oid in enumerate(option_ids) if oid.endswith("." + ROWS)]
        plan.via = [option_ids[i].rsplit(".", 1)[0] for i in rows_opts
                    if float(out.slots["via"][i]) > via_threshold][:2]
        plan.join_cols = [col(i) for i in out.slots["join"].argsort(descending=True)[:6].tolist()
                          if float(out.slots["join"][i]) > 0 and not option_ids[i].endswith("." + ROWS)]
    lk = _choice(out, "limit_kind")
    if lk == "one":
        plan.limit = 1
    elif lk == "literal":
        s = model.limit_scores(out)
        nums = [j for j, c in enumerate(cands) if isinstance(c.value, int) and c.value > 0]
        if nums:
            j = max(nums, key=lambda j: float(s[j]))
            plan.limit = int(cands[j].value)
    return plan
