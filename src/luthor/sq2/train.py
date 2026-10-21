"""Examples, losses and one training / inference pass for LuthorSQ2.

Per step: a few databases x several questions. Per database, the schema prefix runs once
(with gradient), the questions continue it in one batched call, and the database's loss is
back-propagated before the next database (bounded memory).
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..sq.candidates import _WORD, Candidate, _norm, find_candidates
from ..sq.plan import ROWS, Plan, join_tree
from ..sq.schema_card import SchemaCard
from .model import Encoded, LuthorSQ2, TorchScores
from .serialize import MAX_CANDS, SchemaAug, cand_rank, random_aug, schema_text, serve_tables, train_tables  # noqa: F401

AUG_P = 0.0  # share of database-steps whose schema is shown renamed (round 3: set by train_sq2.py --schema-aug)
DESC_DROP = 0.0  # share of the other database-steps shown with no descriptions (train_sq2.py --desc-drop)
from .targets import Targets, Unsupported, build_targets, decode, gold_options

_LOG = logging.getLogger("luthorsql")

W = {"choice": 1.0, "ptr": 1.0, "set": 1.0, "pair": 1.0, "vals": 1.0, "limit": 0.5}


def question_body(row: dict) -> str:
    return row["question"] + (f"\nContext: {row['context']}" if row.get("context") else "")


CAND_ORDER = "v1"  # v1: round-6 behaviour; v2: train/serve parity (train_sq2.py --cand-order, stored in args.json)


def twin_db_values(row: dict, base: dict) -> list[tuple]:
    """Database-lookup hits for a twin row (prepare_plans.twins) as ValueIndex.match on its own body would give them,
    without the database: an order twin carries its base row's hits at stale offsets (the swapped word changed
    length) and a value twin was matched on its question alone, so it lacks the hits in the shared context. Hits
    are re-anchored to where their words are in this body (nearest to the old offset; gone if absent), and the
    base row's context hits are added, shifted."""
    toks = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(question_body(row).lower())]

    def place(v, a: int):  # the n-gram ValueIndex.match would hit (its words = _norm(v)), nearest to offset a
        k, best = _norm(v).split(), None
        for j in range(len(toks) - len(k) + 1) if k else ():
            if [t for t, _, _ in toks[j:j + len(k)]] == k and (best is None or abs(toks[j][1] - a) < abs(best[0] - a)):
                best = (toks[j][1], toks[j + len(k) - 1][2])
        return best

    out, seen = [], set()
    shift = len(row["question"]) - len(base["question"])
    q0 = len(base["question"])
    for src, d in ((row.get("db_values") or [], 0), ([h for h in base.get("db_values") or [] if h[2] >= q0], shift)):
        for col, v, a, b in src:
            p = place(v, a + d)
            if p is not None and (col, v, *p) not in seen:
                seen.add((col, v, *p))
                out.append((col, v, *p))
    return out


def select_cands(cands: list[Candidate], need: set[int] = frozenset(), cap: int = MAX_CANDS,
                 rnd: random.Random | None = None, order: str | None = None,
                 body: str = "") -> tuple[list[Candidate], dict[int, int]]:
    """At most `cap` candidates, kept in find order. v1: everything the question mentions first, then enumerated
    low-cardinality values (shuffled when `rnd` is given); `need` (training: gold literals) is always kept, even
    past the cap. v2: cand_rank's order cut at the cap, with no shuffle and no force-keep, so training sees exactly
    the list serving builds (`need` and `rnd` are ignored; the caller drops targets whose literal was cut)."""
    if (order or CAND_ORDER) == "v2":
        keep = sorted(cand_rank(cands, body)[:cap])
    else:
        ment = [i for i, c in enumerate(cands) if c.span is not None]
        enum = [i for i, c in enumerate(cands) if c.span is None]
        if rnd is not None:
            rnd.shuffle(enum)
        keep = list(dict.fromkeys(sorted(need) + ment + enum))[:max(cap, len(need))]
        keep.sort()
    m = {old: new for new, old in enumerate(keep)}
    return [cands[i] for i in keep], m


@dataclass
class Example:
    id: str
    db_id: str
    question: str
    context: str
    cands: list[Candidate]
    t: Targets | None
    tables: set[str]   # tables the plan touches + join bridges (kept when a schema is trimmed)


def _remap(t: Targets, m: dict[int, int]) -> Targets:
    """Candidate indices -> positions in the kept list. A value target whose literal was cut (v2 only: v1 force-keeps
    gold literals) is dropped, with all its literals (half an IN list would teach a wrong answer); the example's
    other targets (structure, columns, operators) still train. Serving could not have produced that literal."""
    t.vals = [(s, o, [m[i] for i in idx]) for s, o, idx in t.vals if all(i in m for i in idx)]
    t.limit = {k: m[v] for k, v in t.limit.items() if v in m}
    return t


def make_example(row: dict, card: SchemaCard, rnd: random.Random | None = None, strict: bool = True,
                 order: str | None = None) -> Example | None:
    """strict=False keeps a question whose plan the slots can't express (t=None, eval only). `order`: candidate
    order (v1 / v2, default CAND_ORDER); v2 uses row["_base"] (a twin's base row, set by train_sq2.load_sets)."""
    body = question_body(row)
    dbv = [tuple(x) for x in row.get("db_values") or []]
    if (order or CAND_ORDER) == "v2" and row.get("_base"):  # a twin: its hits as serving would find them
        dbv = twin_db_values(row, row["_base"])
    cands = find_candidates(body, card, extra=dbv)
    t, tables = None, set()
    if row.get("outcome") == "plan":
        plan = Plan.from_dict(row["plan"])
        try:
            t = build_targets(plan, cands)
        except (Unsupported, ValueError):
            if strict:
                return None
            cands, _ = select_cands(cands, order=order, body=body)
            return Example(row["id"], row["db_id"], row["question"], row.get("context", ""), cands, None, set())
        need = {i for _, _, idx in t.vals for i in idx} | set(t.limit.values())
        cands, m = select_cands(cands, need, rnd=rnd, order=order, body=body)
        t = _remap(t, m)
        tables = {o.split(".", 1)[0] for o in gold_options(t)}
        try:
            g = card.graph()
            jt, _ = join_tree(g, [x for x in tables if x in g.columns])
            tables |= set(jt)
        except Exception:
            _LOG.debug("%s: no join tree for the gold tables", row["id"], exc_info=True)
    else:
        cands, _ = select_cands(cands, order=order, body=body)
    return Example(row["id"], row["db_id"], row["question"], row.get("context", ""), cands, t, tables)


def load_dir(d: Path, drop_empty: bool = True, limit: int = 0) -> tuple[dict, list[Example], dict]:
    cards = {p.stem: SchemaCard.load(p) for p in (d / "cards").glob("*.json")}
    exs, gold, skipped = [], {}, 0
    rnd = random.Random(0)
    for line in open(d / "plans.jsonl"):
        r = json.loads(line)
        if r["db_id"] not in cards or r.get("outcome") != "plan" or (drop_empty and r.get("empty_result")):
            continue
        e = make_example(r, cards[r["db_id"]], rnd)
        if e is None:
            skipped += 1
            continue
        gold[r["id"]] = r
        exs.append(e)
        if limit and len(exs) >= limit:
            break
    print(f"{d}: {len(exs)} examples, {skipped} skipped (outside the v2 slots)", flush=True)
    return cards, exs, gold


# ---- losses ----------------------------------------------------------------------------

def _multi_pos_ce(s: torch.Tensor, pos: list[int]) -> torch.Tensor:
    """Each positive against all negatives (positives don't compete with each other)."""
    mask = torch.ones_like(s, dtype=torch.bool)
    mask[pos] = False
    neg = s[mask]
    out = []
    for p in pos:
        out.append(torch.logsumexp(torch.cat([s[p:p + 1], neg]), 0) - s[p])
    return torch.stack(out).mean()


def example_loss(model: LuthorSQ2, e: Encoded, t: Targets, oix: dict[str, int]) -> tuple[torch.Tensor, dict]:
    parts = defaultdict(list)
    dev = e.S.device
    for k, c in t.choice.items():
        parts["choice"].append(F.cross_entropy(model.choice_logits(e, k)[None], torch.tensor([c], device=dev)))
    for s, o in t.ptr.items():
        parts["ptr"].append(F.cross_entropy(model.ptr_scores(e, s)[None], torch.tensor([oix[o]], device=dev)))
    for s, os_ in t.sets.items():
        if os_:
            parts["set"].append(_multi_pos_ce(model.ptr_scores(e, s), [oix[o] for o in os_]))
    for s, o, h, c in t.pair:
        parts["pair"].append(F.cross_entropy(model.pair_logits(e, s, oix[o], h)[None], torch.tensor([c], device=dev)))
    for s, o, idx in t.vals:
        parts["vals"].append(_multi_pos_ce(model.value_logits(e, s, oix[o]), list(dict.fromkeys(idx))))
    for s, i in t.limit.items():
        parts["limit"].append(F.cross_entropy(model.value_logits(e, s, None)[None], torch.tensor([i], device=dev)))
    loss = sum(W[k] * torch.stack(v).mean() for k, v in parts.items())
    return loss, {k: float(torch.stack(v).detach().mean()) for k, v in parts.items()}


# ---- passes ------------------------------------------------------------------------------

_TOK_CACHE: dict = {}


def encode_db(model: LuthorSQ2, card: SchemaCard, exs: list[Example], tables: list[str] | None = None, aug=None):
    st = schema_text(card, tables, aug)
    key = (card.db_id, tuple(tables) if tables is not None else None, id(aug) if aug else None)
    if key not in _TOK_CACHE:
        if len(_TOK_CACHE) > 4096:
            _TOK_CACHE.clear()
        _TOK_CACHE[key] = model.prepare_schema(st)
    oix = {o: i for i, o in enumerate(st.option_ids)}
    se = model.encode_schema(_TOK_CACHE[key])
    names = aug.names if aug is not None and getattr(model, "value_side", False) else None
    preps = [model.prepare_question(e.question, e.cands, e.context, names) for e in exs]
    mix = None
    if aug is not None:  # renamed schema: match the names the model is shown, not the real ones
        def shown(o: str) -> str:
            t, _, c = o.partition(".")
            return f"{aug.names.get(t, t)}.{c if c == ROWS else aug.names.get(o, c)}"
        mix = {shown(o): i for o, i in oix.items()}
    enc = model.encode_questions(se, preps, [e.cands for e in exs], oix, mix)
    return st, oix, [model.project(x) for x in enc]


def train_db(model: LuthorSQ2, card: SchemaCard, exs: list[Example], budget_chars: int, rnd: random.Random,
             backward: bool = True, scale: float = 1.0, drop_desc: bool = False) -> tuple[float, dict]:
    gold_t = set().union(*(e.tables for e in exs))
    tables = train_tables(card, gold_t, budget_chars, rnd)
    aug = random_aug(card, rnd, AUG_P) if AUG_P > 0 else None
    if aug is None and DESC_DROP > 0 and rnd.random() < DESC_DROP:  # a database that ships no descriptions
        aug = SchemaAug({}, drop_desc=True)
    if drop_desc:  # train_sq2 --oom-drop-desc: this batch did not fit with descriptions (names augmentation kept)
        aug = SchemaAug(aug.names if aug is not None else {}, drop_desc=True)
    st, oix, enc = encode_db(model, card, exs, tables, aug)
    if aug is not None:
        _TOK_CACHE.pop((card.db_id, tuple(tables) if tables is not None else None, id(aug)), None)
    losses, parts = [], defaultdict(list)
    for e, x in zip(exs, enc):
        l, p = example_loss(model, x, e.t, oix)
        losses.append(l)
        for k, v in p.items():
            parts[k].append(v)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):  # never back-propagate a non-finite loss (one bad batch would poison the weights)
        print(f"non-finite loss on {card.db_id}: skipped", flush=True)
        return 0.0, {}
    if backward:
        try:
            (loss * scale).backward()
        except torch.OutOfMemoryError as e:
            raise BackwardOOM(card.db_id) from e
    return float(loss), {k: sum(v) / len(v) for k, v in parts.items()}


class BackwardOOM(RuntimeError):
    """Out of memory during backward: some gradients were already accumulated, so the whole optimizer step is
    unusable (an OOM in the forward pass leaves the gradients untouched and the questions can be retried)."""


def group_seed(seed: int, epoch: int) -> int:
    """db_groups' seed for one epoch of a run trained with --seed `seed`. Seed 0 gives the epoch number itself, the
    order every run before --seed had; other seeds get their own stream per epoch."""
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return epoch if seed == 0 else (seed << 32) + epoch


def db_groups(exs: list[Example], q_per_db: int, seed: int) -> list[tuple[str, list[Example]]]:
    rnd = random.Random(seed)
    by_db = defaultdict(list)
    for e in exs:
        by_db[e.db_id].append(e)
    chunks = []
    for db, es in by_db.items():
        rnd.shuffle(es)
        chunks += [(db, es[i:i + q_per_db]) for i in range(0, len(es), q_per_db)]
    rnd.shuffle(chunks)
    return chunks


def serve_chunks(card: SchemaCard, exs: list[Example], bs: int):
    """(tables, chunk) for inference. tables is None (whole schema, one cached prefix per database) unless the
    schema overflows the model's context; then each question gets its own trimmed table list, and questions
    that pick the same tables share a prefix."""
    if serve_tables(card, "") is None:
        for i in range(0, len(exs), bs):
            yield None, exs[i:i + bs]
        return
    groups = defaultdict(list)
    for e in exs:
        cols = {c.col for c in e.cands if c.kind == "value" and c.col}
        groups[tuple(serve_tables(card, question_body({"question": e.question, "context": e.context}), cols) or ())].append(e)
    for tabs, es in groups.items():
        for i in range(0, len(es), bs):
            yield (list(tabs) or None), es[i:i + bs]


@torch.no_grad()
def predict(model: LuthorSQ2, card: SchemaCard, exs: list[Example], bs: int = 16) -> list[Plan | None]:
    got = {}  # by position: trimmed schemas regroup the questions
    pos = {id(e): i for i, e in enumerate(exs)}
    for tabs, chunk in serve_chunks(card, exs, bs):
        st, oix, enc = encode_db(model, card, chunk, tabs)
        for e, x in zip(chunk, enc):
            try:
                got[pos[id(e)]] = decode(TorchScores(model, x, st.option_ids, e.cands))
            except Exception:
                _LOG.warning("%s: decode failed", e.id, exc_info=True)
                got[pos[id(e)]] = None
    return [got.get(i) for i in range(len(exs))]
