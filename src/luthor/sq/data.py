"""Training examples and batches for LuthorSQ.

A batch holds a few databases x a few questions each. Options are shared by the questions
of one database, so each option text is encoded once per step (with gradient: the option
side learns too, which is what makes unseen databases work). Per database the option set
is the gold options of its questions + structural hard negatives (the other columns of the
gold tables, same-named columns of other tables) + random fill, up to `max_opts`.

Targets are rebuilt from the plan every time (candidates are found and literals grounded
again), so twins and originals are labelled by the same code path as inference.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .candidates import Candidate, find_candidates, ground
from .model import CHOICES
from .plan import AGGS, OPS, PARTS, ROWS, Plan
from .schema_card import OPT_TEMPLATE, Q_TEMPLATE, SchemaCard, value_text

_C = {k: {v: i for i, v in enumerate(vs)} for k, vs in CHOICES.items()}


@dataclass
class Target:
    choice: dict[str, int]
    single: dict[str, int] = field(default_factory=dict)        # role -> option id (global "t.c")
    sets: dict[str, list[str]] = field(default_factory=dict)    # group / filter -> option ids
    agg: dict[str, int] = field(default_factory=dict)           # proj0..2 / order -> AGGS index
    op: dict[str, list[int]] = field(default_factory=dict)      # filter -> per filter; cond0/1 -> [op]
    part: dict[str, list[int]] = field(default_factory=dict)
    values: dict[str, list[list[int]]] = field(default_factory=dict)  # filter / cond0 / cond1 -> candidate idx lists
    limit: int | None = None                                   # candidate index of a LIMIT literal


def question_text(question: str, context: str = "") -> tuple[str, int]:
    """Model input and the char offset where the question body starts (for span alignment)."""
    body = question + (f"\nContext: {context}" if context else "")
    pre = Q_TEMPLATE.split("{q}")[0]
    return Q_TEMPLATE.format(q=body), len(pre)


def build_target(row: dict, cands: list[Candidate]) -> Target | None:
    if row["outcome"] != "plan":
        return Target(choice={"outcome": _C["outcome"]["unsupported"]})
    p = Plan.from_dict(row["plan"])
    if p.uses_v2():  # grammar v2 needs architecture v2's slots; the v1 model skips these rows
        return None
    ch = {"outcome": _C["outcome"]["plan"], "n_proj": len(p.projections) - 1, "compose": _C["compose"][p.compose],
          "distinct": int(p.distinct), "n_group": len(p.group_by), "n_filter": len(p.filters),
          "grain": _C["grain"][p.grain], "order_dir": _C["order_dir"][p.order_dir]}
    conds = [i for i, pr in enumerate(p.projections[:2]) if pr.cond is not None]
    if any(pr.cond is not None for pr in p.projections[2:]):
        return None
    ch["cond_mask"] = {(): 0, (0,): 1, (1,): 2, (0, 1): 3}[tuple(conds)]
    ch["limit_kind"] = 0 if p.limit is None else (1 if p.limit == 1 else 2)
    t = Target(choice=ch)
    for i, pr in enumerate(p.projections):
        t.single[f"proj{i}"] = pr.col.id
        t.agg[f"proj{i}"] = AGGS.index(pr.agg if pr.col.column != ROWS else "count")
        if pr.cond is not None:
            r = f"cond{i}"
            t.single[r] = pr.cond.col.id
            t.op[r] = [OPS.index(pr.cond.op)]
            t.part[r] = [PARTS.index(pr.cond.part)]
            idx = [ground(v, pr.cond.col.id, cands) for v in pr.cond.values]
            if any(x is None for x in idx):
                return None
            t.values[r] = [idx]
    t.sets["group"] = [g.id for g in p.group_by]
    t.sets["via"] = [f"{v}.{ROWS}" for v in p.via]
    t.sets["join"] = [c.id for c in p.join_cols]
    t.sets["filter"] = [f.col.id for f in p.filters]
    t.op["filter"] = [OPS.index(f.op) for f in p.filters]
    t.part["filter"] = [PARTS.index(f.part) for f in p.filters]
    t.values["filter"] = []
    for f in p.filters:
        idx = [ground(v, f.col.id, cands) for v in f.values]
        if any(x is None for x in idx):
            return None
        t.values["filter"].append(idx)
    if p.time_col is not None:
        t.single["time"] = p.time_col.id
    if p.order is not None:
        t.single["order"] = p.order.col.id
        t.agg["order"] = AGGS.index(p.order.agg if p.order.col.column != ROWS else "count")
    if ch["limit_kind"] == 2:
        t.limit = ground(p.limit, "", cands)
        if t.limit is None:
            return None
    return t


@dataclass
class Example:
    id: str
    db_id: str
    text: str
    body_offset: int
    cands: list[Candidate]
    target: Target
    twin_of: str | None = None
    twin_slot: str | None = None


def _trim_enumerated(cands: list[Candidate], row: dict, card: SchemaCard, max_enum: int) -> list[Candidate]:
    """Training only: keep every mentioned candidate and every candidate a gold literal needs,
    but at most `max_enum` of the enumerated (unmentioned) low-cardinality values; the rest
    are dropped before targets are built, so indices stay consistent."""
    gold_vals = set()
    if row.get("outcome") == "plan":
        p = Plan.from_dict(row["plan"])
        for f in p.filters + [pr.cond for pr in p.projections if pr.cond]:
            for v in f.values:
                gold_vals.add((f.col.id, str(v)))
    keep, extra = [], []
    for c in cands:
        if c.kind == "value" and c.span is None and (c.col, str(c.value)) not in gold_vals:
            extra.append(c)
        else:
            keep.append(c)
    rnd = random.Random(hash(row["id"]) & 0xFFFF)
    rnd.shuffle(extra)
    kept = set(map(id, extra[:max_enum]))
    return [c for c in cands if not (c.kind == "value" and c.span is None and id(c) not in kept
                                     and (c.col, str(c.value)) not in gold_vals)]


def load_examples(plans: Path, cards: dict[str, SchemaCard], drop_empty: bool = True,
                  max_enum: int | None = 24) -> list[Example]:
    out = []
    for line in open(plans):
        row = json.loads(line)
        if drop_empty and row.get("empty_result"):
            continue
        card = cards.get(row["db_id"])
        if card is None:
            continue
        text, off = question_text(row["question"], row.get("context", ""))
        body = text[off:]
        cands = find_candidates(body, card)
        if max_enum is not None:
            cands = _trim_enumerated(cands, row, card, max_enum)
        tgt = build_target(row, cands)
        if tgt is None:
            continue
        out.append(Example(row["id"], row["db_id"], text, off, cands, tgt, row.get("twin_of"), row.get("twin_slot")))
    return out


@dataclass
class DBBatch:
    db_id: str
    option_ids: list[str]
    option_texts: list[str]
    examples: list[Example]
    grad_opts: list[bool] | None = None  # options encoded WITH gradient in "partial" mode (gold + hard negatives)


def _structural_negatives(card: SchemaCard, gold: set[str]) -> list[str]:
    tables = {g.split(".", 1)[0] for g in gold}
    names = {g.split(".", 1)[1].lower() for g in gold}
    out = []
    for o in card.options():
        if o["id"] in gold:
            continue
        if o["table"] in tables or o["column"].lower() in names:
            out.append(o["id"])
    return out


def make_db_batch(card: SchemaCard, exs: list[Example], max_opts: int = 96, rnd: random.Random | None = None,
                  all_options: bool = False, grad_negs: int = 16) -> DBBatch:
    rnd = rnd or random.Random(0)
    opts = card.options()
    by_id = {o["id"]: o for o in opts}
    if all_options or len(opts) <= max_opts:
        chosen = [o["id"] for o in opts]
    else:
        gold = set()
        for e in exs:
            gold |= set(e.target.single.values())
            for v in e.target.sets.values():
                gold |= set(v)
        gold &= by_id.keys()
        neg = _structural_negatives(card, gold)
        rnd.shuffle(neg)
        rest = [o["id"] for o in opts if o["id"] not in gold and o["id"] not in neg]
        rnd.shuffle(rest)
        chosen = list(gold) + neg[: max(0, max_opts - len(gold))]
        chosen += rest[: max(0, max_opts - len(chosen))]
        order = {o["id"]: i for i, o in enumerate(opts)}
        chosen.sort(key=order.get)
    gold_ids = set()
    for e in exs:
        gold_ids |= set(e.target.single.values())
        for v in e.target.sets.values():
            gold_ids |= set(v)
    hard = set(_structural_negatives(card, gold_ids & set(chosen))[:grad_negs])
    grad = [(i in gold_ids) or (i in hard) for i in chosen]
    return DBBatch(card.db_id, chosen, [OPT_TEMPLATE.format(h=by_id[i]["text"]) for i in chosen], exs, grad)


def batches(examples: list[Example], cards: dict[str, SchemaCard], dbs_per_batch: int = 4, q_per_db: int = 4,
            max_opts: int = 96, seed: int = 0):
    """Yield lists of DBBatch. Twins are kept in the same DB group as their original."""
    rnd = random.Random(seed)
    by_db: dict[str, list[list[Example]]] = defaultdict(list)
    groups: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        groups[e.twin_of or e.id].append(e)
    for g in groups.values():
        by_db[g[0].db_id].append(g)
    chunks = []
    for db, gs in by_db.items():
        rnd.shuffle(gs)
        cur: list[Example] = []
        for g in gs:
            if cur and len(cur) + len(g) > q_per_db:
                chunks.append((db, cur))
                cur = []
            cur += g
        if cur:
            chunks.append((db, cur))
    rnd.shuffle(chunks)
    for i in range(0, len(chunks), dbs_per_batch):
        yield [make_db_batch(cards[db], exs, max_opts, rnd) for db, exs in chunks[i:i + dbs_per_batch]]


def value_candidate_texts(ex: Example) -> list[str]:
    return [OPT_TEMPLATE.format(h=value_text(c.col, str(c.value))) for c in ex.cands if c.kind == "value"]


def span_token_ranges(ex: Example, offsets: list[tuple[int, int]]) -> list[tuple[int, int] | None]:
    """Token range [s, e) where each candidate occurs in the question (None: not mentioned)."""
    out = []
    for c in ex.cands:
        if c.span is None:
            out.append(None)
            continue
        a, b = c.span[0] + ex.body_offset, c.span[1] + ex.body_offset
        toks = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
        out.append((toks[0], toks[-1] + 1) if toks else (0, 1))
    return out
