"""Plan (grammar v2) <-> slot targets for architecture v2.

A plan is read as up to three blocks of slots (serialize.BLOCK_SLOTS): the main plan "m",
the inner plan "s" of one subquery filter, and the right-hand plan "o" of a set operation.

Targets per question:
  choice  "b.head" -> class           counts, distinct, order direction, limit kind, ...
  ptr     "b.slot" -> option id       one column / table per positional slot
  sets    "b.slot" -> [option ids]    filters, group by, required / via tables, join columns
  pair    (slot, option id, head, class)   aggregation / operator / date part / OR group / subquery
  vals    (slot, option id, [candidate idx])  the literal(s) of a filter, condition or HAVING
  limit   "b.limit" -> candidate idx  a LIMIT literal other than 1

`decode` is the inverse: scores -> Plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..sq.candidates import Candidate, ground
from ..sq.plan import (AGGS, COL_OPS, COMPOSE, GRAINS, LIMIT_KINDS, OPS, ORDER_DIRS, PARTS, ROWS, SET_OPS, Col,
                       Filter, Having, Plan, Proj)
from .serialize import BLOCK_SLOTS, GRAMMAR_V4

N_OR = 3  # OR-group ids per block: 0 = AND-ed, 1, 2
CHOICES = {  # every block
    "n_proj": tuple("1234"), "distinct": ("no", "yes"), "n_group": tuple("012"), "n_filter": tuple("01234"),
    "order_dir": ORDER_DIRS, "limit_kind": LIMIT_KINDS, "n_require": tuple("012"),
}
MAIN_CHOICES = {  # main block only
    "compose": COMPOSE, "grain": GRAINS, "cond_mask": ("none", "p0", "p1", "both"), "set_op": SET_OPS,
    "n_having": ("0", "1"), "n_via": tuple("012"), "n_join": tuple("0123456"),  # join columns come in edge pairs
}
PAIRS = {"agg": AGGS, "op": OPS, "part": PARTS, "or": tuple(str(i) for i in range(N_OR)), "sub": ("no", "yes")}
NO_VALUE_OPS = ("is_null", "not_null") + tuple(COL_OPS)
TWO_VALUE_OPS = ("between", "gt_lt", "ge_le_or", "gt_lt_or")
MAX_PROJ = {"m": 4, "s": 1, "o": 4}
MAX_GROUP = 2
if GRAMMAR_V4:  # grammar v4 heads: wider plans, OFFSET, a second ORDER BY key (column-compare ops are new
    # classes of the existing op head; their second column uses the block's x pointer slots)
    MAX_PROJ["m"], MAX_GROUP = 6, 4
    CHOICES["n_proj"] = tuple("123456")
    CHOICES["n_group"] = tuple("01234")
    CHOICES["offset"] = tuple("0123456789")
    CHOICES["order2_dir"] = ORDER_DIRS
    MAIN_CHOICES["n_join"] = tuple("012345678")  # up to 4 join conditions as ordered column pairs
    MAIN_CHOICES["cond2_mask"] = ("none", "p0", "p1", "both")  # grammar v4.1: second conditions (m.d0, m.d1)
    CHOICES["n_filter"] = tuple("0123456")  # grammar v4.1: up to 6 filters (the converter's limit)
N_EXTRA = {"m": 2, "s": 1, "o": 1}   # second-column ({B}) pointer slots per block: x0, x1
WRAPS = ("none", "avg", "sum", "max", "min", "count")

# Grammar v3 vocabulary (closed lists learned on training data, scripts/templates_v3.py). set_vocab must
# run before the model is built: head sizes depend on it, and checkpoints store it.
VOCAB: dict = {"templates": [""], "compose": list(COMPOSE), "nums": []}


def set_vocab(templates: list[str], nums: list) -> None:
    VOCAB["templates"] = [""] + sorted(t for t in templates if "{A}" in t)
    VOCAB["compose"] = list(COMPOSE) + sorted(t for t in templates if "{P0}" in t)
    VOCAB["nums"] = list(nums)
    nv = ("-",) + tuple(str(v) for v in nums)
    PAIRS["fn"] = tuple(VOCAB["templates"])
    for i in range(3):
        PAIRS[f"n{i}"] = nv
        MAIN_CHOICES[f"cn{i}"] = nv
    MAIN_CHOICES["compose"] = tuple(VOCAB["compose"])
    MAIN_CHOICES["compose_pos"] = tuple("0123")
    MAIN_CHOICES["wrap"] = WRAPS
    MAIN_CHOICES["wrap_idx"] = tuple("0123")


def _n_slots(t: str, letter: str) -> int:
    return sum("{%s%d}" % (letter, i) in t for i in range(4))


class Unsupported(Exception):
    """The plan uses something these slots cannot express (the row is skipped in training)."""


@dataclass
class Targets:
    choice: dict[str, int] = field(default_factory=dict)
    ptr: dict[str, str] = field(default_factory=dict)
    sets: dict[str, list[str]] = field(default_factory=dict)
    pair: list[tuple[str, str, str, int]] = field(default_factory=list)
    vals: list[tuple[str, str, list[int]]] = field(default_factory=list)
    limit: dict[str, int] = field(default_factory=dict)


def _agg(p) -> int:
    return AGGS.index("count" if p.col.column == ROWS else p.agg)


def _values(t: Targets, slot: str, col: Col, op: str, values: list, cands: list[Candidate]):
    if op in NO_VALUE_OPS:
        return
    idx = [ground(v, col.id, cands) for v in values]
    if not idx or any(i is None for i in idx):
        raise Unsupported("ungrounded literal")
    t.vals.append((slot, col.id, idx))


def _num_class(v) -> int:
    try:
        return VOCAB["nums"].index(v) + 1
    except ValueError:
        raise Unsupported("number not in vocabulary")


def _expr(t: Targets, b: str, slot: str, col: Col, cands: list[Candidate], extra: list):
    """Grammar v3 operand targets for (slot, column): template class, its numbers, its values (value
    roles 1..3 of the value head) and its second column (the block's next x slot)."""
    if not col.fn:
        if "fn" in PAIRS:
            t.pair.append((slot, col.id, "fn", 0))
        return
    if col.fn not in VOCAB["templates"]:
        raise Unsupported("template not in vocabulary")
    t.pair.append((slot, col.id, "fn", VOCAB["templates"].index(col.fn)))
    for i in range(3):
        t.pair.append((slot, col.id, f"n{i}", _num_class(col.nums[i]) if i < len(col.nums) else 0))
    for i, v in enumerate(col.vals):
        k = ground(v, col.id, cands)
        if k is None:
            raise Unsupported("ungrounded template value")
        t.vals.append((f"{slot}|t{i}", col.id, [k]))
    if col.col2:
        if len(extra) >= N_EXTRA[b]:
            raise Unsupported("too many second columns")
        t.ptr[f"{b}.x{len(extra)}"] = col.col2
        extra.append(col.col2)


def _uses_v4(p: Plan) -> bool:
    conds = [f for pr in p.projections for f in (pr.cond, pr.cond2) if f is not None]
    return bool(p.offset or p.order2 is not None or any(pr.cond2 is not None for pr in p.projections)
                or any(f.op in COL_OPS for f in p.filters + conds))


def _block(t: Targets, b: str, p: Plan, cands: list[Candidate]):
    if not GRAMMAR_V4 and _uses_v4(p):  # the v3 heads have no place for these (LUTHOR_GRAMMAR=v4 has)
        raise Unsupported("grammar v4 feature")
    if (len(p.projections) > MAX_PROJ[b] or len(p.group_by) > MAX_GROUP or len(p.filters) > len(CHOICES["n_filter"]) - 1
            or len(p.require) > 2):
        raise Unsupported(f"{b}: too many slots")
    if b != "m" and (p.having or p.set_op != "none" or p.compose != "none" or p.grain != "none"
                     or any(pr.cond for pr in p.projections) or any(f.sub for f in p.filters)):
        raise Unsupported(f"{b}: nested feature")
    c = t.choice
    c[f"{b}.n_proj"] = len(p.projections) - 1
    c[f"{b}.distinct"] = int(p.distinct)
    c[f"{b}.n_group"] = len(p.group_by)
    c[f"{b}.n_filter"] = len(p.filters)
    c[f"{b}.order_dir"] = ORDER_DIRS.index(p.order_dir if p.order is not None else "none")
    c[f"{b}.limit_kind"] = 0 if p.limit is None else (1 if p.limit == 1 else 2)
    c[f"{b}.n_require"] = len(p.require)
    extra: list = []
    if b != "m" and (p.wrap != "none" or p.compose_pos):
        raise Unsupported(f"{b}: nested feature")
    for i, pr in enumerate(p.projections):
        s = f"{b}.p{i}"
        t.ptr[s] = pr.col.id
        t.pair.append((s, pr.col.id, "agg", _agg(pr)))
        _expr(t, b, s, pr.col, cands, extra)
    if b == "m":  # conditions take x slots before filters (decode follows the same order)
        for i, pr in enumerate(p.projections[:2]):
            if pr.cond is not None:
                _expr(t, b, f"m.c{i}", pr.cond.col, cands, extra)
        for i, pr in enumerate(p.projections[:2]):
            if pr.cond2 is not None:
                _expr(t, b, f"m.d{i}", pr.cond2.col, cands, extra)
    fcols = [f.col.id for f in p.filters]
    if len(set(fcols)) != len(fcols):
        raise Unsupported("two filters on one column")
    t.sets[f"{b}.filter"] = fcols
    groups = sorted({f.group for f in p.filters if f.group})
    if len(groups) > N_OR - 1:
        raise Unsupported("too many OR groups")
    gmap = {g: i + 1 for i, g in enumerate(groups)}
    n_sub = 0
    for f in sorted(p.filters, key=lambda f: f.col.id):
        _expr(t, b, f"{b}.filter", f.col, cands, extra)
    for f in p.filters:
        s = f"{b}.filter"
        t.pair += [(s, f.col.id, "op", OPS.index(f.op)), (s, f.col.id, "part", PARTS.index(f.part)),
                   (s, f.col.id, "or", gmap.get(f.group, 0))]
        if b == "m":
            t.pair.append((s, f.col.id, "sub", int(f.sub is not None)))
        if f.sub is not None:
            n_sub += 1
            if n_sub > 1:
                raise Unsupported("two subqueries")
            _block(t, "s", f.sub, cands)
        else:
            _values(t, s, f.col, f.op, f.values, cands)
    if any(g.col2 for g in p.group_by):
        raise Unsupported("group by over two columns")
    for g in p.group_by:
        _expr(t, b, f"{b}.group", g, cands, extra)
    t.sets[f"{b}.group"] = [g.id for g in p.group_by]
    t.sets[f"{b}.require"] = [f"{r}.{ROWS}" for r in p.require]
    if p.order is not None:
        t.ptr[f"{b}.order"] = p.order.col.id
        t.pair.append((f"{b}.order", p.order.col.id, "agg", _agg(p.order)))
        _expr(t, b, f"{b}.order", p.order.col, cands, extra)
    if GRAMMAR_V4:
        c[f"{b}.offset"] = p.offset
        c[f"{b}.order2_dir"] = ORDER_DIRS.index(p.order2_dir if p.order2 is not None else "none")
        if p.order2 is not None:
            if "order2" not in BLOCK_SLOTS[b] or p.order is None:
                raise Unsupported(f"{b}: second order key")
            t.ptr[f"{b}.order2"] = p.order2.col.id
            t.pair.append((f"{b}.order2", p.order2.col.id, "agg", _agg(p.order2)))
            _expr(t, b, f"{b}.order2", p.order2.col, cands, extra)
        # column-vs-column comparisons: the second column takes the next x slot (conditions, then filters)
        cmp = [pr.cond for pr in p.projections if pr.cond is not None and pr.cond.op in COL_OPS]
        cmp += [pr.cond2 for pr in p.projections if pr.cond2 is not None and pr.cond2.op in COL_OPS]
        cmp += sorted((f for f in p.filters if f.op in COL_OPS), key=lambda f: f.col.id)
        for f in cmp:
            if len(extra) >= N_EXTRA[b]:
                raise Unsupported("too many second columns")
            t.ptr[f"{b}.x{len(extra)}"] = f.col.col2
            extra.append(f.col.col2)
    if c[f"{b}.limit_kind"] == 2:
        i = ground(p.limit, "", cands)
        if i is None:
            raise Unsupported("ungrounded limit")
        t.limit[f"{b}.limit"] = i
    if b != "m":
        return
    if p.compose not in MAIN_CHOICES["compose"]:
        raise Unsupported("compose template not in vocabulary")
    c["m.compose"] = MAIN_CHOICES["compose"].index(p.compose)
    if "compose_pos" in MAIN_CHOICES:
        if p.compose_pos > 3 or p.wrap_idx > 3:
            raise Unsupported("compose / wrap position")
        c["m.compose_pos"] = p.compose_pos
        for i in range(3):
            c[f"m.cn{i}"] = _num_class(p.compose_nums[i]) if i < len(p.compose_nums) else 0
        c["m.wrap"] = WRAPS.index(p.wrap)
        c["m.wrap_idx"] = p.wrap_idx
    elif p.compose_pos or p.wrap != "none" or p.compose_nums:
        raise Unsupported("grammar v3 compose / wrap without a vocabulary")
    c["m.grain"] = GRAINS.index(p.grain)
    if p.time_col is not None:
        t.ptr["m.time"] = p.time_col.id
    conds = [i for i, pr in enumerate(p.projections) if pr.cond is not None]
    if any(i > 1 for i in conds):
        raise Unsupported("condition on projection 3+")
    c["m.cond_mask"] = {(): 0, (0,): 1, (1,): 2, (0, 1): 3}[tuple(conds)]
    for i in conds:
        f = p.projections[i].cond
        s = f"m.c{i}"
        t.ptr[s] = f.col.id
        t.pair += [(s, f.col.id, "op", OPS.index(f.op)), (s, f.col.id, "part", PARTS.index(f.part))]
        _values(t, s, f.col, f.op, f.values, cands)
    if GRAMMAR_V4:
        conds2 = [i for i, pr in enumerate(p.projections) if pr.cond2 is not None]
        if any(i > 1 for i in conds2) or any(p.projections[i].cond is None for i in conds2):
            raise Unsupported("second condition on projection 3+")
        c["m.cond2_mask"] = {(): 0, (0,): 1, (1,): 2, (0, 1): 3}[tuple(conds2)]
        for i in conds2:
            f = p.projections[i].cond2
            s = f"m.d{i}"
            t.ptr[s] = f.col.id
            t.pair += [(s, f.col.id, "op", OPS.index(f.op)), (s, f.col.id, "part", PARTS.index(f.part))]
            _values(t, s, f.col, f.op, f.values, cands)
    if len(p.having) > 1:
        raise Unsupported("two HAVING conditions")
    c["m.n_having"] = len(p.having)
    for h in p.having:
        if h.col.fn:
            raise Unsupported("having on an expression")
        t.ptr["m.having"] = h.col.id
        t.pair += [("m.having", h.col.id, "agg", AGGS.index("count" if h.col.column == ROWS else h.agg)),
                   ("m.having", h.col.id, "op", OPS.index(h.op))]
        _values(t, "m.having", h.col, h.op, h.values, cands)
    if len(p.via) > 2 or len(p.join_cols) > len(MAIN_CHOICES["n_join"]) - 1:
        raise Unsupported("too many join hints")
    c["m.n_via"] = len(p.via)
    c["m.n_join"] = len(p.join_cols)
    t.sets["m.via"] = [f"{v}.{ROWS}" for v in p.via]
    if GRAMMAR_V4:  # ordered pairs: j0-j1, j2-j3, ... are the gold's join conditions
        for k, j in enumerate(p.join_cols):
            t.ptr[f"m.j{k}"] = j.id
    else:
        t.sets["m.join"] = [j.id for j in p.join_cols]
    c["m.set_op"] = SET_OPS.index(p.set_op)
    if p.set_op != "none":
        _block(t, "o", p.other, cands)


def build_targets(plan: Plan, cands: list[Candidate]) -> Targets:
    t = Targets()
    _block(t, "m", plan, cands)
    return t


def gold_options(t: Targets) -> set[str]:
    out = set(t.ptr.values())
    for v in t.sets.values():
        out |= set(v)
    return out


# ---- decoding --------------------------------------------------------------------------

class Scores:
    """What decode needs from the model for one question (implemented in model.py)."""

    option_ids: list[str]
    cands: list[Candidate]

    def choice(self, key: str) -> list[float]: ...
    def ptr(self, slot: str) -> list[float]: ...
    def pair(self, slot: str, opt: int, head: str) -> list[float]: ...
    def vals(self, slot: str, opt: int) -> list[float]: ...
    def limit(self, slot: str) -> list[float]: ...


# RL sampling (scripts/rl/rl_core.py) sets this to a list: every decision decode makes on a score vector
# that carries a `key` is recorded as (key, chosen indices in order, allowed indices or None, approximate). The
# plan's log-probability is then the sum over exactly these decisions. None (always, outside RL): no effect.
_TRACE: list | None = None


def _note(xs, order, allowed=None, approx: bool = False) -> None:
    if _TRACE is not None and order and getattr(xs, "key", None) is not None:
        _TRACE.append((xs.key, tuple(order), None if allowed is None else
                       tuple(i for i in range(len(xs)) if allowed(i)), approx))


def _argmax(xs) -> int:
    i = max(range(len(xs)), key=lambda i: xs[i])
    _note(xs, [i])
    return i


def _topk(xs, k: int, allowed) -> list[int]:
    idx = [i for i in range(len(xs)) if allowed(i)]
    out = sorted(idx, key=lambda i: -xs[i])[:k]
    _note(xs, out, allowed)
    return out


def _col(oid: str) -> Col:
    t, c = oid.split(".", 1)
    return Col(t, c)


def _pick_values(sc: Scores, slot: str, opt: int, op: str) -> list | None:
    if op in NO_VALUE_OPS:
        return []
    s = sc.vals(slot, opt)
    if not s:
        return None
    order = sorted(range(len(s)), key=lambda i: -s[i])
    if op in TWO_VALUE_OPS:
        if len(order) < 2:
            return None
        _note(s, order[:2])
        vs = [sc.cands[i].value for i in order[:2]]
        try:
            vs.sort(key=float)
        except (TypeError, ValueError):
            vs.sort(key=str)  # dates
        return vs
    if op in ("in", "not_in"):
        import math
        z = max(s)
        p = [math.exp(x - z) for x in s]
        tot = sum(p)
        chosen = [i for i in order if p[i] / tot >= 0.2][:6] or order[:1]
        _note(s, chosen, approx=len(chosen) > 1)  # the probability threshold itself is not modelled
        return [sc.cands[i].value for i in chosen]
    _note(s, order[:1])
    return [sc.cands[order[0]].value]


_PART_FIT = {"year": re.compile(r"^\d{4}$"), "year_month": re.compile(r"^\d{4}-\d{2}$"), "month": re.compile(r"^\d{1,2}$")}
HAVING_OPS = ("=", "!=", ">", ">=", "<", "<=", "between")


def _fit_part(part: str, vals: list) -> str:
    """A date part only makes sense with a literal of that shape ('2004' for year, '2004-08' for
    year_month). With a full date, compare the raw column; otherwise pick the part the literal fits."""
    if not vals:
        return part
    v = str(vals[0])
    if part != "none" and _PART_FIT[part].match(v):
        return part
    if part != "none" and re.match(r"^\d{4}-\d{2}-\d{2}", v):
        return "none"
    if part != "none":
        return next((k for k, rx in _PART_FIT.items() if rx.match(v)), "none")
    return part


def _dec_expr(sc: Scores, b: str, slot: str, o: int, xs: list) -> Col:
    """Grammar v3: the operand for option o at this slot (inverse of _expr). xs[0] counts x slots used."""
    ids = sc.option_ids
    base = _col(ids[o])
    if "fn" not in PAIRS or ids[o].endswith(f".{ROWS}"):
        return base
    fn = VOCAB["templates"][_argmax(sc.pair(slot, o, "fn"))]
    if not fn:
        return base
    nums = []
    for i in range(_n_slots(fn, "N")):
        k = _argmax_in(sc.pair(slot, o, f"n{i}"), lambda j: j > 0)
        nums.append(VOCAB["nums"][k - 1])
    vals = []
    for i in range(_n_slots(fn, "V")):
        sv = sc.vals(f"{slot}|t{i}", o)
        if not sv:
            return base
        vals.append(sc.cands[_argmax(sv)].value)
    col2 = ""
    if "{B}" in fn:
        if xs[0] >= N_EXTRA[b]:
            return base
        col2 = ids[_argmax_in(sc.ptr(f"{b}.x{xs[0]}"), lambda j: not ids[j].endswith(f".{ROWS}"))]
        xs[0] += 1
    return Col(base.table, base.column, fn, col2, tuple(vals), tuple(nums))


def _argmax_in(xs, allowed) -> int:
    i = max((i for i in range(len(xs)) if allowed(i)), key=lambda i: xs[i])
    _note(xs, [i], allowed)
    return i


def _decode_block(sc: Scores, b: str) -> Plan:
    ids = sc.option_ids
    is_rows = lambda i: ids[i].endswith(f".{ROWS}")
    ch = lambda k, names: names[_argmax(sc.choice(f"{b}.{k}"))]
    n_proj = min(MAX_PROJ[b], int(ch("n_proj", CHOICES["n_proj"])))
    cond_mask = ch("cond_mask", MAIN_CHOICES["cond_mask"]) if b == "m" else "none"
    projs = []
    used = set()
    xs = [0]
    for i in range(n_proj):
        s = f"{b}.p{i}"
        ps = sc.ptr(s)
        has_cond = cond_mask in (f"p{i}", "both")
        ranked = sorted(range(len(ps)), key=lambda j: -ps[j])
        for n_seen, o in enumerate(ranked, 1):  # best option whose (column, aggregation) is new
            agg = "count" if is_rows(o) else AGGS[_argmax(sc.pair(s, o, "agg"))]
            fn = _argmax(sc.pair(s, o, "fn")) if "fn" in PAIRS and not is_rows(o) else 0
            if (o, agg, fn, has_cond) not in used or has_cond:
                break
        _note(ps, ranked[:n_seen])  # the options passed over, in order, then the one taken
        used.add((o, agg, fn, has_cond))
        projs.append(Proj(agg, _dec_expr(sc, b, s, o, xs)))
    for i in range(min(2, n_proj)):  # conditions after all projections (the order targets use for x slots)
        if cond_mask in (f"p{i}", "both"):
            cs = f"m.c{i}"
            co = _argmax(sc.ptr(cs))
            op = OPS[_argmax(sc.pair(cs, co, "op"))]
            vals = _pick_values(sc, cs, co, op)
            if vals is not None:
                projs[i].cond = Filter(_dec_expr(sc, b, cs, co, xs), op, vals,
                                       _fit_part(PARTS[_argmax(sc.pair(cs, co, "part"))], vals))
    cond2_mask = ch("cond2_mask", MAIN_CHOICES["cond2_mask"]) if b == "m" and "cond2_mask" in MAIN_CHOICES else "none"
    for i in range(min(2, n_proj)):  # grammar v4.1: second conditions (after the first ones, as in targets)
        if cond2_mask in (f"p{i}", "both") and projs[i].cond is not None:
            cs = f"m.d{i}"
            co = _argmax(sc.ptr(cs))
            op = OPS[_argmax(sc.pair(cs, co, "op"))]
            vals = _pick_values(sc, cs, co, op)
            if vals is not None:
                projs[i].cond2 = Filter(_dec_expr(sc, b, cs, co, xs), op, vals,
                                        _fit_part(PARTS[_argmax(sc.pair(cs, co, "part"))], vals))
    p = Plan(projections=projs, distinct=ch("distinct", CHOICES["distinct"]) == "yes")
    n_filter = int(ch("n_filter", CHOICES["n_filter"]))
    fs = f"{b}.filter"
    used_sub = False
    chosen = _topk(sc.ptr(fs), n_filter, lambda i: not is_rows(i))
    fcol = {o: _dec_expr(sc, b, fs, o, xs) for o in sorted(chosen, key=lambda o: ids[o])}
    for o in chosen:
        op = OPS[_argmax(sc.pair(fs, o, "op"))]
        part = PARTS[_argmax(sc.pair(fs, o, "part"))]
        grp = _argmax(sc.pair(fs, o, "or"))
        if b == "m" and not used_sub and _argmax(sc.pair(fs, o, "sub")) == 1:
            used_sub = True
            p.filters.append(Filter(fcol[o], op, [], part, grp, _decode_block(sc, "s")))
            continue
        vals = _pick_values(sc, fs, o, op)
        if vals is not None:
            p.filters.append(Filter(fcol[o], op, vals, _fit_part(part, vals) if not fcol[o].fn else "none", grp))
    p.group_by = [_dec_expr(sc, b, f"{b}.group", o, [N_EXTRA[b]])
                  for o in _topk(sc.ptr(f"{b}.group"), int(ch("n_group", CHOICES["n_group"])), lambda i: not is_rows(i))]
    p.require = [ids[o].split(".", 1)[0] for o in _topk(sc.ptr(f"{b}.require"),
                                                        int(ch("n_require", CHOICES["n_require"])), is_rows)]
    od = ch("order_dir", ORDER_DIRS)
    if od != "none":
        o = _argmax(sc.ptr(f"{b}.order"))
        p.order = Proj("count" if is_rows(o) else AGGS[_argmax(sc.pair(f"{b}.order", o, "agg"))],
                       _dec_expr(sc, b, f"{b}.order", o, xs))
        p.order_dir = od
    if GRAMMAR_V4:
        p.offset = int(ch("offset", CHOICES["offset"]))
        od2 = ch("order2_dir", ORDER_DIRS)
        if od2 != "none" and p.order is not None and "order2" in BLOCK_SLOTS[b]:
            o = _argmax(sc.ptr(f"{b}.order2"))
            p.order2 = Proj("count" if is_rows(o) else AGGS[_argmax(sc.pair(f"{b}.order2", o, "agg"))],
                            _dec_expr(sc, b, f"{b}.order2", o, xs))
            p.order2_dir = od2
        cmp = [pr.cond for pr in p.projections if pr.cond is not None and pr.cond.op in COL_OPS]
        cmp += [pr.cond2 for pr in p.projections if pr.cond2 is not None and pr.cond2.op in COL_OPS]
        cmp += sorted((f for f in p.filters if f.op in COL_OPS), key=lambda f: f.col.id)
        for f in cmp:
            if xs[0] >= N_EXTRA[b]:
                f.op = "not_null"  # no slot left for the second column: degrade to a harmless filter
                continue
            o2 = _argmax_in(sc.ptr(f"{b}.x{xs[0]}"), lambda j: not is_rows(j))
            xs[0] += 1
            f.col = Col(f.col.table, f.col.column, col2=ids[o2])
    lk = ch("limit_kind", LIMIT_KINDS)
    if lk == "one":
        p.limit = 1
    elif lk == "literal":
        s = sc.limit(f"{b}.limit")
        if s:
            v = sc.cands[_argmax(s)].value
            try:
                p.limit = max(1, int(float(v)))
            except (TypeError, ValueError):
                p.limit = None
    if b != "m":
        return p
    p.compose = ch("compose", MAIN_CHOICES["compose"])
    if "compose_pos" in MAIN_CHOICES:
        p.compose_pos = int(ch("compose_pos", MAIN_CHOICES["compose_pos"]))
        if p.compose not in COMPOSE:
            for i in range(_n_slots(p.compose, "N")):
                k = _argmax_in(sc.choice(f"m.cn{i}"), lambda j: j > 0)
                p.compose_nums.append(VOCAB["nums"][k - 1])
        p.wrap = ch("wrap", MAIN_CHOICES["wrap"])
        p.wrap_idx = int(ch("wrap_idx", MAIN_CHOICES["wrap_idx"]))
    p.grain = ch("grain", GRAINS)
    if p.grain != "none":
        p.time_col = _col(ids[_argmax(sc.ptr("m.time"))])
    if ch("n_having", MAIN_CHOICES["n_having"]) == "1":
        o = _argmax(sc.ptr("m.having"))
        op = OPS[_argmax_in(sc.pair("m.having", o, "op"), lambda i: OPS[i] in HAVING_OPS)]
        vals = _pick_values(sc, "m.having", o, op)
        if vals is not None:
            agg = "count" if is_rows(o) else AGGS[_argmax(sc.pair("m.having", o, "agg"))]
            p.having = [Having(agg, _col(ids[o]), op, vals)]
    p.via = [ids[o].split(".", 1)[0] for o in _topk(sc.ptr("m.via"), int(ch("n_via", MAIN_CHOICES["n_via"])), is_rows)]
    if GRAMMAR_V4:
        p.join_cols = [_col(ids[_argmax_in(sc.ptr(f"m.j{k}"), lambda i: not is_rows(i))])
                       for k in range(int(ch("n_join", MAIN_CHOICES["n_join"])))]
    else:
        p.join_cols = [_col(ids[o]) for o in _topk(sc.ptr("m.join"), int(ch("n_join", MAIN_CHOICES["n_join"])),
                                                    lambda i: not is_rows(i))]
    p.set_op = ch("set_op", SET_OPS)
    if p.set_op != "none":
        p.other = _decode_block(sc, "o")
    return p


def decode(sc: Scores) -> Plan:
    return _decode_block(sc, "m")
