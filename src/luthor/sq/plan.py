"""The bounded query plan (schema-agnostic) and its deterministic compiler.

A plan names only *destinations*: which columns to project / aggregate / group / filter /
order by, with which operators and literals. It never names joins. The compiler finds the
tables the plan touches, connects them through the database's foreign-key graph (shortest
paths, a Steiner-tree approximation), and writes the SQL. So the network picks columns and
the code builds the bridges.

    SELECT [DISTINCT] <projections | compose(p0, p1)>
    FROM   <tables joined along FK edges>
    WHERE  <filters, AND-ed; filters sharing an OR group are OR-ed>
           (a filter may compare against an inner plan: IN / NOT IN / = / > ... (SELECT ...))
    GROUP BY <group_by> [, <time bucket>]
    HAVING <aggregate filters>
    ORDER BY <order_agg(order_col)> <order_dir>
    LIMIT  <limit>
    [UNION | INTERSECT | EXCEPT <another plan>]

Grammar v2 (2026-09-30): subqueries one level deep, HAVING, OR groups, set operations,
LIKE prefix/suffix. The v1 model does not predict these yet (architecture v2 will).

Grammar v4 (2026-10-06): OFFSET, a second ORDER BY key, column-vs-column filters (ops *_col, the
second column in Col.col2), up to 6 projections / 4 GROUP BY columns, and join-hint edges for tables
the FK graph does not connect. The converter also rewrites CTEs, outer queries over a derived table
and arithmetic over scalar subqueries into plans (see convert.V4).

Grammar v4.1 (round 2): a second AND-ed condition on a conditional aggregate (Proj.cond2, slots m.d0/m.d1),
SUM(<predicate>) as a conditional count, repeated compose parts share one {P} slot, one form for "now",
up to 6 filters in the slots, arithmetic STRFTIME templates compared as numbers (TEXT_FN_WHOLE), and more
value spans (candidates.VALUES_V41).
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any

import sqlglot
from sqlglot import exp

# Grammar switch, read once at import: LUTHOR_GRAMMAR=v4 turns grammar v4 on (converter constructs, join-hint
# routing, the v4 slot layout and heads in luthor.sq2). Default v3, so v3 plans, targets and checkpoints (R1, R2)
# reproduce bit for bit. Set it before importing luthor; a checkpoint trained under one grammar needs the same one.
GRAMMAR = os.environ.get("LUTHOR_GRAMMAR", "v3").strip().lower() or "v3"
if GRAMMAR not in ("v3", "v4"):
    raise ValueError(f"LUTHOR_GRAMMAR={GRAMMAR!r}: expected v3 or v4")
GRAMMAR_V4 = GRAMMAR == "v4"

AGGS = ("none", "count", "count_distinct", "sum", "avg", "min", "max")
OPS = ("=", "!=", ">", ">=", "<", "<=", "like", "not_like", "in", "not_in", "between",
       "is_null", "not_null", "like_prefix", "like_suffix",
       # grammar v3: two conditions on one column as one filter (values = [low, high] or [value])
       "gt_lt", "ge_le_or", "gt_lt_or", "eq_or_null")
# grammar v4: compare against a second column of the same row (Filter.col.col2), no literal. The compiler always
# understands them; they join OPS (the op heads' classes) only under v4, so v3 / v1 op heads keep their size.
COL_OPS = {"eq_col": "=", "ne_col": "!=", "gt_col": ">", "ge_col": ">=", "lt_col": "<", "le_col": "<="}
if GRAMMAR_V4:
    OPS = OPS + tuple(COL_OPS)
SET_OPS = ("none", "union", "intersect", "except")
PARTS = ("none", "year", "year_month", "month")  # compare a date part instead of the raw column
_PART_FMT = {"year": "%Y", "year_month": "%Y-%m", "month": "%m"}
COMPOSE = ("none", "ratio", "percent", "difference", "add")
GRAINS = ("none", "day", "week", "month", "quarter", "year")
ORDER_DIRS = ("none", "asc", "desc")
LIMIT_KINDS = ("none", "one", "literal")
_TEXT_FNS = {"SUBSTR", "SUBSTRING", "STRFTIME", "UPPER", "LOWER", "TRIM", "REPLACE", "LTRIM", "RTRIM"}


def _text_valued(fn: str) -> bool:
    """A template whose value is text: one text function call around the whole expression. Grammar v4.1:
    "STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', {A})" is arithmetic (a number), not text."""
    if fn.split("(")[0].upper() not in _TEXT_FNS:
        return False
    if not TEXT_FN_WHOLE:
        return True
    depth = 0
    for i, ch in enumerate(fn):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and ch == ")":
            return i == len(fn.rstrip()) - 1
    return True


TEXT_FN_WHOLE = GRAMMAR_V4  # grammar v4.1 fix (False = v3 / v4 behaviour, for ablations)
ROWS = "*"  # the pseudo-column "rows of this table" (COUNT(*))

MAX_PROJ, MAX_GROUP, MAX_FILTERS = 4, 2, 4
# grammar v4 limits (the converter's; the v1 model keeps the v3 ones above)
MAX_PROJ_V4, MAX_GROUP_V4 = 6, 4
# grammar v4: the compiler may join along the plan's join-hint columns when the FK graph has no route
# (BIRD databases often omit the foreign key the question needs). Hint edges cost more than declared and
# inferred ones, so they never displace an existing route.
JOIN_HINT_EDGES = GRAMMAR_V4
HINT_EDGE_COST = 5.0
# "disconnected": hint edges only between tables with no FK route (never changes a route that exists);
# "all": every two hinted columns of different tables may also be joined directly, cheaply (HINT_PAIR_COST,
#        before the hint bonus), so the gold's own join beats a different FK route;
# "pairs": like "all" but only consecutive hint pairs (j0-j1, j2-j3, ...), i.e. join columns kept as edges
#        (needs ordered join slots in the model; the v3 join slot is an unordered set)
# (v3 reads join columns as an unordered set, so pairing them would join random column pairs: "disconnected")
JOIN_HINT_MODE = "pairs" if GRAMMAR_V4 else "disconnected"
HINT_PAIR_COST = 0.7
# Join-route guard (grammar v4 fix, 2026-10-11; docs/paper_notes/research/grammar_v4_fix_2026-10-11.md), read once at
# import like LUTHOR_GRAMMAR. In "pairs" mode a hint pair (cost 0.7 - 0.6 bonus) beats a declared FK (1.0 - 0.6), so a
# wrongly predicted j-slot pair overrides a correct FK route. With a guard, a hint pair becomes a cheap edge (and an
# extra ON condition) only when the guard allows it (Graph.pair_allowed); other pairs are dropped as edges and their
# columns stay route hints (the v3 bonus). Every guard allows a pair that is an FK edge of the graph (declared or
# inferred) or whose two tables have no FK route at all (nothing to displace). Beyond that:
#   "same":   the two column names match after normalisation (case, punctuation)
#   "name":   the names link the two columns as one key (_name_link: same name, a column named after the other table
#             or <table><column>, a key-suffix match such as dest_address_id ~ address_id, a one/two-letter table
#             prefix such as p_partkey ~ l_partkey)
#   "direct": as a route edge: "name", or no FK edge joins the two tables directly (the pair may replace a multi-hop
#             route, never a one-hop FK between the same two tables); as an extra ON condition of a multi-column join
#             (extra_on, always between two tables the route already joins): "name"
# Default "off": v3 and v4 compile exactly as before (v3 never pairs hints, so no guard changes v3).
JOIN_GUARDS = ("off", "same", "name", "direct")
JOIN_GUARD = os.environ.get("LUTHOR_JOIN_GUARD", "off").strip().lower() or "off"
if JOIN_GUARD not in JOIN_GUARDS:
    raise ValueError(f"LUTHOR_JOIN_GUARD={JOIN_GUARD!r}: expected one of {JOIN_GUARDS}")


def _norm_name(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _name_link(ta: str, ca: str, tb: str, cb: str) -> bool:
    """Do the names say column ta.ca and column tb.cb hold the same key? Schema-agnostic, names only."""
    na, nb = _norm_name(ca), _norm_name(cb)
    if not na or not nb:
        return False
    if na == nb:
        return True

    def one_way(t_other: str, n_self: str, n_other: str) -> bool:
        t = _norm_name(t_other)
        forms = {t, t[:-1] if t.endswith("s") and len(t) > 3 else t}
        if n_self in forms or n_self in {f + n_other for f in forms}:  # Country -> country.Code; university_id -> university.id
            return True
        return len(n_other) >= 4 and n_self.endswith(n_other)  # dest_address_id -> address.address_id

    if one_way(tb, na, nb) or one_way(ta, nb, na):
        return True
    strip = lambda c: _norm_name(c.split("_", 1)[1]) if "_" in c and 1 <= len(c.split("_", 1)[0]) <= 2 else None
    sa, sb = strip(ca), strip(cb)  # TPC-H style p_partkey ~ l_partkey
    return bool(sa) and bool(sb) and len(sa) >= 4 and (sa == sb or sa == nb or sb == na)


@dataclass(frozen=True)
class Col:
    table: str
    column: str  # ROWS for COUNT(*) over the table
    # grammar v3 operand: an expression template over this column ({A}), an optional second column
    # ({B}, "table.column") and up to two string values ({V0}, {V1}), e.g. "SUBSTR({A}, 1, 4)" or
    # "CAST({A} AS REAL) / {B}". The model picks the template from a closed list learned on training data.
    fn: str = ""
    col2: str = ""
    vals: tuple = ()
    nums: tuple = ()  # numeric template constants {N0}, {N1}, ... (picked from a closed vocabulary)

    @property
    def id(self) -> str:
        return f"{self.table}.{self.column}"

    def second(self) -> "Col | None":
        if not self.col2:
            return None
        t, c = self.col2.split(".", 1)
        return Col(t, c)


def _col(x: dict | None) -> Col | None:
    return Col(**{**x, "vals": tuple(x.get("vals") or ()), "nums": tuple(x.get("nums") or ())}) if x else None


@dataclass
class Filter:
    col: Col
    op: str
    values: list[Any] = field(default_factory=list)
    part: str = "none"
    group: int = 0                 # 0 = AND-ed; filters sharing a group k > 0 are OR-ed together
    sub: "Plan | None" = None       # compare against an inner plan (one projection) instead of literals


@dataclass
class Having:
    agg: str
    col: Col
    op: str
    values: list[Any] = field(default_factory=list)


@dataclass
class Proj:
    agg: str
    col: Col
    cond: Filter | None = None  # conditional aggregate: only rows where cond holds
    cond2: Filter | None = None  # grammar v4.1: a second condition AND-ed with cond (CASE WHEN a AND b ...)


_V4_PLAN_DEFAULTS = {"offset": 0, "order2": None, "order2_dir": "none"}


def _drop_v4_defaults(d):
    """Plan.to_dict: leave out grammar v4 fields that hold their defaults, so a v3 plan serialises exactly as before
    v4 existed (from_dict fills them back in)."""
    if isinstance(d, list):
        return [_drop_v4_defaults(x) for x in d]
    if not isinstance(d, dict):
        return d
    out = {k: _drop_v4_defaults(v) for k, v in d.items()
           if not (k in _V4_PLAN_DEFAULTS and v == _V4_PLAN_DEFAULTS[k]) and not (k == "cond2" and v is None)}
    return out


@dataclass
class Plan:
    projections: list[Proj]
    compose: str = "none"   # a COMPOSE name, or (grammar v3) a template over {P0}..{P2}
    compose_pos: int = 0      # index of the first projection the compose consumes
    compose_nums: list = field(default_factory=list)  # {N0}.. of a compose template
    distinct: bool = False
    filters: list[Filter] = field(default_factory=list)
    group_by: list[Col] = field(default_factory=list)
    grain: str = "none"
    time_col: Col | None = None
    order_dir: str = "none"
    order: Proj | None = None
    limit: int | None = None
    # join hints (never join conditions): tables the join must pass through, and columns
    # whose FK edges it should prefer. The compiler still builds every join itself.
    via: list[str] = field(default_factory=list)
    join_cols: list[Col] = field(default_factory=list)
    having: list[Having] = field(default_factory=list)
    # tables the query must join although no column of theirs is used: existence ("students who
    # have a pet") or multiplicity (COUNT(*) over the join). Never pruned, unlike `via`.
    require: list[str] = field(default_factory=list)
    set_op: str = "none"
    other: "Plan | None" = None    # the right-hand plan of a set operation
    # grammar v3: an aggregate over one projection of this plan as a derived table
    # (AVG of per-group counts, ...): SELECT wrap(c<wrap_idx>) FROM (<this plan>)
    wrap: str = "none"
    wrap_idx: int = 0
    # grammar v4: OFFSET (LIMIT n OFFSET k; "the second most ..."), and a second ORDER BY key
    offset: int = 0
    order2: Proj | None = None
    order2_dir: str = "none"

    def uses_v2(self) -> bool:
        """True when the plan needs grammar v2 (the v1 model cannot predict it)."""
        return bool(self.having or self.set_op != "none" or len(self.projections) > 3 or self.require
                    or any(f.sub is not None or f.group or f.op in ("like_prefix", "like_suffix")
                           for f in self.filters))

    def to_dict(self) -> dict:
        return _drop_v4_defaults(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        c = _col
        flt = lambda f: Filter(c(f["col"]), f["op"], list(f.get("values") or []), f.get("part", "none"),
                               f.get("group", 0), cls.from_dict(f["sub"]) if f.get("sub") else None) if f else None
        return cls(
            projections=[Proj(p["agg"], c(p["col"]), flt(p.get("cond")), flt(p.get("cond2"))) for p in d["projections"]],
            compose=d.get("compose", "none"), compose_pos=d.get("compose_pos", 0),
            compose_nums=list(d.get("compose_nums") or []), distinct=d.get("distinct", False),
            filters=[flt(f) for f in d.get("filters") or []],
            group_by=[c(g) for g in d.get("group_by") or []],
            grain=d.get("grain", "none"), time_col=c(d.get("time_col")),
            order_dir=d.get("order_dir", "none"),
            order=Proj(d["order"]["agg"], c(d["order"]["col"])) if d.get("order") else None,
            limit=d.get("limit"),
            via=list(d.get("via") or []),
            join_cols=[c(j) for j in d.get("join_cols") or []],
            having=[Having(h["agg"], c(h["col"]), h["op"], list(h.get("values") or [])) for h in d.get("having") or []],
            require=list(d.get("require") or []),
            set_op=d.get("set_op", "none"),
            other=cls.from_dict(d["other"]) if d.get("other") else None,
            wrap=d.get("wrap", "none"), wrap_idx=d.get("wrap_idx", 0),
            offset=d.get("offset", 0) or 0,
            order2=Proj(d["order2"]["agg"], c(d["order2"]["col"])) if d.get("order2") else None,
            order2_dir=d.get("order2_dir", "none"),
        )

    def columns(self) -> list[Col]:
        cols = [p.col for p in self.projections] + [p.cond.col for p in self.projections if p.cond]
        cols += [p.cond2.col for p in self.projections if p.cond2]
        cols += [f.col for f in self.filters] + list(self.group_by) + [h.col for h in self.having]
        if self.time_col:
            cols.append(self.time_col)
        if self.order:
            cols.append(self.order.col)
            if self.order.cond:
                cols.append(self.order.cond.col)
        if self.order2:
            cols.append(self.order2.col)
        return cols + [c.second() for c in cols if c.col2]

    def tables(self) -> list[str]:
        return list(dict.fromkeys(c.table for c in self.columns()))


# ---- schema graph -------------------------------------------------------------------------
@dataclass
class Graph:
    """Tables, their columns, and FK edges (t1, c1, t2, c2), undirected for path finding.
    `inferred` edges (guessed from column names) cost more than declared ones, so they are
    used only when no declared route exists."""
    columns: dict[str, list[str]]
    edges: list[tuple[str, str, str, str]]
    dialect: str = "sqlite"
    inferred: set = field(default_factory=set)

    DECLARED, INFERRED, HINT_BONUS = 1.0, 3.0, 0.6

    def __post_init__(self):
        self.adj: dict[str, list[tuple[str, str, str, float]]] = {t: [] for t in self.columns}
        self._edge_set = {tuple(e) for e in self.edges}
        inf = {tuple(e) for e in self.inferred} | {(c, d, a, b) for a, b, c, d in self.inferred}
        for t1, c1, t2, c2 in self.edges:
            if t1 in self.adj and t2 in self.adj and t1 != t2:
                w = self.INFERRED if (t1, c1, t2, c2) in inf else self.DECLARED
                self.adj[t1].append((t2, c1, c2, w))
                self.adj[t2].append((t1, c2, c1, w))

    def component(self, t: str) -> set[str]:
        seen, todo = {t}, [t]
        while todo:
            n = todo.pop()
            for m, *_ in self.adj.get(n, []):
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        return seen

    def with_hint_edges(self, cols: list["Col"], cost: float) -> "Graph":
        """Grammar v4: a copy that may also join any two hinted columns of tables the FK graph does not
        connect at all (so an existing route is never displaced). The hints come from the plan's join
        columns (gold ON / WHERE equalities when converting; the model's join slot when decoding)."""
        ok = lambda a, b: a.table != b.table and a.table in self.adj and b.table in self.adj
        pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:] if ok(a, b)]
        cheap = []
        if JOIN_HINT_MODE == "all":
            cheap = pairs
        elif JOIN_HINT_MODE == "pairs":
            cheap = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1, 2) if ok(cols[i], cols[i + 1])]
            if JOIN_GUARD != "off":
                cheap = [(a, b) for a, b in cheap if self.pair_allowed(a, b)]
        far = [(a, b) for a, b in pairs if b.table not in self.component(a.table) and (a, b) not in cheap]
        if not far and not cheap:
            return self
        g = Graph(self.columns, self.edges, self.dialect, self.inferred)
        for (a, b), w in [(p, cost) for p in far] + [(p, HINT_PAIR_COST) for p in cheap]:
            g.adj[a.table].append((b.table, a.column, b.column, w))
            g.adj[b.table].append((a.table, b.column, a.column, w))
        return g

    def pair_allowed(self, a: "Col", b: "Col", guard: str | None = None, extra: bool = False) -> bool:
        """Join guard (LUTHOR_JOIN_GUARD, see JOIN_GUARDS): may the hint pair a = b be its own join edge (extra=False)
        or an extra ON condition of a multi-column join (extra=True)?"""
        guard = guard or JOIN_GUARD
        if guard == "off":
            return True
        if (a.table, a.column, b.table, b.column) in self._edge_set or (b.table, b.column, a.table, a.column) in self._edge_set:
            return True
        if b.table not in self.component(a.table):
            return True
        if guard == "same":
            return _norm_name(a.column) == _norm_name(b.column)
        if _name_link(a.table, a.column, b.table, b.column):
            return True
        if guard == "direct" and not extra:
            return not any(m == b.table for m, *_ in self.adj.get(a.table, []))
        return False

    def path(self, a: str, b: str, hints: set[tuple[str, str]] | None = None) -> tuple[float, list] | None:
        """Cheapest FK path a -> b (Dijkstra). An edge touching a hinted (table, column) is
        cheaper. Ties break deterministically on names. Returns (cost, edges) or None."""
        import heapq

        if a == b:
            return 0.0, []
        hints = hints or set()
        best = {a: 0.0}
        heap = [(0.0, a, [])]
        while heap:
            cost, n, path = heapq.heappop(heap)
            if n == b:
                return cost, path
            if cost > best.get(n, float("inf")):
                continue
            for m, cn, cm, w in sorted(self.adj.get(n, [])):
                hit = ((n, cn) in hints) + ((m, cm) in hints)
                if hit:  # both ends hinted beats one end (parallel FKs to the same key, e.g. eye/hair/skin colour)
                    w = w - self.HINT_BONUS * (0.5 + 0.5 * (hit == 2))
                nc = cost + w
                if nc < best.get(m, float("inf")) - 1e-9:
                    best[m] = nc
                    heapq.heappush(heap, (nc, m, path + [(n, cn, m, cm)]))
        return None


class CompileError(ValueError):
    pass


def join_tree(graph: Graph, tables: list[str], via: list[str] | None = None,
              hints: set[tuple[str, str]] | None = None) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    """Greedy Steiner tree over the plan's tables plus any `via` tables: grow from the first
    table, each time attaching the remaining table with the cheapest FK path to the tree.
    Returns (all tables in order, join edges)."""
    need = list(dict.fromkeys(list(tables) + [t for t in (via or []) if t in graph.columns]))
    if not need:
        raise CompileError("plan touches no table")
    in_tree = [need[0]]
    edges: list[tuple[str, str, str, str]] = []
    todo = [t for t in need[1:]]
    while todo:
        best = None
        for t in todo:
            if t in in_tree:
                best = (t, 0.0, [])
                break
            for s in in_tree:
                r = graph.path(s, t, hints)
                if r is not None and (best is None or r[0] < best[1] - 1e-9):
                    best = (t, r[0], r[1])
        if best is None:
            raise CompileError(f"no foreign-key path to {todo[0]}")
        t, _, p = best
        for e in p:
            if e[2] not in in_tree:
                in_tree.append(e[2])
                edges.append(e)
        todo.remove(t)
    # a hinted bridge table can never be a dead end: prune via-only leaves (they would only add
    # a join that multiplies rows, e.g. AVG over a spurious one-to-many table)
    required = set(tables)
    while True:
        deg = {t: 0 for t in in_tree}
        for a, _, b, _ in edges:
            deg[a] += 1
            deg[b] += 1
        leaves = [t for t in in_tree if deg[t] <= 1 and t not in required and len(in_tree) > 1]
        if not leaves:
            break
        in_tree = [t for t in in_tree if t not in leaves]
        edges = [e for e in edges if e[0] not in leaves and e[2] not in leaves]
    return in_tree, edges


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def _bucket(expr: str, grain: str, dialect: str) -> str:
    if dialect == "sqlite":
        return {"day": f"DATE({expr})", "week": f"STRFTIME('%Y-%W', {expr})",
                "month": f"STRFTIME('%Y-%m', {expr})", "year": f"STRFTIME('%Y', {expr})",
                "quarter": f"(STRFTIME('%Y', {expr}) || '-Q' || ((CAST(STRFTIME('%m', {expr}) AS INTEGER) + 2) / 3))"}[grain]
    return f"DATE_TRUNC('{grain}', {expr})"


def compile_plan(plan: Plan, graph: Graph, _alias: str = "T", _top: bool = True) -> str:
    """Plan -> SQL in the graph's dialect (written as SQLite-style SQL, then transpiled).
    Inner plans (subqueries) and the right side of a set operation compile recursively with
    their own alias prefix and join tree."""
    if plan.wrap != "none":
        inner = Plan.from_dict(plan.to_dict())
        inner.wrap = "none"
        isql = compile_plan(inner, graph, _alias + "W", _top=False)
        isql = _alias_outputs(isql)
        sql = f"SELECT {plan.wrap.upper()}(c{plan.wrap_idx}) FROM ({isql}) AS {_alias}W"
        return sqlglot.transpile(sql, read="sqlite", write=graph.dialect)[0] if _top and graph.dialect != "sqlite" else sql
    if plan.set_op != "none":
        if plan.other is None:
            raise CompileError("set operation without a second plan")
        left = compile_plan(_strip_set(plan), graph, _alias, _top=False)
        right = compile_plan(plan.other, graph, _alias + "R", _top=False)
        sql = f"{left} {plan.set_op.upper()} {right}"
        return sqlglot.transpile(sql, read="sqlite", write=graph.dialect)[0] if _top and graph.dialect != "sqlite" else sql
    for c in plan.columns():
        if c.table not in graph.columns:
            raise CompileError(f"unknown table {c.table}")
        if c.column != ROWS and c.column not in graph.columns[c.table]:
            raise CompileError(f"unknown column {c.id}")
    need = plan.tables() + [t for t in plan.require if t in graph.columns and t not in plan.tables()]
    jg = graph.with_hint_edges(plan.join_cols, HINT_EDGE_COST) if JOIN_HINT_EDGES and plan.join_cols else graph
    tables, edges = join_tree(jg, need, plan.via, {(c.table, c.column) for c in plan.join_cols})
    alias = {t: f"{_alias}{i + 1}" for i, t in enumerate(tables)}
    multi = len(tables) > 1
    extra_on: dict = {}  # (t1, t2) -> column pairs the join hints give for that table pair
    if JOIN_HINT_EDGES and JOIN_HINT_MODE == "pairs":
        jc = plan.join_cols
        for i in range(0, len(jc) - 1, 2):
            a, b = jc[i], jc[i + 1]
            if a.table != b.table and a.table in graph.columns and b.table in graph.columns \
                    and (JOIN_GUARD == "off" or graph.pair_allowed(a, b, extra=True)):
                extra_on.setdefault((a.table, b.table), []).append((a.column, b.column))
                extra_on.setdefault((b.table, a.table), []).append((b.column, a.column))
        extra_on = {k: v for k, v in extra_on.items() if len(v) > 1}  # only table pairs joined on 2+ columns

    def ref(c: Col) -> str:
        r = f"{alias[c.table]}.{_q(c.column)}" if multi else _q(c.column)
        if not c.fn:
            return r
        out = c.fn.replace("{A}", r)
        if c.col2:
            out = out.replace("{B}", ref(c.second()))
        for i, v in enumerate(c.vals):
            out = out.replace("{V%d}" % i, _lit(v))
        for i, v in enumerate(c.nums):
            out = out.replace("{N%d}" % i, repr(v))
        return out

    def agg(p: Proj) -> str:
        if p.cond is not None:
            c = cond(p.cond)
            if p.cond2 is not None:
                c = f"{c} AND {cond(p.cond2)}"
            if p.col.column == ROWS or p.agg == "count":
                return f"SUM(CASE WHEN {c} THEN 1 ELSE 0 END)"
            inner = f"CASE WHEN {c} THEN {ref(p.col)} END"
            return {"sum": f"SUM({inner})", "avg": f"AVG({inner})", "min": f"MIN({inner})", "max": f"MAX({inner})",
                    "count_distinct": f"COUNT(DISTINCT {inner})"}[p.agg]
        if p.col.column == ROWS:
            return "COUNT(*)"
        r = ref(p.col)
        return {"none": r, "count": f"COUNT({r})", "count_distinct": f"COUNT(DISTINCT {r})",
                "sum": f"SUM({r})", "avg": f"AVG({r})", "min": f"MIN({r})", "max": f"MAX({r})"}[p.agg]

    def cond(f: Filter) -> str:
        r, v = ref(f.col), f.values
        if f.sub is not None:
            inner = compile_plan(f.sub, graph, _alias + "S", _top=False)
            if f.op in ("in", "not_in"):
                return f"{r} {'NOT ' if f.op == 'not_in' else ''}IN ({inner})"
            if f.op in ("=", "!=", ">", ">=", "<", "<="):
                return f"{r} {f.op} ({inner})"
            raise CompileError(f"op {f.op} with a subquery")
        if f.op in COL_OPS:  # grammar v4: column vs column of the same row
            if f.col.fn or not f.col.col2:
                raise CompileError(f"op {f.op} needs a plain column and a second column")
            return f"{r} {COL_OPS[f.op]} {ref(f.col.second())}"
        if f.col.fn and _text_valued(f.col.fn):  # text-valued template: compare as text
            v = [str(int(x)) if isinstance(x, float) and x.is_integer() else str(x) for x in v]
        if f.part != "none":
            fmt = _PART_FMT[f.part]
            r = f"STRFTIME('{fmt}', {r})" if graph.dialect == "sqlite" else f"TO_CHAR({r}, '{fmt}')"
            v = [str(x) for x in v]
        if f.op in ("=", "!=", ">", ">=", "<", "<="):
            return f"{r} {f.op} {_lit(v[0])}"
        if f.op in ("like", "not_like"):
            return f"{r} {'NOT ' if f.op == 'not_like' else ''}LIKE {_lit('%' + str(v[0]) + '%')}"
        if f.op == "like_prefix":
            return f"{r} LIKE {_lit(str(v[0]) + '%')}"
        if f.op == "like_suffix":
            return f"{r} LIKE {_lit('%' + str(v[0]))}"
        if f.op in ("in", "not_in"):
            return f"{r} {'NOT ' if f.op == 'not_in' else ''}IN ({', '.join(map(_lit, v))})"
        if f.op == "between":
            return f"{r} BETWEEN {_lit(v[0])} AND {_lit(v[1])}"
        if f.op == "gt_lt":
            return f"{r} > {_lit(v[0])} AND {r} < {_lit(v[1])}"
        if f.op == "ge_le_or":  # outside [low, high], bounds included
            return f"({r} <= {_lit(v[0])} OR {r} >= {_lit(v[1])})"
        if f.op == "gt_lt_or":
            return f"({r} < {_lit(v[0])} OR {r} > {_lit(v[1])})"
        if f.op == "eq_or_null":
            return f"({r} = {_lit(v[0])} OR {r} IS NULL)"
        if f.op == "is_null":
            return f"{r} IS NULL"
        if f.op == "not_null":
            return f"{r} IS NOT NULL"
        raise CompileError(f"unknown op {f.op}")

    if not plan.projections:
        raise CompileError("no projection")
    exprs = [agg(p) for p in plan.projections]
    if plan.compose != "none":
        k, pos = plan.compose_pos, plan.compose_pos
        if plan.compose in COMPOSE:
            if len(exprs) < pos + 2:
                raise CompileError("compose needs two projections")
            a, b = exprs[pos], exprs[pos + 1]
            comp = {"ratio": f"CAST({a} AS REAL) / {b}", "percent": f"CAST({a} AS REAL) * 100 / {b}",
                    "difference": f"{a} - {b}", "add": f"{a} + {b}"}[plan.compose]
            k = 2
        else:  # grammar v3 template
            k = sum("{P%d}" % i in plan.compose for i in range(3))
            if len(exprs) < pos + k:
                raise CompileError("compose needs more projections")
            comp = plan.compose
            for i in range(k):
                comp = comp.replace("{P%d}" % i, exprs[pos + i])
            for i, v in enumerate(plan.compose_nums):
                comp = comp.replace("{N%d}" % i, repr(v))
        exprs = exprs[:pos] + [comp] + exprs[pos + k:]
    bucket = None
    if plan.grain != "none":
        if plan.time_col is None:
            raise CompileError("time grain without a time column")
        bucket = _bucket(ref(plan.time_col), plan.grain, graph.dialect)
        exprs = [bucket] + exprs

    sql = "SELECT " + ("DISTINCT " if plan.distinct else "") + ", ".join(exprs)
    sql += f" FROM {_q(tables[0])}" + (f" AS {alias[tables[0]]}" if multi else "")
    for (t1, c1, t2, c2) in edges:
        on = f"{alias[t1]}.{_q(c1)} = {alias[t2]}.{_q(c2)}"
        for a, b in extra_on.get((t1, t2), []):  # grammar v4: the other columns of a multi-column join
            if (a, b) != (c1, c2):
                on += f" AND {alias[t1]}.{_q(a)} = {alias[t2]}.{_q(b)}"
        sql += f" INNER JOIN {_q(t2)} AS {alias[t2]} ON {on}"

    conds, groups = [], {}
    for f in plan.filters:
        if f.group:
            groups.setdefault(f.group, []).append(cond(f))
        else:
            conds.append(cond(f))
    conds += ["(" + " OR ".join(g) + ")" for _, g in sorted(groups.items())]
    if conds:
        sql += " WHERE " + " AND ".join(conds)

    group = [ref(g) for g in plan.group_by]
    if bucket:
        group = [bucket] + group
    has_agg = any(p.agg != "none" or p.col.column == ROWS for p in plan.projections)
    if has_agg:  # bare projected columns next to aggregates are grouping keys
        for p in plan.projections:
            if p.agg == "none" and p.col.column != ROWS and ref(p.col) not in group:
                group.append(ref(p.col))
    if group:
        sql += " GROUP BY " + ", ".join(group)
    if plan.having:
        hs = []
        for h in plan.having:
            e = agg(Proj(h.agg, h.col))
            if h.op == "between":
                hs.append(f"{e} BETWEEN {_lit(h.values[0])} AND {_lit(h.values[1])}")
            else:
                hs.append(f"{e} {h.op} {_lit(h.values[0])}")
        sql += " HAVING " + " AND ".join(hs)
    if plan.order_dir != "none" and plan.order is not None:
        sql += f" ORDER BY {agg(plan.order)} {plan.order_dir.upper()}"
        if plan.order2 is not None and plan.order2_dir != "none":
            sql += f", {agg(plan.order2)} {plan.order2_dir.upper()}"
    if plan.limit is not None:
        sql += f" LIMIT {int(plan.limit)}"
    if plan.offset:
        sql += ("" if plan.limit is not None else " LIMIT -1") + f" OFFSET {int(plan.offset)}"
    if _top and graph.dialect != "sqlite":
        sql = sqlglot.transpile(sql, read="sqlite", write=graph.dialect)[0]
    return sql


def merge_same_column(plan: "Plan") -> None:
    """Grammar v3: fold two filters on one column into one (so a set-valued filter slot holds each column
    once): x > a AND x < b -> gt_lt; x <= a OR x >= b -> ge_le_or; x < a OR x > b -> gt_lt_or;
    x >= a AND x <= b -> between; x = a OR x IS NULL -> eq_or_null. Others are left as they are."""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    by: dict = {}
    for f in plan.filters:
        if f.sub is None:
            by.setdefault((f.col, f.part, f.group), []).append(f)
    for (col, part, grp), fs in by.items():
        if len(fs) != 2:
            continue
        a, b = fs
        ops = {a.op: a, b.op: b}
        new = None
        if not grp and {">", "<"} == set(ops) or not grp and {">=", "<="} == set(ops):
            lo, hi = (ops.get(">") or ops.get(">=")), (ops.get("<") or ops.get("<="))
            if num(lo.values[0]) is not None and num(hi.values[0]) is not None:
                new = Filter(col, "gt_lt" if ">" in ops else "between", [lo.values[0], hi.values[0]], part, 0)
        elif grp and ({"<=", ">="} == set(ops) or {"<", ">"} == set(ops)):
            lo, hi = (ops.get("<=") or ops.get("<")), (ops.get(">=") or ops.get(">"))
            if num(lo.values[0]) is not None and num(hi.values[0]) is not None and num(lo.values[0]) <= num(hi.values[0]):
                new = Filter(col, "ge_le_or" if "<=" in ops else "gt_lt_or", [lo.values[0], hi.values[0]], part, 0)
        elif grp and {"=", "is_null"} == set(ops):
            new = Filter(col, "eq_or_null", list(ops["="].values), part, 0)
        if new is not None:
            if grp and sum(f.group == grp for f in plan.filters) != 2:
                continue  # the OR group holds other filters too
            i = plan.filters.index(a)
            plan.filters = [f for f in plan.filters if f is not a and f is not b]
            plan.filters.insert(i, new)


def _alias_outputs(sql: str) -> str:
    """Name a SELECT's output columns c0, c1, ... (for an outer query over it)."""
    t = sqlglot.parse_one(sql, read="sqlite")
    t.set("expressions", [exp.alias_(e.unalias() if isinstance(e, exp.Alias) else e, f"c{i}")
                          for i, e in enumerate(t.expressions)])
    return t.sql(dialect="sqlite")


def _strip_set(p: Plan) -> Plan:
    q = Plan.from_dict(p.to_dict())
    q.set_op, q.other = "none", None
    return q
