"""Gold SQL (BIRD / Spider / SynSQL) -> bounded Plan, or a reason it cannot be one.

The converter is deliberately strict: anything outside the plan grammar (subqueries, set
operations, CTEs, window functions, HAVING, CASE, OR across columns, arithmetic inside an
aggregate, self-joins, OFFSET, ...) is reported as unsupported with a short reason. The
prepare script then compiles every converted plan and keeps it only if it returns the same
rows as the gold SQL, so a label is never trusted on the parser's word alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .plan import (merge_same_column, COL_OPS, GRAMMAR_V4, MAX_FILTERS, MAX_GROUP, MAX_GROUP_V4, MAX_PROJ,
                   MAX_PROJ_V4, ROWS, Col, Filter, Graph, Having, Plan, Proj)


class Unsupported(Exception):
    pass


# Grammar v4 constructs, each switchable (an empty set gives grammar v3 behaviour, for ablations):
#   cte      inline non-recursive WITH clauses as derived tables (then the derived-table rules apply)
#   flatten  an outer query over one derived table with WHERE / ORDER BY / LIMIT / aggregates is merged
#            into the inner query (incl. BIRD's "SELECT T FROM (SELECT CASE WHEN c THEN x END AS T ...)
#            WHERE T IS NOT NULL" idiom); COUNT(*) over a grouped derived table becomes wrap=count
#   offset   LIMIT n OFFSET k / LIMIT k, n                          (Plan.offset)
#   order2   a second ORDER BY key                                   (Plan.order2, order2_dir)
#   colcmp   column vs column of the same row (a = b, a > b, ...)    (ops eq_col.. with Col.col2)
#   wide     up to 6 projections and 4 GROUP BY columns
#   scalar   arithmetic over scalar subqueries on the same tables -> conditional aggregates (one query)
#   (grammar v4.1, round 2)
#   cond2    CASE WHEN a AND b THEN .. / IIF(a AND b, ..) inside an aggregate: two structured conditions
#            (Proj.cond, Proj.cond2) instead of an expression template over the whole CASE
#   boolsum  SUM(<predicate>) (SQLite booleans) -> a conditional count
#   dedup    a compose template that repeats an aggregate reuses its {P} slot ("(P0 - P1) * 100 / P1")
#   now      'now' / DATE('now') / CURRENT_DATE inside a date-only STRFTIME -> CURRENT_TIMESTAMP (one form)
V4_ALL = frozenset({"cte", "flatten", "scalar", "offset", "order2", "colcmp", "wide",
                    "cond2", "boolsum", "dedup", "now"})
# Default: grammar v3 (V4 empty), so v3 data and checkpoints reproduce exactly; LUTHOR_GRAMMAR=v4 turns every
# construct on (plan.GRAMMAR_V4). Scripts may also edit V4 at run time (ablations, reports/.../cov.py --v4).
V4: set = set(V4_ALL) if GRAMMAR_V4 else set()
JOIN_PAIRS = GRAMMAR_V4  # grammar v4: join_cols are consecutive (left, right) pairs (see plan.JOIN_HINT_MODE)
TAGS: list = []  # v4 constructs the last convert() call used (for coverage accounting)


# Grammar v3: expression templates the model may pick. None = accept any (used to collect the list from
# training data); a set = only these (dev coverage must use a list learned on training data only).
ALLOWED_TEMPLATES: set[str] | None = None
ALLOWED_NUMS: set | None = None  # numeric template constants ({N0}, {N1}) seen in training
MAX_TEMPLATE_VALS = 3
MAX_TEMPLATE_NUMS = 3


def _lift_nums(e: exp.Expression) -> list:
    """Numeric constants not compared against anything (SUBSTR positions, ROUND digits, * 100, / 12) become
    {N0}, {N1}...: the model picks each from a small closed vocabulary, so one template covers them all."""
    nums = []
    for node in list(e.walk(bfs=False)):
        if isinstance(node, exp.Literal) and not node.is_string and not isinstance(node.parent, tuple(_CMP) + (exp.Between, exp.In)):
            v = int(node.this) if node.this.lstrip("-").isdigit() else float(node.this)
            if isinstance(node.parent, exp.Neg):
                v, node = -v, node.parent
            if ALLOWED_NUMS is not None and v not in ALLOWED_NUMS:
                raise Unsupported("number unseen")
            nums.append(v)
            node.replace(exp.column("__N%d__" % (len(nums) - 1)))
    if len(nums) > MAX_TEMPLATE_NUMS:
        raise Unsupported("template numbers > 3")
    return nums


def _num_text(t: str, n: int) -> str:
    for i in range(n):
        t = t.replace("__N%d__" % i, "{N%d}" % i)
    return t


@dataclass
class Converted:
    plan: Plan | None
    reason: str = ""


_AGG = {exp.Sum: "sum", exp.Avg: "avg", exp.Min: "min", exp.Max: "max"}
_CMP = {exp.EQ: "=", exp.NEQ: "!=", exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<="}
_FLIP = {"=": "=", "!=": "!=", ">": "<", ">=": "<=", "<": ">", "<=": ">="}


def convert(sql: str, graph: Graph, dialect: str = "sqlite") -> Converted:
    TAGS.clear()
    try:
        return Converted(_convert(sql, graph, dialect))
    except Unsupported as e:
        return Converted(None, str(e))
    except sqlglot.errors.SqlglotError as e:
        return Converted(None, f"parse: {type(e).__name__}")


_SET = {exp.Union: "union", exp.Intersect: "intersect", exp.Except: "except"}


_DATE_ONLY_FMTS = {"%Y", "%m", "%d", "%Y-%m", "%Y-%m-%d", "%m-%d", "%W", "%j"}


def _is_now(e: exp.Expression) -> bool:
    e = _unwrap(e)
    while isinstance(e, (exp.TsOrDsToTimestamp, exp.TsOrDsToDate, exp.Date, exp.Datetime)) and e.this is not None:
        if isinstance(e, (exp.Date, exp.Datetime)) and e.expressions:
            return False  # DATE('now', '-1 day') is not "now"
        e = _unwrap(e.this)
    if isinstance(e, exp.Anonymous) and e.name.lower() in ("date", "datetime") and len(e.expressions) == 1:
        e = _unwrap(e.expressions[0])
    return isinstance(e, (exp.CurrentDate, exp.CurrentTimestamp, exp.CurrentDatetime)) or (
        isinstance(e, exp.Literal) and e.is_string and e.this.lower() == "now")


def _canon_now(tree: exp.Expression) -> bool:
    """Grammar v4.1: the current date written any way inside a date-only STRFTIME becomes CURRENT_TIMESTAMP,
    so one template covers STRFTIME('%Y', 'now'), STRFTIME('%Y', DATE('NOW')), STRFTIME('%Y', CURRENT_DATE)..."""
    hit = False
    for node in list(tree.find_all(exp.TimeToStr)):
        fmt = node.args.get("format")
        if fmt is None or fmt.name not in _DATE_ONLY_FMTS or node.this is None or not _is_now(node.this):
            continue
        inner = node.this
        if isinstance(inner, exp.TsOrDsToTimestamp) and isinstance(inner.this, exp.CurrentTimestamp):
            continue  # already canonical
        node.set("this", exp.TsOrDsToTimestamp(this=exp.CurrentTimestamp()))
        hit = True
    return hit


def _convert(sql: str, graph: Graph, dialect: str) -> Plan:
    tree = sqlglot.parse_one(sql.strip().rstrip(";"), read=dialect)
    if "now" in V4 and _canon_now(tree):
        TAGS.append("now")
    if "cte" in V4 and (tree.args.get("with_") or tree.args.get("with")):  # (v3: left to the later checks)
        tree = _inline_ctes(tree)
        TAGS.append("cte")
    for cls, name in _SET.items():
        if isinstance(tree, cls):
            if tree.args.get("order") or tree.args.get("limit"):
                raise Unsupported("set operation with order/limit")
            if cls is exp.Union and tree.args.get("distinct") is False:
                raise Unsupported("union all")
            l, rr = tree.this, tree.expression
            if not (isinstance(l, exp.Select) and isinstance(rr, exp.Select)):
                raise Unsupported("nested set operation")
            left, right = _convert_select(l, graph, 0), _convert_select(rr, graph, 0)
            if len(left.projections) != len(right.projections):
                raise Unsupported("set operation width")
            left.set_op, left.other = name, right
            return left
    if not isinstance(tree, exp.Select):
        raise Unsupported(f"statement: {tree.key}")
    if "scalar" in V4 and isinstance(tree, exp.Select) and any(e.find(exp.Select) for e in tree.expressions):
        tree = _scalar_compose(tree)
        TAGS.append("scalar")
    derived = _derived(tree)
    if derived is not None:
        if "flatten" not in V4:  # grammar v3 behaviour, exactly
            if any(tree.args.get(k) for k in ("where", "group", "having", "order", "limit", "with")):
                raise Unsupported("derived table with outer clauses")
            return _convert_derived(tree, derived, graph)
        outer = any(tree.args.get(k) for k in ("where", "group", "having", "order", "limit", "offset"))
        if not outer:
            try:
                return _convert_derived(tree, derived, graph)
            except Unsupported:
                pass
        flat = _flatten(tree, derived)
        TAGS.append("flatten")
        if _derived(flat) is not None:  # nested derived tables: one more level
            flat = _flatten(flat, _derived(flat))
        return _convert_select(flat, graph, 0)
    return _convert_select(tree, graph, 0)


_WRAP_AGG = {exp.Avg: "avg", exp.Sum: "sum", exp.Max: "max", exp.Min: "min", exp.Count: "count"}


def _inline_ctes(tree: exp.Expression) -> exp.Expression:
    """Grammar v4: WITH a AS (q) ... FROM a  ->  ... FROM (q) AS a (non-recursive CTEs, in order)."""
    w = tree.args.get("with_") or tree.args.get("with")
    if w.args.get("recursive"):
        raise Unsupported("recursive cte")
    tree = tree.copy()
    tree.set("with_", None)
    tree.set("with", None)
    ctes: dict[str, exp.Expression] = {}
    for c in w.expressions:
        body = c.this.copy()
        if not isinstance(body, exp.Select):
            raise Unsupported("cte kind")
        _subst_ctes(body, ctes)
        ctes[c.alias_or_name.lower()] = body
    _subst_ctes(tree, ctes)
    return tree


def _subst_ctes(node: exp.Expression, ctes: dict) -> None:
    for t in list(node.find_all(exp.Table)):
        body = ctes.get(t.name.lower())
        if body is None or t.args.get("db"):
            continue
        alias = t.alias or t.name
        t.replace(exp.Subquery(this=body.copy(), alias=exp.TableAlias(this=exp.to_identifier(alias))))


def _scalar_compose(tree: exp.Select) -> exp.Select:
    """Grammar v4: arithmetic over scalar subqueries that read the same tables and differ only in their
    WHERE becomes one query with conditional aggregates:
      SELECT (SELECT COUNT(x) FROM F WHERE c AND r1) - (SELECT COUNT(x) FROM F WHERE c AND r2)
      -> SELECT COUNT(CASE WHEN r1 THEN x END) - COUNT(CASE WHEN r2 THEN x END) FROM F WHERE c
    The outer query, when it has a FROM, is one more such source. A scalar subquery without an aggregate
    (one row by assumption) reads MAX(CASE WHEN r THEN x END). Only kept as a label after the execution
    check confirms the gold rows."""
    def bad(s: exp.Select) -> bool:
        return any(s.args.get(k) for k in ("group", "having", "order", "limit", "offset", "distinct", "with_"))
    if bad(tree):
        raise Unsupported("subquery")
    subs = []
    for e in tree.expressions:
        for s in e.find_all(exp.Select):
            if not isinstance(s.parent, exp.Subquery) or s.find_ancestor(exp.Select) is not tree:
                continue
            if bad(s) or len(s.expressions) != 1 or any(n is not s and isinstance(n, exp.Select) for n in s.walk()):
                raise Unsupported("subquery")
            subs.append(s)
    if not subs:
        raise Unsupported("subquery")

    def sig(s: exp.Select) -> str:
        f = s.args.get("from_")
        if f is None or not isinstance(f.this, exp.Table):
            raise Unsupported("subquery")
        return f.sql() + " " + " ".join(j.sql() for j in s.args.get("joins") or [])
    sources = subs + ([tree] if tree.args.get("from_") is not None else [])
    if len({sig(s) for s in sources}) != 1:
        raise Unsupported("subquery")
    conj = [[c for c in (_conjuncts(s.args["where"].this) if s.args.get("where") is not None else [])] for s in sources]
    keys = [{c.sql() for c in cs} for cs in conj]
    common = set.intersection(*keys)

    def condition(cs):
        rest = [c.copy() for c in cs if c.sql() not in common]
        return exp.and_(*rest) if rest else None

    def wrap_aggs(e: exp.Expression, r) -> exp.Expression:
        if r is None:
            return e
        aggs = [n for n in e.find_all(exp.AggFunc)
                if n.find_ancestor(exp.AggFunc) is None and n.find_ancestor(exp.Select) is None]
        if not aggs:  # a single-row scalar: MAX(CASE WHEN r THEN x END)
            if e.find(exp.Column) is None:
                raise Unsupported("subquery")
            return exp.Max(this=exp.Case(ifs=[exp.If(this=r.copy(), true=e.copy())]))
        for a in aggs:
            x = a.this
            if isinstance(a, exp.Count) and (x is None or isinstance(x, exp.Star)):
                a.replace(exp.Sum(this=exp.Case(ifs=[exp.If(this=r.copy(), true=exp.Literal.number(1))],
                                                default=exp.Literal.number(0))))
                continue
            if x is None or isinstance(x, exp.Distinct) or x.find(exp.Case, exp.If) or not isinstance(
                    a, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
                raise Unsupported("subquery")
            a.set("this", exp.Case(ifs=[exp.If(this=r.copy(), true=x.copy())]))
        return e

    new = tree.copy()
    if tree.args.get("from_") is None:
        new.set("from_", subs[0].args["from_"].copy())
        new.set("joins", [j.copy() for j in subs[0].args.get("joins") or []])
    else:  # the outer query's own aggregates take its residual condition
        r0 = condition(conj[-1])
        new.set("expressions", [wrap_aggs(e.copy(), r0) if r0 is not None else e.copy() for e in new.expressions])
    repl = {s.sql(): wrap_aggs(s.expressions[0].copy().unalias() if isinstance(s.expressions[0], exp.Alias)
                               else s.expressions[0].copy(), condition(cs)) for s, cs in zip(subs, conj)}
    for s in list(new.find_all(exp.Subquery)):
        if isinstance(s.this, exp.Select) and s.this.sql() in repl:
            s.replace(exp.Paren(this=repl[s.this.sql()].copy()))
    keep = [c.copy() for c in conj[0] if c.sql() in common]
    new.set("where", exp.Where(this=exp.and_(*keep)) if keep else None)
    if any(isinstance(n, exp.Select) and n is not new for n in new.walk()):
        raise Unsupported("subquery")
    return new


def _derived(tree: exp.Select) -> exp.Select | None:
    """SELECT ... FROM (SELECT ...) [AS t] with no join: the inner SELECT, else None. With outer clauses
    the caller flattens (grammar v4) or reports them."""
    frm = tree.args.get("from_") or tree.args.get("from")
    src = frm.this if frm is not None else None
    if not isinstance(src, exp.Subquery) or not isinstance(src.this, exp.Select) or tree.args.get("joins"):
        return None
    return src.this


def _case_value(e: exp.Expression):
    """CASE WHEN c THEN x [ELSE NULL] END / IIF(c, x, NULL) -> (x, c), else None."""
    e = _unwrap(e)
    if isinstance(e, exp.Case) and len(e.args.get("ifs") or []) == 1 and not e.this:
        d = e.args.get("default")
        if d is None or isinstance(_unwrap(d), exp.Null):
            br = e.args["ifs"][0]
            return br.args["true"], br.this
    if isinstance(e, exp.If):
        d = e.args.get("false")
        if d is None or isinstance(_unwrap(d), exp.Null):
            return e.args["true"], e.this
    return None


def _flatten(tree: exp.Select, inner: exp.Select) -> exp.Select:
    """Grammar v4: SELECT outs FROM (inner) [AS d] [WHERE w] [ORDER BY o] [LIMIT l] -> one SELECT.
    Outer references to the derived table's outputs are replaced by the inner expressions; an outer
    predicate over an aggregate output becomes HAVING. The result is only a training label after the
    execution check (prepare_plans) confirms it returns the gold rows."""
    if tree.args.get("joins") or tree.args.get("group") or tree.args.get("having"):
        raise Unsupported("derived table with outer clauses")
    if any(isinstance(e, exp.Star) for e in inner.expressions) or inner.args.get("with_"):
        raise Unsupported("derived table projection")
    dalias = (tree.args["from_"].this.alias or "").lower()
    names: dict[str, exp.Expression] = {}
    for e in inner.expressions:
        names.setdefault(_out_name(e), e.this if isinstance(e, exp.Alias) else e)
    inner_lim = inner.args.get("limit") or inner.args.get("offset")
    if inner_lim and any(tree.args.get(k) for k in ("where", "order", "limit", "offset")):
        raise Unsupported("derived table with outer clauses")
    inner_agg = bool(inner.args.get("group")) or any(e.find(exp.AggFunc) for e in inner.expressions)
    preds = _conjuncts(tree.args["where"].this) if tree.args.get("where") is not None else []
    rest, conds = [], []
    for p in preds:  # BIRD idiom: ... WHERE T IS NOT NULL over T = CASE WHEN c THEN x END
        q = p.this if isinstance(p, exp.Paren) else p
        if (isinstance(q, exp.Not) and isinstance(q.this, exp.Is) and isinstance(q.this.expression, exp.Null)
                and isinstance(q.this.this, exp.Column) and q.this.this.name.lower() in names
                and _case_value(names[q.this.this.name.lower()]) is not None):
            x, c = _case_value(names[q.this.this.name.lower()])
            names[q.this.this.name.lower()] = x
            conds.append(c.copy())
            continue
        rest.append(p)

    def sub(node):
        if isinstance(node, exp.Column) and (not node.table or node.table.lower() == dalias):
            n = node.name.lower()
            if n not in names:
                raise Unsupported("derived table column")
            return names[n].copy()
        return node

    new = inner.copy()
    outs = [e.copy().transform(sub) for e in tree.expressions]
    outer_agg = any(e.find(exp.AggFunc) for e in outs)
    if outer_agg:
        if inner_agg or inner_lim:
            raise Unsupported("derived table aggregate")
        if new.args.get("distinct"):
            raise Unsupported("derived table aggregate")
    new.set("expressions", outs)
    if tree.args.get("distinct"):
        new.set("distinct", exp.Distinct())
    wh, hv = list(conds), []
    for p in rest:
        m = p.copy().transform(sub)
        (hv if m.find(exp.AggFunc) else wh).append(m)
    if wh:
        cur = new.args.get("where")
        new.set("where", exp.Where(this=exp.and_(*([cur.this] if cur is not None else []), *wh)))
    if hv:
        if not inner_agg:
            raise Unsupported("derived table with outer clauses")
        cur = new.args.get("having")
        new.set("having", exp.Having(this=exp.and_(*([cur.this] if cur is not None else []), *hv)))
    if tree.args.get("order") is not None:
        new.set("order", tree.args["order"].copy().transform(sub))
    for k in ("limit", "offset"):
        if tree.args.get(k) is not None:
            new.set(k, tree.args[k].copy())
    return new


def _out_name(e: exp.Expression) -> str:
    if isinstance(e, exp.Alias):
        return e.alias.lower()
    e = _unwrap(e)
    return e.name.lower() if isinstance(e, exp.Column) else e.sql().lower()


def _convert_derived(tree: exp.Select, inner: exp.Select, graph: Graph) -> Plan:
    """Grammar v3: an outer query over a derived table. Two shapes:
      passthrough  SELECT t.a, t.b FROM (plan) t          -> the inner plan keeping only a, b
      wrap         SELECT AVG(t.n) FROM (plan ... GROUP BY) t -> plan.wrap = avg over projection n"""
    plan = _convert_select(inner, graph, 0)
    if plan.compose != "none" or plan.set_op != "none":
        raise Unsupported("derived table shape")
    names = [_out_name(e) for e in inner.expressions]
    outs = [_unwrap(e) for e in tree.expressions]
    if ("flatten" in V4 and len(outs) == 1 and isinstance(outs[0], exp.Count) and isinstance(outs[0].this, exp.Star)
            and (inner.args.get("group") or inner.args.get("distinct"))):
        plan.wrap, plan.wrap_idx = "count", 0  # grammar v4: COUNT(*) over groups / distinct rows
        TAGS.append("wrap_count")
        return plan
    if len(outs) == 1 and type(outs[0]) in _WRAP_AGG and isinstance(_unwrap(outs[0].this), exp.Column):
        n = _unwrap(outs[0].this).name.lower()
        if n not in names:
            raise Unsupported("derived table column")
        plan.wrap, plan.wrap_idx = _WRAP_AGG[type(outs[0])], names.index(n)
        return plan
    keep = []
    for e in outs:
        if not isinstance(e, exp.Column) or e.name.lower() not in names:
            raise Unsupported("derived table projection")
        keep.append(names.index(e.name.lower()))
    plan.projections = [plan.projections[i] for i in keep]
    return plan


def _allowed_sub(sel: exp.Select, root: exp.Select) -> bool:
    """An inner SELECT is allowed only one level deep, as the right side of IN / a comparison
    inside the outer WHERE."""
    anc, n = 0, sel.parent
    while n is not None and n is not root:
        if isinstance(n, exp.Select):
            anc += 1
        n = n.parent
    if anc:
        return False
    p = sel.parent
    if isinstance(p, exp.Subquery):
        p = p.parent
    while isinstance(p, exp.Paren):
        p = p.parent
    if not isinstance(p, (exp.In,) + tuple(_CMP)):
        return False
    where = root.args.get("where")
    n = p
    while n is not None and n is not root:
        if n is where:
            return True
        n = n.parent
    return False


def _convert_select(tree: exp.Select, graph: Graph, depth: int) -> Plan:
    if tree.args.get("with"):
        raise Unsupported("cte")
    for node in tree.walk():
        if node is not tree and isinstance(node, exp.Select):
            if depth > 0 or not _allowed_sub(node, tree):
                raise Unsupported("subquery")
        if isinstance(node, exp.Window):
            raise Unsupported("window")
    offset = tree.args.get("offset") or (tree.args.get("limit") and tree.args["limit"].args.get("offset"))
    if offset and "offset" not in V4:
        raise Unsupported("offset")

    r = _Resolver(tree, graph, depth)
    plan = Plan(projections=[])
    plan.distinct = bool(tree.args.get("distinct"))

    # projections
    for e in tree.expressions:
        e = _unwrap(e)
        agg_inside = e.find(exp.AggFunc) is not None and not isinstance(e, exp.AggFunc)
        if agg_inside and r.conditional_or_none(e) is None:
            if plan.compose != "none":
                raise Unsupported("two composes")
            comp, parts = None, None
            if isinstance(e, (exp.Div, exp.Mul, exp.Sub, exp.Add)) and not plan.projections:
                try:
                    comp, a, b = _compose(e)
                    parts = [r.proj(a), r.proj(b)]
                except Unsupported:
                    comp = None
            if comp is None:  # grammar v3: any arithmetic over up to three aggregates, as a template
                comp, parts = r.compose_template(e)
                plan.compose_nums = list(r.last_compose_nums)
            plan.compose, plan.compose_pos = comp, len(plan.projections)
            plan.projections += parts
        else:
            plan.projections.append(r.proj(e))
    max_proj = MAX_PROJ_V4 if "wide" in V4 else MAX_PROJ
    if not plan.projections or len(plan.projections) > max_proj:
        raise Unsupported(f"projections: {len(plan.projections)}")
    if len(plan.projections) > MAX_PROJ:
        TAGS.append("proj5+")

    # filters
    where = tree.args.get("where")
    n_or = 0
    if where is not None:
        for pred in _conjuncts(where.this):
            up = pred.this if isinstance(pred, exp.Paren) else pred
            if isinstance(up, exp.Or):
                try:
                    f = r.filter(up)  # same-column equalities become IN
                except Unsupported:
                    parts = _disjuncts(up)
                    if any(isinstance(q.this if isinstance(q, exp.Paren) else q, exp.And) for q in parts):
                        raise Unsupported("or of ands")
                    n_or += 1
                    for q in parts:
                        g = r.filter(q)
                        if g is None:
                            raise Unsupported("or with join condition")
                        g.group = n_or
                        plan.filters.append(g)
                    continue
                if f is not None:
                    plan.filters.append(f)
                continue
            f = r.filter(pred)
            if f is not None:
                plan.filters.append(f)
    merge_same_column(plan)
    if len(plan.filters) > MAX_FILTERS + 2:
        raise Unsupported("too many filters")
    having = tree.args.get("having")
    if having is not None:
        for pred in _conjuncts(having.this):
            plan.having.append(r.having(pred))

    # grouping
    group = tree.args.get("group")
    if group is not None:
        for g in group.expressions:
            g = _unwrap(g)
            plan.group_by.append(r.operand(g))
    if len(plan.group_by) > (MAX_GROUP_V4 if "wide" in V4 else MAX_GROUP):
        raise Unsupported("group_by > 2")
    if len(plan.group_by) > MAX_GROUP:
        TAGS.append("group3+")

    # order
    order = tree.args.get("order")
    if order is not None:
        if len(order.expressions) > (2 if "order2" in V4 else 1):
            raise Unsupported("multi-key order")
        o = order.expressions[0]
        plan.order = r.proj(_unwrap(o.this))
        plan.order_dir = "desc" if o.args.get("desc") else "asc"
        if len(order.expressions) == 2:
            o2 = order.expressions[1]
            plan.order2 = r.proj(_unwrap(o2.this))
            if plan.order2.cond is not None or (plan.order.cond is not None):
                raise Unsupported("multi-key order")
            plan.order2_dir = "desc" if o2.args.get("desc") else "asc"
            TAGS.append("order2")

    lim = tree.args.get("limit")
    if lim is not None:
        v = lim.expression if isinstance(lim, exp.Limit) else lim
        if not (isinstance(v, exp.Literal) and not v.is_string):
            raise Unsupported("limit expression")
        plan.limit = int(v.this)
    if offset:
        v = offset.expression if isinstance(offset, exp.Offset) else offset
        if not (isinstance(v, exp.Literal) and not v.is_string) or not 0 <= int(v.this) <= 9:
            raise Unsupported("offset")
        plan.offset = int(v.this)
        if plan.offset:
            TAGS.append("offset")

    # joins: every FROM/JOIN table must be reachable through the FK graph; the compiler
    # re-derives the path, and the execution check decides whether that matches the gold
    for t in r.tables:
        if t not in graph.columns:
            raise Unsupported(f"unknown table {t}")
    # join hints from the gold SQL: bridge tables, and the columns its join conditions use
    used = set(plan.tables())
    # gold joins a table it takes no column from: that join is meaning (existence / count),
    # so it is required, not a prunable route hint
    plan.require = [t for t in r.tables if t not in used]
    hint: list[Col] = []
    conds = [j.args.get("on") for j in tree.args.get("joins") or [] if j.args.get("on") is not None]
    if where is not None:
        conds.append(where.this)
    for cond in conds:
        for eq in cond.find_all(exp.EQ):
            a, b = _unwrap(eq.this), _unwrap(eq.expression)
            if isinstance(a, exp.Column) and isinstance(b, exp.Column):
                try:
                    ca, cb = r.col(a), r.col(b)
                except Unsupported:
                    continue
                if ca.table != cb.table:
                    hint += [ca, cb]
    if len(r.tables) > 1:
        if JOIN_PAIRS:  # grammar v4: keep (left, right) pairs, so each gold join condition stays an edge
            pairs: list = []
            for i in range(0, len(hint), 2):
                if (hint[i], hint[i + 1]) not in pairs and (hint[i + 1], hint[i]) not in pairs:
                    pairs.append((hint[i], hint[i + 1]))
            plan.join_cols = [c for pr in pairs for c in pr]
        else:
            plan.join_cols = list(dict.fromkeys(hint))
    return plan


def _unwrap(e: exp.Expression) -> exp.Expression:
    while True:
        if isinstance(e, (exp.Alias, exp.Paren)):
            e = e.this
        elif isinstance(e, exp.Cast) and e.to.this in (exp.DataType.Type.FLOAT, exp.DataType.Type.DOUBLE,
                                                         exp.DataType.Type.DECIMAL, exp.DataType.Type.INT,
                                                         exp.DataType.Type.BIGINT):
            e = e.this
        else:
            return e


def _is_hundred(e: exp.Expression) -> bool:
    e = _unwrap(e)
    return isinstance(e, exp.Literal) and not e.is_string and float(e.this) == 100.0


def _compose(e: exp.Expression) -> tuple[str, exp.Expression, exp.Expression]:
    """a/b -> ratio, a*100/b or (a/b)*100 or 100*a/b -> percent, a-b, a+b."""
    if isinstance(e, exp.Mul):
        l, rr = _unwrap(e.this), _unwrap(e.expression)
        if _is_hundred(rr) and isinstance(l, exp.Div):
            return "percent", _unwrap(l.this), _unwrap(l.expression)
        if _is_hundred(l) and isinstance(rr, exp.Div):
            return "percent", _unwrap(rr.this), _unwrap(rr.expression)
        raise Unsupported("multiplication")
    if isinstance(e, exp.Div):
        num, den = _unwrap(e.this), _unwrap(e.expression)
        if isinstance(num, exp.Mul):
            a, b = _unwrap(num.this), _unwrap(num.expression)
            if _is_hundred(b):
                return "percent", a, den
            if _is_hundred(a):
                return "percent", b, den
            raise Unsupported("multiplication")
        return "ratio", num, den
    if isinstance(e, exp.Sub):
        return "difference", _unwrap(e.this), _unwrap(e.expression)
    return "add", _unwrap(e.this), _unwrap(e.expression)


def _conjuncts(e: exp.Expression) -> list[exp.Expression]:
    e = _unwrap(e) if isinstance(e, exp.Paren) else e
    if isinstance(e, exp.And):
        return _conjuncts(e.this) + _conjuncts(e.expression)
    return [e]


def _literal(e: exp.Expression):
    e = _unwrap(e) if not isinstance(e, exp.Literal) else e
    if isinstance(e, exp.Neg) and isinstance(e.this, exp.Literal):
        v = _literal(e.this)
        return -v
    if isinstance(e, exp.Literal):
        if e.is_string:
            return e.this
        s = e.this
        return int(s) if s.lstrip("-").isdigit() else float(s)
    if isinstance(e, exp.Boolean):
        return 1 if e.this else 0
    raise Unsupported(f"non-literal value: {e.key}")


_FMT_PART = {"%Y": "year", "%Y-%m": "year_month", "%m": "month"}


def _date_part(e: exp.Expression) -> tuple[exp.Column, str] | None:
    """strftime('%Y', col) / YEAR(col) -> (col, part)."""
    e = _unwrap(e)
    if isinstance(e, exp.TimeToStr):
        inner = _unwrap(e.this)
        while isinstance(inner, (exp.TsOrDsToTimestamp, exp.TsOrDsToDate, exp.StrToTime, exp.TimeStrToTime)):
            inner = _unwrap(inner.this)
        fmt = e.args.get("format")
        if isinstance(inner, exp.Column) and fmt is not None and fmt.name in _FMT_PART:
            return inner, _FMT_PART[fmt.name]
    if isinstance(e, exp.Anonymous) and e.name.lower() == "strftime" and len(e.expressions) == 2:
        fmt, col = e.expressions
        if isinstance(fmt, exp.Literal) and fmt.this in _FMT_PART and isinstance(_unwrap(col), exp.Column):
            return _unwrap(col), _FMT_PART[fmt.this]
    if isinstance(e, exp.Year) and isinstance(_unwrap(e.this), exp.Column):
        return _unwrap(e.this), "year"
    return None


class _Resolver:
    """Maps query aliases and bare column names to real (table, column) of the schema."""

    def __init__(self, tree: exp.Select, graph: Graph, depth: int = 0):
        self.graph = graph
        self.depth = depth
        self.lower_tables = {t.lower(): t for t in graph.columns}
        self.lower_cols = {t: {c.lower(): c for c in cols} for t, cols in graph.columns.items()}
        self.alias: dict[str, str] = {}
        self.tables: list[str] = []
        for t in tree.find_all(exp.Table):
            if t.find_ancestor(exp.Select) is not tree:
                continue  # belongs to an inner query
            real = self.lower_tables.get(t.name.lower())
            if real is None:
                raise Unsupported(f"unknown table {t.name}")
            if real in self.tables:
                raise Unsupported("self-join")
            self.tables.append(real)
            self.alias[(t.alias or t.name).lower()] = real
            self.alias[t.name.lower()] = real

    def col(self, c: exp.Column) -> Col:
        name = c.name.lower()
        if c.table:
            t = self.alias.get(c.table.lower())
            if t is None:
                raise Unsupported(f"unknown alias {c.table}")
            real = self.lower_cols[t].get(name)
            if real is None:
                raise Unsupported(f"unknown column {c.table}.{c.name}")
            return Col(t, real)
        owners = [t for t in self.tables if name in self.lower_cols[t]]
        if not owners:
            raise Unsupported(f"unknown column {c.name}")
        if len(owners) > 1:
            raise Unsupported(f"ambiguous column {c.name}")
        return Col(owners[0], self.lower_cols[owners[0]][name])

    def _conditional(self, agg: str, arg: exp.Expression) -> Proj | None:
        """SUM/COUNT/AVG(CASE WHEN cond THEN x ELSE 0|NULL END) and IIF(cond, x, 0)."""
        arg = _unwrap(arg)
        if isinstance(arg, exp.Case) and len(arg.args.get("ifs") or []) == 1:
            branch = arg.args["ifs"][0]
            pred, then, default = branch.this, _unwrap(branch.args["true"]), arg.args.get("default")
        elif isinstance(arg, exp.If):
            pred, then, default = arg.this, _unwrap(arg.args["true"]), arg.args.get("false")
        else:
            return None
        if default is not None and not isinstance(_unwrap(default), exp.Null):
            if not (isinstance(_unwrap(default), exp.Literal) and _unwrap(default).this in ("0", "0.0")):
                raise Unsupported("case default")
        f2 = None
        conj = _conjuncts(pred)
        if len(conj) == 2 and "cond2" in V4:  # grammar v4.1: two structured conditions
            try:
                f, f2 = self.filter(conj[0]), self.filter(conj[1])
            except Unsupported:
                f = f2 = None
            if f is None or f2 is None or f.sub is not None or f2.sub is not None:
                f = f2 = None
            else:
                TAGS.append("cond2")
        if f2 is None and len(conj) != 1:  # grammar v3: the whole CASE as an expression template
            return Proj("count" if agg == "count" else agg, self.operand(arg))
        if f2 is None:
            f = self.filter(pred)
        if f is None:
            raise Unsupported("case condition")
        if isinstance(then, exp.Literal) and not then.is_string and then.this in ("1", "1.0"):
            if agg not in ("sum", "count"):
                raise Unsupported("case aggregate")
            return Proj("count", Col(self.tables[0], ROWS), f, f2)
        if isinstance(then, exp.Column):
            return Proj("count" if agg == "count" else agg, self.col(then), f, f2)
        return Proj("count" if agg == "count" else agg, self.operand(then), f, f2)

    def _boolsum(self, e: exp.Expression) -> Proj | None:
        """Grammar v4.1: SUM(a = 'x') / SUM(a = 'x' AND b > 3) count the rows where the predicate holds."""
        if "boolsum" not in V4 or not isinstance(e, exp.Sum) or e.this is None:
            return None
        pred = _unwrap(e.this)
        if not isinstance(pred, tuple(_CMP) + (exp.And, exp.Like, exp.In, exp.Between, exp.Is)):
            return None
        conj = _conjuncts(pred)
        if len(conj) > (2 if "cond2" in V4 else 1):
            return None
        fs = [self.filter(c) for c in conj]
        if any(f is None or f.sub is not None for f in fs):
            return None
        TAGS.append("boolsum")
        return Proj("count", Col(self.tables[0], ROWS), fs[0], fs[1] if len(fs) > 1 else None)

    def proj(self, e: exp.Expression) -> Proj:
        e = _unwrap(e)
        bp = self._boolsum(e)
        if bp is not None:
            return bp
        for cls, name in ((exp.Sum, "sum"), (exp.Count, "count"), (exp.Avg, "avg")):
            if isinstance(e, cls) and e.this is not None:
                cp = self._conditional(name, e.this)
                if cp is not None:
                    return cp
        if isinstance(e, exp.Column):
            return Proj("none", self.col(e))
        if isinstance(e, exp.Count):
            inner = e.this
            if inner is None or isinstance(inner, exp.Star):
                return Proj("count", Col(self.tables[0], ROWS))
            if isinstance(inner, exp.Distinct):
                if len(inner.expressions) != 1:
                    raise Unsupported("count distinct expression")
                return Proj("count_distinct", self.operand(_unwrap(inner.expressions[0])))
            return Proj("count", self.operand(_unwrap(inner)))
        for cls, name in _AGG.items():
            if isinstance(e, cls):
                inner = _unwrap(e.this)
                if isinstance(inner, exp.Distinct):
                    raise Unsupported("aggregate distinct")
                return Proj(name, self.operand(inner))
        if isinstance(e, exp.Star):
            raise Unsupported("select *")
        if e.find(exp.AggFunc):
            raise Unsupported(f"projection: {e.key}")
        return Proj("none", self.operand(e))

    def conditional_or_none(self, e):
        bp = self._boolsum(e)
        if bp is not None:
            return bp
        for cls, name in ((exp.Sum, "sum"), (exp.Count, "count"), (exp.Avg, "avg")):
            if isinstance(e, cls) and e.this is not None:
                return self._conditional(name, e.this)
        return None

    def compose_template(self, e: exp.Expression) -> tuple[str, list[Proj]]:
        """Grammar v3 projection: arithmetic / functions over up to three aggregates ({P0}..{P2}),
        e.g. "ROUND(CAST({P0} AS REAL) * 100 / {P1}, 2)". Each aggregate is an ordinary projection."""
        e = e.copy()
        parts, seen = [], {}
        for node in list(e.walk(bfs=False)):
            if isinstance(node, exp.AggFunc) and node.find_ancestor(exp.AggFunc) is None:
                key = node.sql(dialect="sqlite")
                if "dedup" in V4 and key in seen:  # grammar v4.1: the same aggregate twice is one part
                    node.replace(exp.column("__P%d__" % seen[key]))
                    TAGS.append("dedup")
                    continue
                parts.append(self.proj(node))
                seen[key] = len(parts) - 1
                node.replace(exp.column("__P%d__" % (len(parts) - 1)))
        if not parts or len(parts) > 3:
            raise Unsupported("compose parts")
        nums = _lift_nums(e)
        names = {"__P%d__" % i for i in range(len(parts))} | {"__N%d__" % i for i in range(len(nums))}
        if any(c.name not in names for c in e.find_all(exp.Column)):
            raise Unsupported("compose with bare column")
        t = _num_text(e.sql(dialect="sqlite"), len(nums))
        for i in range(len(parts)):
            t = t.replace("__P%d__" % i, "{P%d}" % i)
        if "'" in t.replace("'%", ""):
            raise Unsupported("compose with value")
        if ALLOWED_TEMPLATES is not None and t not in ALLOWED_TEMPLATES:
            raise Unsupported("template unseen")
        self.last_compose_nums = nums
        return t, parts

    def operand(self, e: exp.Expression) -> Col:
        """A column, or (grammar v3) an expression over at most two columns: the expression with its
        columns written {A}/{B} and its string values {V0}/{V1} is the template; numbers stay inline."""
        e = _unwrap(e)
        if isinstance(e, exp.Column):
            return self.col(e)
        if e.find(exp.Select, exp.AggFunc, exp.Window, exp.Star):
            raise Unsupported(f"expression: {e.key}")
        e = e.copy()
        cols, vals = [], []
        for node in list(e.walk(bfs=False)):
            if isinstance(node, exp.Column):
                c = self.col(node)
                if c not in cols:
                    cols.append(c)
                node.replace(exp.column("__C%d__" % cols.index(c)))
        if not cols:
            raise Unsupported("expression without column")
        if len(cols) > 2:
            raise Unsupported("expression columns > 2")
        for node in list(e.walk(bfs=False)):
            if not isinstance(node, exp.Literal):
                continue
            compared = isinstance(node.parent, tuple(_CMP) + (exp.Between, exp.In, exp.Like))
            if (node.is_string and not node.this.startswith("%")) or (compared and node.this not in ("0", "1")):
                vals.append(node.this if node.is_string else (int(node.this) if node.this.lstrip("-").isdigit()
                                                                else float(node.this)))
                node.replace(exp.Literal.string("__V%d__" % (len(vals) - 1)))
        if len(vals) > MAX_TEMPLATE_VALS:
            raise Unsupported("expression values > 3")
        nums = _lift_nums(e)
        t = _num_text(e.sql(dialect="sqlite"), len(nums)).replace("__C0__", "{A}").replace("__C1__", "{B}")
        for i in range(len(vals)):
            t = t.replace("'__V%d__'" % i, "{V%d}" % i)
        if ALLOWED_TEMPLATES is not None and t not in ALLOWED_TEMPLATES:
            raise Unsupported("template unseen")
        a = cols[0]
        return Col(a.table, a.column, fn=t, col2=cols[1].id if len(cols) > 1 else "", vals=tuple(vals), nums=tuple(nums))

    def _dq_string(self, e: exp.Expression) -> exp.Expression:
        """SQLite reads "text" as a string when no column has that name (Spider relies on it)."""
        if isinstance(e, exp.Column) and not e.table and e.this.args.get("quoted"):
            name = e.name.lower()
            if not any(name in self.lower_cols[t] for t in self.tables):
                return exp.Literal.string(e.name)
        return e

    def filter(self, p: exp.Expression) -> Filter | None:
        neg = False
        if isinstance(p, exp.Paren):
            p = p.this
        if isinstance(p, exp.Not):
            neg, p = True, p.this
            if isinstance(p, exp.Paren):
                p = p.this
        for cls, op in _CMP.items():
            if isinstance(p, cls):
                l, rr = self._dq_string(_unwrap(p.this)), self._dq_string(_unwrap(p.expression))
                if isinstance(rr, (exp.Subquery, exp.Select)) or isinstance(l, (exp.Subquery, exp.Select)):
                    if isinstance(l, (exp.Subquery, exp.Select)):
                        l, rr, op = rr, l, _FLIP[op]
                    if not isinstance(l, exp.Column) or neg:
                        raise Unsupported("subquery comparison")
                    return Filter(self.col(l), op, [], sub=self.subplan(rr))
                if isinstance(l, exp.Column) and isinstance(rr, exp.Column):
                    a, b = self.col(l), self.col(rr)
                    if a.table != b.table and op == "=" and not neg:
                        return None  # a join condition written in WHERE
                    if "colcmp" not in V4 or a == b or (neg and op not in ("=", "!=")):
                        raise Unsupported("column comparison")
                    if neg:
                        op = "!=" if op == "=" else "="
                    TAGS.append("colcmp")
                    name = {v: k for k, v in COL_OPS.items()}[op]
                    return Filter(Col(a.table, a.column, col2=b.id), name, [])
                if _date_part(l) is None and _date_part(rr) is not None:
                    l, rr, op = rr, l, _FLIP[op]
                dp = _date_part(l)
                if dp is not None:
                    if neg:
                        raise Unsupported("negated comparison")
                    return Filter(self.col(dp[0]), op, [str(_literal(rr))], dp[1])
                if not isinstance(l, exp.Column) and isinstance(rr, exp.Column):
                    l, rr, op = rr, l, _FLIP[op]
                if neg:
                    if op not in ("=", "!="):
                        raise Unsupported("negated comparison")
                    op = "!=" if op == "=" else "="
                return Filter(self.operand(l), op, [_literal(rr)])
        if isinstance(p, (exp.Like, exp.ILike)):
            l = _unwrap(p.this)
            pat = str(_literal(self._dq_string(_unwrap(p.expression))))
            core = pat.strip("%")
            if pat.endswith("%") and not pat.startswith("%"):
                core = core.rstrip("-/")  # LIKE '1981-12-%' == LIKE '1981-12%' on dates
            if not core or "%" in core or "_" in core:
                raise Unsupported("like pattern")
            if pat.startswith("%") and pat.endswith("%"):
                return Filter(self.operand(l), "not_like" if neg else "like", [core])
            if neg:
                raise Unsupported("like pattern")
            if pat.endswith("%"):
                return Filter(self.operand(l), "like_prefix", [core])
            if pat.startswith("%"):
                return Filter(self.operand(l), "like_suffix", [core])
            return Filter(self.operand(l), "=", [core])
        if isinstance(p, exp.In):
            l = _unwrap(p.this)
            if p.args.get("query") is not None:
                return Filter(self.operand(l), "not_in" if neg else "in", [], sub=self.subplan(p.args["query"]))
            return Filter(self.operand(l), "not_in" if neg else "in", [_literal(self._dq_string(v)) for v in p.expressions])
        if isinstance(p, exp.Between) and not neg:
            l = _unwrap(p.this)
            dp = _date_part(l)
            if dp is not None:
                return Filter(self.col(dp[0]), "between", [str(_literal(p.args["low"])), str(_literal(p.args["high"]))], dp[1])
            return Filter(self.operand(l), "between", [_literal(p.args["low"]), _literal(p.args["high"])])
        if isinstance(p, exp.Is) and isinstance(p.expression, exp.Null):
            l = _unwrap(p.this)
            return Filter(self.operand(l), "not_null" if neg else "is_null")
        if isinstance(p, exp.Or) and not neg:
            parts = _disjuncts(p)
            cols, vals = set(), []
            for q in parts:
                q = _unwrap(q)
                if not (isinstance(q, exp.EQ) and isinstance(_unwrap(q.this), exp.Column)):
                    raise Unsupported("or")
                cols.add(self.col(_unwrap(q.this)))
                vals.append(_literal(self._dq_string(_unwrap(q.expression))))
            if len(cols) != 1:
                raise Unsupported("or")
            return Filter(cols.pop(), "in", vals)
        raise Unsupported(f"predicate: {p.key}")


def _subplan(resolver, q) -> Plan:
    inner = q.this if isinstance(q, exp.Subquery) else q
    if not isinstance(inner, exp.Select):
        raise Unsupported("subquery kind")
    if resolver.depth > 0:
        raise Unsupported("nested subquery")
    sub = _convert_select(inner, resolver.graph, resolver.depth + 1)
    if len(sub.projections) != 1 or sub.compose != "none":
        raise Unsupported("subquery width")
    return sub


def _having(resolver, p) -> Having:
    if isinstance(p, exp.Between):
        pr = resolver.proj(_unwrap(p.this))
        if pr.agg == "none" or pr.cond is not None:
            raise Unsupported("having expression")
        return Having(pr.agg, pr.col, "between", [_literal(p.args["low"]), _literal(p.args["high"])])
    for cls, op in _CMP.items():
        if isinstance(p, cls):
            l, rr = _unwrap(p.this), _unwrap(p.expression)
            if isinstance(l, (exp.Literal, exp.Neg)):
                l, rr, op = rr, l, _FLIP[op]
            pr = resolver.proj(l)
            if pr.agg == "none" or pr.cond is not None:
                raise Unsupported("having expression")
            return Having(pr.agg, pr.col, op, [_literal(rr)])
    raise Unsupported("having predicate")


_Resolver.subplan = lambda self, q: _subplan(self, q)
_Resolver.having = lambda self, p: _having(self, p)


def _disjuncts(e: exp.Expression) -> list[exp.Expression]:
    e = e.this if isinstance(e, exp.Paren) else e
    if isinstance(e, exp.Or):
        return _disjuncts(e.this) + _disjuncts(e.expression)
    return [e]
