"""Grammar gap: why the gold SQL of the questions grammar v3 does not cover falls outside it, and which constructs
would unlock the most questions (docs/paper_notes/research/grammar_gap_2026-10-10.md).

Ground truth for "covered" is the repo's own check, reports/fanout_r3/grammar_coverage/cov.py (gold SQL -> convert.py
plan -> ground literals -> compile_plan -> execute == gold rows; o == "plan" is covered, the same flag
scripts/analysis/heldout_v2.py uses for DECISION). The outcome files come from reports/grammar_gap_2026-10-10/run.sh
(v3 = default grammar, v4 = LUTHOR_GRAMMAR=v4 with every convert.V4 construct, train-safe vocab, no values41).

This script only reads TRAIN data: BIRD train.json, Spider train_spider/train_others, and the DECISION id list. It never
opens bird_dev, sealed or RESERVED files.

For every v3-uncovered gold it records
  stage     the repo's verdict: unsupported:<reason> | ungrounded_value | exec_mismatch | compile:no | exec_error
  tags      every construct the gold contains (sqlglot; multi-label, informational: CASE, IIF, CAST, STRFTIME, ...)
  blockers  the constructs in the gold that grammar v3 cannot express (detectors below, sqlglot)
  minimal   the single blocking construct when there is exactly one (else "multi:<a>+<b>..." or the stage family)
  v4        whether the already-built grammar v4 covers it, and with which v4 tags (exact counterfactual for those)

  uv run python scripts/analysis/grammar_gap.py [--out-dir reports/grammar_gap_2026-10-10]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import sqlglot
from sqlglot import exp

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/grammar_gap_2026-10-10"
BIRD_TRAIN = REPO / "data/raw/bird_train/train.json"
SPIDER = [REPO / "data/raw/spider_data/train_spider.json", REPO / "data/raw/spider_data/train_others.json"]
DECISION = REPO / "data/sys1/heldout_v2/decision.json"

AGGF = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
CMP = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
STRING_FNS = ("SUBSTR(", "SUBSTRING(", "INSTR(", "REPLACE(", "LENGTH(", "TRIM(", "UPPER(", "LOWER(", "||")
DATE_FNS = ("STRFTIME", "JULIANDAY", "DATE(", "DATETIME(", "CURRENT_TIMESTAMP", "'NOW'", "TIME(")


# ---------------------------------------------------------------------------------------------------------------
# construct detectors
def _scope(node):
    return node.find_ancestor(exp.Select)


def _selects(t):
    return [s for s in t.find_all(exp.Select)]


def _unwrap(e):
    while isinstance(e, (exp.Paren, exp.Alias)) or (isinstance(e, exp.Cast)):
        e = e.this
    return e


def _conj(e):
    e = e.this if isinstance(e, exp.Paren) else e
    if isinstance(e, exp.And):
        return _conj(e.this) + _conj(e.expression)
    return [e]


def _has_agg(e):
    return any(True for _ in e.find_all(AGGF))


def info_tags(sql: str, t) -> list[str]:
    """Constructs the gold contains (not necessarily blocking)."""
    up = sql.upper()
    tags = []
    if t.find(exp.Case):
        tags.append("CASE")
    if t.find(exp.If) or "IIF(" in up:
        tags.append("IIF")
    if t.find(exp.Cast):
        tags.append("CAST")
    if any(k in up for k in DATE_FNS):
        tags.append("date_fn")
    if any(k in up for k in STRING_FNS):
        tags.append("string_fn")
    if t.find(exp.Like):
        tags.append("LIKE")
    if any(isinstance(n, (exp.Div, exp.Mul, exp.Sub, exp.Add)) and _has_agg(n) and n.find_ancestor(*AGGF) is None
           for n in t.find_all(exp.Div, exp.Mul, exp.Sub, exp.Add)):
        tags.append("agg_arith(ratio/pct)")
    subs = [s for s in _selects(t) if s is not t]
    if subs:
        tags.append("subquery")
    if any(isinstance(x, (exp.Union, exp.Intersect, exp.Except)) for x in t.walk()):
        tags.append("set_op")
    if t.find(exp.Having):
        tags.append("HAVING")
    if t.find(exp.Group):
        tags.append("GROUP BY")
    if t.find(exp.Distinct):
        tags.append("DISTINCT")
    nt = len({x.name.lower() for x in t.find_all(exp.Table)})
    tags.append(f"tables={min(nt, 5)}{'+' if nt >= 5 else ''}")
    return tags


def _dq_to_literal(t):
    """SQLite reads "text" as a string when no column has that name (Spider relies on it); without the schema, treat
    an unqualified double-quoted identifier on the right of a comparison / IN / LIKE as a string literal."""
    for c in list(t.find_all(exp.Column)):
        if not c.table and c.this.args.get("quoted") and isinstance(c.parent, CMP + (exp.In, exp.Like, exp.Between)) \
                and c is not c.parent.this:
            c.replace(exp.Literal.string(c.name))
    return t


def blockers(sql: str, t) -> list[str]:
    """Constructs grammar v3 cannot express (convert.py with V4 empty + plan.py limits). Multi-label."""
    t = _dq_to_literal(t.copy())
    b = set()
    up = sql.upper()
    top = t
    # --- statement shape
    if t.find(exp.Window):
        b.add("window_fn")
    if t.find(exp.CTE):
        b.add("cte")
    for so in [x for x in t.walk() if isinstance(x, (exp.Union, exp.Intersect, exp.Except))]:
        if isinstance(so, exp.Union) and so.args.get("distinct") is False:
            b.add("union_all")
        if so.args.get("order") or so.args.get("limit"):
            b.add("set_op_order_limit")
        if isinstance(so.this, (exp.Union, exp.Intersect, exp.Except)) or isinstance(
                so.expression, (exp.Union, exp.Intersect, exp.Except)):
            b.add("set_op_nested")
        if so.find_ancestor(exp.Select) is not None:
            b.add("set_op_in_subquery")
    sels = _selects(t)
    for s in sels:
        if s.find_ancestor(exp.Select) is None:
            continue
        sq = s
        while isinstance(sq.parent, (exp.Subquery, exp.Union, exp.Intersect, exp.Except)):
            sq = sq.parent
        par = sq.parent
        while isinstance(par, exp.Paren):
            par = par.parent
        outer = s.find_ancestor(exp.Select)
        if isinstance(par, (exp.From, exp.Join)):
            b.add("derived_table" if isinstance(par, exp.From) and not outer.args.get("joins") else "derived_join")
            continue
        if isinstance(par, exp.In) and par.find_ancestor(exp.Not) is None and False:
            pass
        if s.find_ancestor(exp.Union, exp.Intersect, exp.Except) is not None and \
                s.find_ancestor(exp.Union, exp.Intersect, exp.Except).find_ancestor(exp.Select) is not None:
            b.add("set_op_in_subquery")
        anc, n = 0, s.parent
        while n is not None:
            if isinstance(n, exp.Select):
                anc += 1
            n = n.parent
        if anc >= 2:
            b.add("subquery_depth2+")
        if s.find_ancestor(exp.Having) is not None and s.find_ancestor(exp.Having).find_ancestor(exp.Select) is outer:
            b.add("subquery_in_having")
        elif s.find_ancestor(exp.Where) is None or s.find_ancestor(exp.Where).find_ancestor(exp.Select) is not outer:
            if s.find_ancestor(exp.Order):
                b.add("subquery_in_orderby")
            else:
                b.add("scalar_subquery_in_select")
        else:
            if isinstance(par, exp.Exists) or (isinstance(par, exp.Not) and isinstance(par.this, exp.Exists)):
                b.add("exists")
            elif not isinstance(par, (exp.In,) + CMP):
                b.add("subquery_in_arith")
            elif isinstance(par, CMP):
                other = par.this if par.expression is sq or _unwrap(par.expression) is s else par.expression
                if not isinstance(_unwrap(other), exp.Column):
                    b.add("subquery_cmp_expr")
        inner_alias = {(x.alias or x.name).lower() for x in s.find_all(exp.Table) if x.find_ancestor(exp.Select) is s}
        outer_alias = {(x.alias or x.name).lower() for x in outer.find_all(exp.Table) if x.find_ancestor(exp.Select) is outer}
        if any(c.table and c.table.lower() in outer_alias - inner_alias for c in s.find_all(exp.Column)
               if c.find_ancestor(exp.Select) is s):
            b.add("correlated_subquery")
        if s.find(exp.Select) is not None and any(x is not s for x in s.find_all(exp.Select)):
            b.add("subquery_depth2+")
        ex = [e for e in s.expressions]
        if len(ex) != 1 or any(_has_agg(_unwrap(e)) and isinstance(_unwrap(e), (exp.Div, exp.Mul, exp.Sub, exp.Add)) for e in ex):
            if not isinstance(par, (exp.From, exp.Join)):
                b.add("subquery_wide_or_compose")
    if t.find(exp.Exists):
        b.add("exists")
    # --- per SELECT scope
    for s in sels:
        if s.find_ancestor(exp.Select) is not None and isinstance(s.parent, exp.Subquery) and isinstance(
                s.parent.parent, (exp.From, exp.Join)):
            pass
        tabs = [x.name.lower() for x in s.find_all(exp.Table) if x.find_ancestor(exp.Select) is s]
        if len(tabs) != len(set(tabs)):
            b.add("self_join")
        for j in s.args.get("joins") or []:
            side = (j.args.get("side") or "").upper()
            kind = (j.args.get("kind") or "").upper()
            if side in ("LEFT", "RIGHT", "FULL") or kind in ("CROSS", "OUTER"):
                b.add("outer_join")
            if j.args.get("on") is None and j.args.get("using") is None and not isinstance(j.this, exp.Subquery):
                b.add("join_without_on")
        n_proj = len(s.expressions)
        if any(isinstance(e, exp.Star) for e in s.expressions):
            b.add("select_star")
        proj_aggs = [e for e in s.expressions]
        n_parts = 0
        n_compose = 0
        for e in proj_aggs:
            u = _unwrap(e)
            if _has_agg(u) and not isinstance(u, AGGF):
                n_compose += 1
                aggs = [a for a in u.find_all(AGGF) if a.find_ancestor(*AGGF) is None]
                n_parts += len({a.sql() for a in aggs})
                if len({a.sql() for a in aggs}) > 3:
                    b.add("compose_parts>3")
                bare = [c for c in u.find_all(exp.Column) if c.find_ancestor(*AGGF) is None
                        and c.find_ancestor(exp.Select) is s]
                if bare:
                    b.add("compose_with_bare_column")
                if any(l.is_string and not l.this.startswith("%") for l in u.find_all(exp.Literal)
                       if l.find_ancestor(*AGGF) is None):
                    b.add("compose_with_string")
            else:
                n_parts += 1
        if n_compose > 1:
            b.add("two_composes")
        if n_parts > 4:
            b.add("projections>4")
        g = s.args.get("group")
        if g is not None and len(g.expressions) > 2:
            b.add("group_by>2")
        o = s.args.get("order")
        if o is not None:
            if len(o.expressions) > 1:
                b.add("order_by_2+_keys")
            for oe in o.expressions:
                u = _unwrap(oe.this)
                if _has_agg(u) and not isinstance(u, AGGF):
                    b.add("order_by_agg_arith")
        lim = s.args.get("limit")
        if s.args.get("offset") or (lim is not None and lim.args.get("offset")):
            b.add("offset")
        if lim is not None:
            v = lim.expression if isinstance(lim, exp.Limit) else lim
            if not (isinstance(v, exp.Literal) and not v.is_string):
                b.add("limit_expr")
        for a in s.find_all(exp.Sum, exp.Avg, exp.Min, exp.Max):
            if a.find_ancestor(exp.Select) is s and isinstance(a.this, exp.Distinct):
                b.add("agg_distinct(sum/avg)")
        for a in s.find_all(exp.Count):
            if a.find_ancestor(exp.Select) is s and isinstance(a.this, exp.Distinct) and len(a.this.expressions) > 1:
                b.add("count_distinct_multi")
        # WHERE
        w = s.args.get("where")
        if w is not None:
            conj = _conj(w.this)
            n_f = 0
            for p in conj:
                q = p.this if isinstance(p, exp.Paren) else p
                if isinstance(q, exp.Or):
                    if any(isinstance(_unwrap(d), exp.And) or (isinstance(d, exp.Paren) and isinstance(d.this, exp.And))
                           for d in _disj(q)):
                        b.add("or_of_ands")
                    n_f += len(_disj(q))
                    continue
                if isinstance(q, CMP) and isinstance(_unwrap(q.this), exp.Column) and isinstance(_unwrap(q.expression), exp.Column):
                    l, r = _unwrap(q.this), _unwrap(q.expression)
                    if not (isinstance(q, exp.EQ) and (l.table or "").lower() != (r.table or "").lower()):
                        b.add("column_vs_column")
                    continue
                n_f += 1
            if n_f > 6:
                b.add("filters>6")
            for p in w.find_all(*CMP, exp.Between):
                if p.find_ancestor(exp.Select) is not s or p.find(exp.Select):
                    continue
                rhs = [p.expression] if not isinstance(p, exp.Between) else [p.args["low"], p.args["high"]]
                lhs = _unwrap(p.this)
                for r in rhs:
                    r = _unwrap(r)
                    if isinstance(r, exp.Neg):
                        r = r.this
                    if not isinstance(r, (exp.Literal, exp.Column, exp.Boolean, exp.Null)):
                        if not (isinstance(lhs, (exp.Literal,)) and isinstance(r, exp.Column)):
                            b.add("nonliteral_compare_value")
                    if isinstance(r, exp.Column) and not isinstance(lhs, exp.Column) and not isinstance(lhs, exp.Literal):
                        b.add("column_vs_column")
            for lk in w.find_all(exp.Like):
                if lk.find_ancestor(exp.Select) is not s:
                    continue
                pat = _unwrap(lk.expression)
                if isinstance(pat, exp.Literal) and pat.is_string:
                    core = pat.this.strip("%")
                    if "%" in core or "_" in core or not core:
                        b.add("like_wildcard_pattern")
                    neg = isinstance(lk.parent, exp.Not)
                    if neg and not (pat.this.startswith("%") and pat.this.endswith("%")):
                        b.add("like_wildcard_pattern")
                else:
                    b.add("like_wildcard_pattern")
        h = s.args.get("having")
        if h is not None:
            for p in _conj(h.this):
                q = p.this if isinstance(p, exp.Paren) else p
                if isinstance(q, exp.Or):
                    b.add("having_or")
                    continue
                if not isinstance(q, CMP + (exp.Between,)):
                    b.add("having_other_predicate")
                    continue
                sides = [_unwrap(q.this)] + ([_unwrap(q.expression)] if not isinstance(q, exp.Between) else [])
                agg_side = [x for x in sides if _has_agg(x)]
                lit_side = [x for x in sides if isinstance(x, (exp.Literal, exp.Neg))]
                if q.find(exp.Select):
                    b.add("subquery_in_having")
                elif len(agg_side) != 1 or (not isinstance(q, exp.Between) and len(lit_side) != 1) or not isinstance(
                        agg_side[0], AGGF) or agg_side[0].find(exp.Case, exp.If):
                    b.add("having_expression")
        # operands over > 2 columns (one operand template takes {A},{B}): projections, predicate left sides, GROUP/ORDER keys
        ops = [_unwrap(e) for e in s.expressions]
        if w is not None:
            for p in w.find_all(*CMP, exp.Like, exp.In, exp.Between, exp.Is):
                if p.find_ancestor(exp.Select) is s:
                    ops.append(_unwrap(p.this))
        for e in ops:
            if e is None or e.find(exp.Select) is not None:
                continue
            for a in [e] if not _has_agg(e) else [x.this for x in e.find_all(*AGGF) if x.this is not None]:
                cols = {c.sql().lower() for c in a.find_all(exp.Column)}
                if len(cols) > 2:
                    b.add("expression_cols>2")
        for a in s.find_all(*AGGF):
            if a.find_ancestor(exp.Select) is not s or a.find_ancestor(*AGGF) is not None:
                continue
            inner = a.this
            cols = {c.sql().lower() for c in a.find_all(exp.Column)}
            if inner is not None and isinstance(_unwrap(inner), (exp.Case, exp.If)):
                c = _unwrap(inner)
                ifs = c.args.get("ifs") or [] if isinstance(c, exp.Case) else [c]
                cond = ifs[0].this if ifs else None
                if cond is not None and len(_conj(cond)) == 2:
                    if len(cols) > 2:
                        b.add("case_two_conditions(cond2)")
                if cond is not None and len(_conj(cond)) > 2 and len(cols) > 2:
                    b.add("case_3+_conditions")
                if isinstance(c, exp.Case) and len(ifs) > 1 and len(cols) > 2:
                    b.add("case_multi_branch")
                default = c.args.get("default") if isinstance(c, exp.Case) else c.args.get("false")
                du = _unwrap(default) if default is not None else None
                if du is not None and not isinstance(du, exp.Null) and not (isinstance(du, exp.Literal) and du.this in ("0", "0.0")):
                    if len(cols) > 2:
                        b.add("case_nonzero_default")
            elif inner is not None and isinstance(a, exp.Sum) and isinstance(_unwrap(inner), CMP + (exp.And, exp.Like, exp.In)):
                b.add("sum_of_predicate(boolsum)")
            elif len(cols) > 2:
                b.add("expression_cols>2")
    if re.search(r"\bLIMIT\s+\d+\s*,", up):
        b.add("offset")
    return sorted(b)


def _disj(e):
    e = e.this if isinstance(e, exp.Paren) else e
    if isinstance(e, exp.Or):
        return _disj(e.this) + _disj(e.expression)
    return [e]


# ---------------------------------------------------------------------------------------------------------------
# repo verdict -> family
def stage_of(r) -> str:
    o = r["o"]
    if o == "unsupported":
        reason = r.get("reason", "")
        full = r.get("full", "")
        if reason.startswith("unknown") or reason.startswith("ambiguous"):
            return "unsupported:" + reason.split(" ")[0] + " " + reason.split(" ")[1] if " " in reason else "unsupported:" + reason
        if full.startswith("ambiguous"):
            return "unsupported:ambiguous column"
        return "unsupported:" + reason
    return o


# Which detector a converter refusal points to (to pick the minimal blocker when detectors miss or over-fire).
REASON_TO_BLOCKER = {
    "window": "window_fn", "cte": "cte", "union all": "union_all", "set operation with order/limit": "set_op_order_limit",
    "nested set operation": "set_op_nested", "self-join": "self_join", "column comparison": "column_vs_column",
    "or of ands": "or_of_ands", "multi-key order": "order_by_2+_keys", "offset": "offset", "limit expression": "limit_expr",
    "group_by > 2": "group_by>2", "projections": "projections>4", "aggregate distinct": "agg_distinct(sum/avg)",
    "count distinct expression": "count_distinct_multi", "compose with bare column": "compose_with_bare_column",
    "compose parts": "compose_parts>3", "two composes": "two_composes", "compose with value": "compose_with_string",
    "expression columns > 2": "expression_cols>2", "like pattern": "like_wildcard_pattern", "select *": "select_star",
    "having expression": "having_expression", "having predicate": "having_other_predicate",
    "too many filters": "filters>6", "case default": "case_nonzero_default",
    "expression values > 3": "expression_values>3", "template numbers > 3": "template_numbers>3",
    "non-literal value": "nonliteral_compare_value", "or": "or_inside_case_condition",
    "case condition": "case_other", "case aggregate": "case_other", "subquery width": "subquery_wide_or_compose",
    "subquery comparison": "subquery_cmp_expr", "derived table projection": "derived_table",
    "derived table with outer clauses": "derived_table", "derived table shape": "derived_table",
    # gold-side quirks, not grammar: 'col' in single quotes is a string; WHERE a OR b = 'x' on bare columns
    "expression without column": "GOLD_QUIRK(string-literal column / constant expression)",
    "predicate": "GOLD_QUIRK(bare column as predicate)",
}
# fallbacks used only when no detector fired
REASON_FALLBACK = {"projection": "order_by_agg_arith", "subquery": "subquery_other"}

# Family a construct belongs to, effort (grammar + compiler + slots + plan conversion), what the generator must learn,
# and whether the v4 code already has it (convert.V4 / plan.py).
META = {
    "join_route(non-FK / different path)": ("J", "S*", "built in v4 (join-hint pairs, m.n_join head); needs v4 retrain", "v4:joinedges"),
    "derived_table": ("D", "S*", "built in v4 (flatten / wrap_count); no new head", "v4:flatten"),
    "cte": ("D", "S*", "built in v4 (inline CTE -> derived -> flatten); no new head", "v4:cte"),
    "column_vs_column": ("S", "S*", "built in v4 (op classes *_col + x pointer slot)", "v4:colcmp"),
    "offset": ("O", "S*", "built in v4 (offset choice head 0-9)", "v4:offset"),
    "order_by_2+_keys": ("O", "S*", "built in v4 (order2 pointer + order2_dir head)", "v4:order2"),
    "projections>4": ("P", "S*", "built in v4 (n_proj 1-6)", "v4:wide"),
    "group_by>2": ("P", "S*", "built in v4 (n_group 0-4)", "v4:wide"),
    "filters>6": ("P", "S", "raise MAX_FILTERS/n_filter (v4: 6)", "v4:wide"),
    "case_two_conditions(cond2)": ("T", "S*", "built in v4.1 (cond2 mask + m.d0/m.d1 slots)", "v4:cond2"),
    "sum_of_predicate(boolsum)": ("T", "S*", "built in v4.1 (rewritten to conditional count; no new head)", "v4:boolsum"),
    "scalar_subquery_in_select": ("N", "M", "v4 'scalar' covers same-table arithmetic only; general: new block 'q' (scalar sub-plan) + compose slot", "v4:scalar (partial)"),
    "subquery_depth2+": ("N", "L", "a second sub-plan block (s2) + recursion in targets/serialize; longer plans", "new"),
    "correlated_subquery": ("N", "L", "correlation-column pointer between blocks; compiler aliasing", "new"),
    "subquery_cmp_expr": ("N", "M", "sub-plan compared to an operand template (fn on the filter column)", "new"),
    "subquery_in_arith": ("N", "M", "sub-plan inside a numeric template ({S} * N0); new template slot", "new"),
    "subquery_wide_or_compose": ("N", "M", "sub-plan with compose / >1 projection (s block gets compose head)", "new"),
    "subquery_in_having": ("N", "M", "HAVING agg op (sub-plan); having block gains sub flag", "new"),
    "exists": ("N", "M", "EXISTS / NOT EXISTS filter op (correlated: needs correlation pointer)", "new"),
    "derived_join": ("D", "L", "derived table joined to base tables: plan-as-table node in join tree", "new"),
    "window_fn": ("D", "L", "rank/row_number heads; rarely needed (ORDER BY ... LIMIT usually equivalent)", "new"),
    "self_join": ("S", "M", "table-instance index on every column pointer (T1/T2 of one table); join tree over instances", "new"),
    "outer_join": ("J", "S", "per-edge join-type head (inner/left); compiler emits LEFT JOIN", "new"),
    "or_of_ands": ("B", "M", "OR-group ids that may hold AND-ed sub-groups (2-level boolean), or DNF slots", "new"),
    "like_wildcard_pattern": ("B", "S", "LIKE ops with '_' / inner '%' as a raw pattern value (value head copies pattern)", "new"),
    "nonliteral_compare_value": ("B", "M", "compare an operand to an expression (date('now'), col*N0): template on RHS", "new"),
    "expression_cols>2": ("T", "M", "operand templates over 3+ columns ({C} slot, more x pointers)", "new"),
    "compose_with_bare_column": ("P", "M", "compose templates that mix {P} and a column ({A}) — new pointer in compose", "new"),
    "compose_parts>3": ("P", "S", "compose over 4+ parts (compose_pos/n_proj widen)", "new"),
    "two_composes": ("P", "M", "two compose heads (or compose over all projections)", "new"),
    "compose_with_string": ("P", "M", "CASE/IIF over aggregates returning a string ('which is more': CASE WHEN SUM(..) > SUM(..) THEN 'A' ELSE 'B'); compose template with {V} values grounded from the question", "new"),
    "order_by_agg_arith": ("O", "M", "ORDER BY a compose template (order slot reuses compose head)", "new"),
    "having_expression": ("P", "M", "HAVING over a compose template / conditional agg", "new"),
    "having_other_predicate": ("P", "S", "HAVING with IN/LIKE/IS NULL ops", "new"),
    "having_or": ("P", "S", "OR group ids in HAVING", "new"),
    "agg_distinct(sum/avg)": ("P", "S", "AGGS += sum_distinct, avg_distinct (agg head +2 classes)", "new"),
    "count_distinct_multi": ("P", "S", "COUNT(DISTINCT a, b) as template", "new"),
    "union_all": ("N", "S", "SET_OPS += union_all (set_op head +1 class)", "new"),
    "set_op_order_limit": ("N", "M", "ORDER/LIMIT over a set op (outer order slots over the union)", "new"),
    "set_op_nested": ("N", "M", "3-way set ops (a second 'o' block)", "new"),
    "set_op_in_subquery": ("N", "M", "set op inside a sub-plan", "new"),
    "select_star": ("P", "S", "SELECT * as a projection kind (all columns of a table)", "new"),
    "limit_expr": ("O", "S", "non-literal LIMIT", "new"),
    "case_nonzero_default": ("T", "S", "conditional aggregate with ELSE <value>", "new"),
    "case_3+_conditions": ("T", "M", "3+ AND-ed conditions on a conditional aggregate (cond3 or template)", "new"),
    "case_multi_branch": ("T", "M", "multi-branch CASE over 3+ columns", "new"),
    "join_without_on": ("J", "S", "cross join / comma join", "new"),
    "subquery_in_orderby": ("N", "M", "ORDER BY (sub-plan)", "new"),
    "join_route(different join columns)": ("J", "M", "v4 join pairs do not fix it: needs a join-column pointer the compiler must obey (not only a hint)", "new"),
    "join_route(different tables)": ("J", "M", "compiler picks another bridge table; 'via' head exists, needs stronger via/require use", "new"),
    "join_route(no FK path)": ("J", "S", "v4 join hints fix most; residual", "v4:joinedges"),
    "expression_values>3": ("T", "S", "MAX_TEMPLATE_VALS 3 -> 4-5 (value-role head +2 classes)", "new"),
    "template_numbers>3": ("T", "S", "MAX_TEMPLATE_NUMS 3 -> 5 (n3, n4 heads)", "new"),
    "or_inside_case_condition": ("T", "S", "conditional aggregate whose condition is an OR (cond OR-flag)", "new"),
    "case_other": ("T", "S", "other conditional-aggregate shapes", "new"),
}


def load_outcomes(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["i"]: r for r in map(json.loads, open(path))}


def classify(sqls: list[str], v3: dict, v4: dict) -> list[dict]:
    rows = []
    for i, sql in enumerate(sqls):
        r3 = v3.get(i)
        if r3 is None or r3["o"] == "plan":
            continue
        r4 = v4.get(i) or {}
        rec = {"i": i, "db": r3.get("db"), "stage": stage_of(r3), "v4_covered": r4.get("o") == "plan",
               "v4_tags": r4.get("v4", []) if r4.get("o") == "plan" else [], "v4_stage": stage_of(r4) if r4 else None}
        try:
            t = sqlglot.parse_one(sql.strip().rstrip(";"), read="sqlite")
            rec["tags"] = info_tags(sql, t)
            bl = blockers(sql, t)
        except Exception as e:  # noqa: BLE001
            rec["tags"], bl = ["parse_error"], ["parse_error"]
        st = rec["stage"]
        rec["detected"] = list(bl)
        if not st.startswith("unsupported"):  # converted: the grammar expressed it; only silent mis-compiles remain
            bl = [x for x in bl if x in ("outer_join", "join_without_on")]
        # The converter's own refusal names one blocker for sure; add it when the detectors missed it.
        if st.startswith("unsupported:"):
            reason = st.split(":", 1)[1]
            rb = REASON_TO_BLOCKER.get(reason)
            if rb and rb.startswith("GOLD_QUIRK"):
                bl = [rb]
            elif rb and rb not in bl:
                bl.append(rb)
            elif not bl and reason in REASON_FALLBACK:
                bl.append(REASON_FALLBACK[reason])
        if st in ("exec_mismatch", "compile:no", "exec_error") or (rec["v4_covered"] and "joinedges" in rec["v4_tags"]):
            if rec["v4_covered"] and "joinedges" in rec["v4_tags"]:
                bl.append("join_route(non-FK / different path)")
        if st in ("exec_mismatch", "compile:no", "exec_error"):
            rec["exec_cause"] = exec_cause(r3, sql)
        rec["blockers"] = sorted(set(bl))
        rec["minimal"] = minimal(rec)
        rows.append(rec)
    return rows


def _join_sig(sql: str):
    t = sqlglot.parse_one(sql.strip().rstrip(";"), read="sqlite")
    al = {}
    for x in t.find_all(exp.Table):
        al[(x.alias or x.name).lower()] = x.name.lower()
        al[x.name.lower()] = x.name.lower()
    tabs = frozenset(x.name.lower() for x in t.find_all(exp.Table))
    pairs = set()
    for eq in t.find_all(exp.EQ):
        a, b = eq.this, eq.expression
        if isinstance(a, exp.Column) and isinstance(b, exp.Column) and a.table and b.table:
            ka = (al.get(a.table.lower(), a.table.lower()), a.name.lower())
            kb = (al.get(b.table.lower(), b.table.lower()), b.name.lower())
            if ka[0] != kb[0]:
                pairs.add(frozenset((ka, kb)))
    return tabs, pairs, bool(t.args.get("distinct"))


def exec_cause(r3: dict, gold: str) -> str:
    """Why a converted plan did not reproduce the gold rows (v3)."""
    if r3["o"] == "compile:no":
        return "join_route(no FK path)"
    if r3["o"] == "exec_error":
        return "EXEC_error"
    try:
        gt, gp, gd = _join_sig(gold)
        ct, cp, cd = _join_sig(r3["sql"])
    except Exception:  # noqa: BLE001
        return "EXEC_other"
    if gt != ct:
        return "join_route(different tables)"
    if gp != cp and len(gt) > 1:
        return "join_route(different join columns)"
    if gd != cd:
        return "EXEC_distinct_semantics"
    return "EXEC_other(same joins: grouping/NULL/type/gold quirk)"


V4_TAG_TO_BLOCKER = {"joinedges": "join_route(non-FK / different path)", "flatten": "derived_table", "cte": "cte",
                     "colcmp": "column_vs_column", "offset": "offset", "order2": "order_by_2+_keys",
                     "proj5+": "projections>4", "group3+": "group_by>2", "filt5+": "filters>6", "cond2": "case_two_conditions(cond2)",
                     "boolsum": "sum_of_predicate(boolsum)", "wrap_count": "derived_table", "scalar": "scalar_subquery_in_select",
                     "dedup": "compose_parts>3", "now": "nonliteral_compare_value"}


def minimal(rec) -> str:
    st = rec["stage"]
    # exact counterfactual: v4 covers it and used exactly one v4 construct
    if rec["v4_covered"]:
        tg = sorted({V4_TAG_TO_BLOCKER.get(x, x) for x in rec["v4_tags"]} - {"filters>6"} if len(rec["v4_tags"]) > 1 else
                    {V4_TAG_TO_BLOCKER.get(x, x) for x in rec["v4_tags"]})
        if len(tg) == 1:
            return tg[0]
        if not tg:
            return "v4_other(no tag: converter/value fix)"
        return "multi:" + " & ".join(tg)
    bl = [b for b in rec["blockers"] if b != "parse_error"]
    if st == "ungrounded_value":
        return "VALUE_GROUNDING" if not bl else ("VALUE_GROUNDING+" + "+".join(bl[:2]))
    if st.startswith("unsupported:unknown") or st.startswith("unsupported:ambiguous"):
        return "SCHEMA_REF(" + st.split(":", 1)[1] + ")"
    if st in ("exec_mismatch", "compile:no", "exec_error"):
        if "outer_join" in bl:
            return "outer_join"
        return rec.get("exec_cause") or "EXEC_" + st.replace(":", "_")
    if len(bl) == 1:
        return bl[0]
    if not bl:
        return "UNDETECTED(" + st + ")"
    return "multi:" + " & ".join(bl)


# ---------------------------------------------------------------------------------------------------------------
def report(name: str, sqls: list[str], v3: dict, v4: dict, subset: set | None = None, fh=sys.stdout):
    ids = [i for i in range(len(sqls)) if i in v3 and (subset is None or i in subset)]
    n = len(ids)
    cov3 = sum(v3[i]["o"] == "plan" for i in ids)
    cov4 = sum((v4.get(i) or {}).get("o") == "plan" for i in ids) if v4 else None
    lost = sum(v3[i]["o"] == "plan" and (v4.get(i) or {}).get("o") != "plan" for i in ids) if v4 else None
    rows = [r for r in classify(sqls, v3, v4) if subset is None or r["i"] in subset]
    p = lambda *a: print(*a, file=fh)
    p(f"\n## {name}: n={n}  v3 covered {cov3} ({100 * cov3 / n:.2f}%)  uncovered {n - cov3}"
      + (f"  | v4 covered {cov4} ({100 * cov4 / n:.2f}%), v3->v4 lost {lost}" if v4 else ""))
    st = collections.Counter(r["stage"] for r in rows)
    p("\n### repo verdict (stage) of v3-uncovered")
    for k, c in st.most_common(40):
        p(f"  {c:5d}  {k}")
    any_ = collections.Counter(b for r in rows for b in r["blockers"])
    only = collections.Counter(r["minimal"] for r in rows)
    v4u = collections.Counter(r["minimal"] for r in rows if r["v4_covered"])
    p("\n### blocking constructs (any = gold contains it among v3-uncovered; minimal = sole blocker / exact v4 counterfactual)")
    p(f"  {'construct':42s} {'any':>5s} {'minimal':>7s} {'v4-unlocked':>11s}")
    keys = sorted(set(any_) | set(only), key=lambda k: (-only.get(k, 0), -any_.get(k, 0)))
    for k in keys:
        p(f"  {k[:42]:42s} {any_.get(k, 0):5d} {only.get(k, 0):7d} {v4u.get(k, 0):11d}")
    fp = collections.Counter()
    for i in ids:
        if v3[i]["o"] != "plan":
            continue
        try:
            for x in blockers(sqls[i], sqlglot.parse_one(sqls[i].strip().rstrip(";"), read="sqlite")):
                fp[x] += 1
        except Exception:  # noqa: BLE001
            pass
    p("\n### detector hits on v3-COVERED golds (false positives of the sqlglot detectors, or constructs v3 absorbs)")
    p("  " + ", ".join(f"{k} {c}" for k, c in fp.most_common()))
    info = collections.Counter(t for r in rows for t in r["tags"])
    p("\n### informational constructs present in v3-uncovered golds")
    p("  " + ", ".join(f"{k} {c}" for k, c in info.most_common()))
    return rows


NON_GRAMMAR = ("VALUE_GROUNDING", "SCHEMA_REF", "GOLD_QUIRK", "EXEC_", "v4_other", "UNDETECTED")


def ranking(sets: dict, fh=sys.stdout):
    """sets: name -> classified rows. Per construct: sole (the only blocker / exact v4 counterfactual), shared (one of
    several blockers), credit = sole + sum 1/k over shared, v4-unlocked (realised by the built v4 code)."""
    p = lambda *a: print(*a, file=fh)
    tab = collections.defaultdict(lambda: collections.defaultdict(float))
    for name, rows in sets.items():
        for r in rows:
            m = r["minimal"]
            if m.startswith("multi:"):
                fixed = m[6:].split(" & ")
                for x in fixed:
                    tab[x][name + ":shared"] += 1
                    tab[x][name + ":credit"] += 1 / len(fixed)
                    if r["v4_covered"]:
                        tab[x][name + ":v4"] += 1 / len(fixed)
            else:
                key = m.split("+")[0] if m.startswith("VALUE_GROUNDING") else m
                tab[key][name + ":sole"] += 1
                tab[key][name + ":credit"] += 1
                if r["v4_covered"]:
                    tab[key][name + ":v4"] += 1
    names = list(sets)
    p("\n## RANKING (grammar constructs; credit = sole + 1/k share of k-blocker questions)")
    hdr = f"  {'construct':38s} " + " ".join(f"{n[:8]+':sole':>13s} {'shared':>6s} {'credit':>6s} {'v4got':>5s}" for n in names)
    p(hdr + "  fam eff  built?")
    keys = sorted(tab, key=lambda k: -tab[k][names[0] + ":credit"])
    for k in keys:
        ng = any(k.startswith(x) for x in NON_GRAMMAR)
        meta = META.get(k, ("?", "?", "", "?"))
        line = f"  {('[non-grammar] ' if ng else '') + k:38s} " + " ".join(
            f"{tab[k][n + ':sole']:13.0f} {tab[k][n + ':shared']:6.0f} {tab[k][n + ':credit']:6.1f} {tab[k][n + ':v4']:5.1f}" for n in names)
        p(line + f"  {meta[0]:3s} {meta[1]:3s}  {meta[3]}")


GROUPS = {
    "A v4 as built (retrain only)": {k for k, v in META.items() if v[3].startswith("v4:") and "partial" not in v[3]},
    "B general sub-plans (expression-level subquery block)": {
        "scalar_subquery_in_select", "subquery_in_arith", "subquery_wide_or_compose", "subquery_cmp_expr", "subquery_depth2+",
        "subquery_in_having", "subquery_in_orderby", "correlated_subquery", "exists", "derived_join", "set_op_in_subquery",
        "set_op_nested", "set_op_order_limit", "union_all"},
    "C compositional expressions (expression tree instead of closed templates)": {
        "expression_cols>2", "expression_values>3", "template_numbers>3", "compose_with_string", "compose_with_bare_column",
        "two_composes", "compose_parts>3", "order_by_agg_arith", "having_expression", "having_other_predicate", "having_or",
        "nonliteral_compare_value", "case_3+_conditions", "case_multi_branch", "case_nonzero_default", "case_other",
        "or_inside_case_condition", "agg_distinct(sum/avg)", "count_distinct_multi"},
    "D boolean / LIKE (or_of_ands, wildcard LIKE)": {"or_of_ands", "like_wildcard_pattern"},
    "E join semantics (outer join, self-join, strict join columns)": {"outer_join", "self_join", "join_without_on",
        "join_route(different join columns)", "join_route(different tables)"},
}


def group_unlocks(sets: dict, fh=sys.stdout):
    """Questions whose every blocker lies in a group (alone), and cumulatively in A, A+B, A+B+C, ..."""
    p = lambda *a: print(*a, file=fh)
    def blk(r):
        m = r["minimal"]
        if any(m.startswith(x) for x in NON_GRAMMAR):
            return None
        return set(m[6:].split(" & ")) if m.startswith("multi:") else {m}
    p("\n## GROUP UNLOCKS (question counted when all its blockers lie in the group(s); before grounding/exec check)")
    names = list(sets)
    p(f"  {'group':78s} " + " ".join(f"{n:>9s}" for n in names))
    for g, ks in GROUPS.items():
        p(f"  {g[:78]:78s} " + " ".join(f"{sum(1 for r in sets[n] if (b := blk(r)) and b <= ks):9d}" for n in names))
    acc = set()
    for g, ks in GROUPS.items():
        acc |= ks
        p(f"  cumulative ..{g[:66]:66s} " + " ".join(f"{sum(1 for r in sets[n] if (b := blk(r)) and b <= acc):9d}" for n in names))
    ng = collections.Counter()
    p("  non-grammar (value grounding / gold or schema bugs / exec semantics): " + " ".join(
        f"{n}={sum(1 for r in sets[n] if blk(r) is None)}" for n in names))


def load_bird():
    return [q["SQL"] for q in json.load(open(BIRD_TRAIN))]


def load_spider():
    out = []
    for f in SPIDER:
        out += [q["query"] for q in json.load(open(f))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    od = Path(a.out_dir)
    dec = {int(x.rsplit("_", 1)[1]) for x in json.load(open(DECISION))["ids"]}
    bird, spider = load_bird(), load_spider()
    b3, b4 = load_outcomes(od / "v3_bird_train.jsonl"), load_outcomes(od / "v4_bird_train.jsonl")
    s3, s4 = load_outcomes(od / "v3_spider_train.jsonl"), load_outcomes(od / "v4_spider_train.jsonl")
    allrows = {}
    with open(od / "grammar_gap.out", "w") as fh:
        for name, sq, v3, v4, sub, tag in (("BIRD-train (all)", bird, b3, b4, None, "bird"),
                                            ("BIRD-train DECISION", bird, b3, b4, dec, "decision"),
                                            ("Spider-train", spider, s3, s4, None, "spider")):
            if not v3:
                print(f"(no outcomes for {name})", file=fh)
                continue
            rows = report(name, sq, v3, v4, sub, fh)
            allrows[tag] = rows
            with open(od / f"gap_{tag}.jsonl", "w") as g:
                for r in rows:
                    g.write(json.dumps(r) + "\n")
        ranking(allrows, fh)
        group_unlocks(allrows, fh)
    print(open(od / "grammar_gap.out").read())


if __name__ == "__main__":
    main()
