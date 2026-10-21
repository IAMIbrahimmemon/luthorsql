"""Evidence-coverage check for BIRD-style evidence (generator-misses fix #4, docs/paper_notes/research/generator_misses_2026-10-10.md).

CPU only, no model. Three parts:

  parse_evidence(text, columns=None, question="") -> [Atom]
      Splits the evidence on ';' and pulls out what a SQL query implementing it must contain:
        cmp      col op literal            ("refers to Gender = 'F'", "Profit > = 1000", "year = '2007'")
        in       col [NOT] IN (a, b, ...)
        between  col BETWEEN a AND b
        null     col IS [NOT] NULL
        arith    an operator a formula asks for: DIVIDE(...) / ' / ' -> '/', SUBTRACT -> '-', MULTIPLY -> '*', ADD -> '+'
        const    a number a formula uses (MULTIPLY(..., 100), '* 100', '/ 9', percentage -> 100)
        col      a column a formula names (only with `columns`, the database's column names)
        mention  "X refers to col" with no operator (soft: never counts as strict)
      An atom is `required` unless its clause's left-hand phrase ("male" in "male refers to Gender = 'M'") is given and
      none of that phrase's words occur in the question: BIRD evidence often defines both sides (male and female)
      while the question uses one.
  sql_facts(sql) -> Facts      columns, tables, literals, predicates (col, op, values), arithmetic ops (sqlglot, sqlite).
  coverage(atoms, sql) -> dict  per-atom covered flags; "frac" over required strict atoms (None if there are none) and
                                "covers" (every required strict atom covered).

The `evidence` subcommand reports, on BIRD-train candidate pools (rerank_sq2 gen format), the AUC of the coverage
fraction against the candidate's `ok` label (pooled over candidates and within question), and how many questions
have no candidate (or no top-k candidate) that covers the evidence:

  uv run python scripts/repair/evidence.py report runs/phase1/p1w/p1w/r1/cands_train.jsonl \
      runs/phase1/p1w/p1w/r2/cands_train.jsonl --dbs data/raw/bird_train/train_databases

Refuses BIRD dev / sealed / gate / score_ files.
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

FORBIDDEN = ("bird_dev", "gate.json", "sealed", "/score_", "sel/score_")


def guard(p) -> Path:
    s = str(p)
    if any(f in s for f in FORBIDDEN):
        raise SystemExit(f"refusing to read {s}: BIRD dev / gate / sealed / score files are off limits")
    return Path(p)


# ---------------------------------------------------------------------------------------------------- atoms
@dataclass
class Atom:
    kind: str                     # cmp | in | between | null | arith | const | col | mention
    col: str | None = None        # bare column name, lower case
    op: str | None = None         # =, !=, >, >=, <, <=, like, not_like, in, not_in, between, is_null, not_null, / - * +
    values: tuple = ()            # normalised literals (norm_lit)
    text: str = ""                # the clause it came from
    required: bool = True
    strict: bool = True           # counts towards the coverage fraction

    def to_dict(self) -> dict:
        return asdict(self)


_ID = r'(?:`[^`]+`|"[^"]+"|\[[^\]]+\]|[A-Za-z_][\w]*(?:\.(?:`[^`]+`|"[^"]+"|[A-Za-z_]\w*))?)'
_STR = r"(?:'(?:[^']|'')*'|\"[^\"]*\"|‘[^’]*’)"
_NUM = r"-?\d+(?:\.\d+)?"
_LIT = rf"(?:{_STR}|{_NUM})"
_OPS = r"(?:>\s*=|<\s*=|<\s*>|!\s*=|=\s*=|=|>|<|\bnot\s+like\b|\blike\b)"
# a left side may be a function over a column: STRFTIME('%Y', DOB) = '2007', CAST(x AS DATE) = 2017, SUBSTR(c, 1, 4) = 'x'
_FN_LHS = r"(?:[A-Za-z_]+\s*\((?:[^()]|\([^()]*\))*\))"
_LHS = rf"(?P<lhs>{_FN_LHS}|{_ID})"
RX_CMP = re.compile(rf"{_LHS}\s*(?P<op>{_OPS})\s*(?P<lit>{_LIT})", re.I)
RX_IN = re.compile(rf"{_LHS}\s+(?P<neg>not\s+)?in\s*\((?P<vals>[^()]*)\)", re.I)
RX_BETWEEN = re.compile(rf"{_LHS}\s+between\s+(?P<a>{_LIT})\s+and\s+(?P<b>{_LIT})", re.I)
RX_NULL = re.compile(rf"{_LHS}\s+is\s+(?P<neg>not\s+)?null\b", re.I)
RX_LEFT = re.compile(r"^(?P<lhs>[^=<>()']*?)\b(?:refers?\s+to|refer\s+to|means|stands\s+for|is\s+the|are\s+the|"
                     r"(?:is|are)(?!\s+(?:not\s+)?null\b))\b", re.I)
RX_FUNC = re.compile(r"\b(divide|subtract|multiply|add|sum|avg|count|max|min|cast|percentage)\s*\(", re.I)

KEYWORDS = {"and", "or", "not", "where", "select", "from", "as", "in", "is", "null", "like", "between", "refers", "refer",
            "to", "the", "a", "an", "of", "by", "on", "if", "then", "else", "end", "case", "when", "count", "sum", "avg",
            "max", "min", "divide", "subtract", "multiply", "add", "cast", "real", "integer", "date", "text", "distinct",
            "strftime", "substr", "percentage", "percent", "iif", "group", "order", "limit", "desc", "asc", "having",
            "true", "false", "with", "for", "all", "has", "have", "are", "be", "it", "its", "this", "that", "than",
            "means", "stands", "which", "who", "what"}
_STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "for", "with", "by", "is", "are", "was", "were", "be",
         "who", "what", "which", "that", "this", "these", "those", "from", "at", "as", "it", "its", "has", "have", "had",
         "their", "his", "her", "refers", "refer"}


def unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "'\"`[‘" and s[-1] in "'\"`]’":
        s = s[1:-1]
    return s.replace("''", "'")


def norm_lit(v) -> str:
    """'2007' == 2007 == 2007.0; strings compare case-insensitively, trimmed."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _num_key(float(v))
    s = unquote(str(v)).strip()
    try:
        return _num_key(float(s))
    except ValueError:
        return s.lower()


def _num_key(x: float) -> str:
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else repr(x)


def norm_col(s: str) -> str:
    s = s.strip()
    if "." in s and not (s.startswith('"') and s.endswith('"') and s.count('"') == 2 and "." in s[1:-1]):
        s = s.rsplit(".", 1)[1]
    return unquote(s).strip().lower()


def norm_op(op: str) -> str:
    o = re.sub(r"\s+", "", op.lower())
    return {"==": "=", "<>": "!=", "notlike": "not_like"}.get(o, o)


def _lhs_col(lhs: str, columns: set[str] | None) -> str | None:
    """The column of a comparison's left side: the identifier itself, or the column inside a function call."""
    lhs = lhs.strip()
    if "(" in lhs:
        inner = lhs[lhs.index("(") + 1:lhs.rindex(")")] if ")" in lhs else lhs
        inner = re.sub(_STR, " ", inner)
        ids = [norm_col(m.group(0)) for m in re.finditer(_ID, inner)]
        ids = [i for i in ids if i not in KEYWORDS]
        if columns is not None:
            ids = [i for i in ids if i in columns]
        return ids[-1] if ids else None
    c = norm_col(lhs)
    if c in KEYWORDS or re.fullmatch(r"\d.*", c):
        return None
    if columns is not None and c not in columns:
        return None
    return c


def _multiword(body: str, start: int, c: str | None, cols: set[str] | None) -> str | None:
    """'Content Rating = ...': the identifier regex sees 'Rating'; take up to 3 preceding words when that makes a
    known column name ('content rating')."""
    if cols is None or c is None:
        return c
    words = re.findall(r"[A-Za-z_][\w?]*", body[max(0, start - 60):start])
    for n in (3, 2, 1):
        if len(words) >= n:
            cand = " ".join(w.lower() for w in words[-n:]) + " " + c
            if cand in cols:
                return cand
    return c


def _content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in _STOP}


def _clauses(text: str) -> list[str]:
    parts = re.split(r";|\n|\.\s+(?=[A-Z])", text or "")
    return [p.strip() for p in parts if p.strip()]


def parse_evidence(text: str, columns: set[str] | None = None, question: str = "") -> list[Atom]:
    """Atoms (see module doc). `columns`: lower-case column names of the database (stricter column detection and
    formula columns); `question`: marks clauses whose left phrase the question never uses as not required."""
    cols = {c.lower() for c in columns} if columns is not None else None
    qwords = _content_words(question) if question else None
    out: list[Atom] = []
    for cl in _clauses(text):
        clause = cl.replace(" ", " ")
        required = True
        m = RX_LEFT.match(clause)
        if m and qwords is not None:
            lw = _content_words(m.group("lhs"))
            # the left phrase is not in the question -> this definition is optional (stems: 'females' ~ 'female')
            if lw and not any(any(q.startswith(w[:max(4, len(w) - 2)]) or w.startswith(q[:max(4, len(q) - 2)])
                                  for q in qwords) for w in lw):
                required = False
        body = clause[m.end():] if m else clause
        taken: list[tuple[int, int]] = []

        def free(sp):
            return all(sp[1] <= a or sp[0] >= b for a, b in taken)

        for rx, kind in ((RX_BETWEEN, "between"), (RX_IN, "in"), (RX_NULL, "null")):
            for mm in rx.finditer(body):
                if not free(mm.span()):
                    continue
                c = _lhs_col(mm.group("lhs"), cols)
                c = _multiword(body, mm.start(), c, cols)
                if c is None:
                    continue
                taken.append(mm.span())
                if kind == "between":
                    out.append(Atom("between", c, "between", (norm_lit(mm.group("a")), norm_lit(mm.group("b"))), cl, required))
                elif kind == "in":
                    vals = tuple(norm_lit(v.group(0)) for v in re.finditer(_LIT, mm.group("vals")))
                    if vals:
                        out.append(Atom("in", c, "not_in" if mm.group("neg") else "in", vals, cl, required))
                else:
                    out.append(Atom("null", c, "not_null" if mm.group("neg") else "is_null", (), cl, required))
        for mm in RX_CMP.finditer(body):
            if not free(mm.span()):
                continue
            lhs = mm.group("lhs")
            if re.match(r"(divide|subtract|multiply|add|sum|avg|count|max|min)\s*\(", lhs, re.I):
                continue  # COUNT(x) > 5 / DIVIDE(...) = ...: an aggregate condition, handled as a formula
            c = _lhs_col(lhs, cols)
            if "(" not in lhs:
                c = _multiword(body, mm.start(), c if c is not None else (norm_col(lhs) if cols is not None else None), cols)
                if cols is not None and c not in cols:
                    c = None
            v = norm_lit(mm.group("lit"))
            taken.append(mm.span())
            if c is None:
                if cols is not None and not mm.group("lit").startswith(("'", '"', "‘")):
                    continue  # unknown left side and a bare number (e.g. 'age > 65' over a computed age): skip
                out.append(Atom("cmp", None, norm_op(mm.group("op")), (v,), cl, required))
            else:
                out.append(Atom("cmp", c, norm_op(mm.group("op")), (v,), cl, required))
        out += _formula_atoms(cl, body, cols, required)
        if m and not any(a.text == cl for a in out) and cols is not None:  # "X refers to col1, col2": soft mentions
            for idm in re.finditer(_ID, re.sub(_STR, " ", body)):
                c = norm_col(idm.group(0))
                if c in cols and c not in KEYWORDS:
                    out.append(Atom("mention", c, None, (), cl, required, strict=False))
    return _dedupe(out)


def _formula_atoms(cl: str, body: str, cols: set[str] | None, required: bool) -> list[Atom]:
    out = []
    low = body.lower()
    fns = {f.lower() for f in RX_FUNC.findall(body)}
    is_formula = bool(fns & {"divide", "subtract", "multiply", "add", "percentage"}) or re.search(
        r"(?<![<>=!])\s[/*]\s|\)\s*[/*]|[/*]\s*\(", body) is not None
    if not is_formula:
        return out
    if "divide" in fns or re.search(r"[)\w]\s*/\s*[(\w]", body):
        out.append(Atom("arith", None, "/", (), cl, required))
    if "subtract" in fns:
        out.append(Atom("arith", None, "-", (), cl, required))
    if "multiply" in fns or re.search(r"[)\w]\s*\*\s*[(\d]", body):
        out.append(Atom("arith", None, "*", (), cl, required, strict=False))  # * 100 is often written / 0.01 etc.
    if "add" in fns:
        out.append(Atom("arith", None, "+", (), cl, required, strict=False))
    nums = set()
    for mm in re.finditer(r"(?<![\w.'%-])(\d+(?:\.\d+)?)(?![\w.'%-])", re.sub(_STR, " ", body)):
        x = float(mm.group(1))
        if x not in (0.0, 1.0):
            nums.add(_num_key(x))
    if ("percentage" in low or "percent" in low or "%" in re.sub(_STR, " ", body)) and "/" in [a.op for a in out]:
        nums.add("100")
    for n in sorted(nums):
        out.append(Atom("const", None, None, (n,), cl, required))
    if cols is not None:  # the columns the formula is over
        seen = {norm_col(m.group(2)) for m in re.finditer(rf"count\s*\(\s*(distinct\s+)?({_ID})\s*\)", body, re.I)}
        for idm in re.finditer(_ID, re.sub(_STR, " ", body)):
            c = norm_col(idm.group(0))
            if c in cols and c not in KEYWORDS and c not in seen:
                seen.add(c)
                out.append(Atom("col", c, None, (), cl, required))
    return out


def _dedupe(atoms: list[Atom]) -> list[Atom]:
    seen, out = {}, []
    for a in atoms:
        k = (a.kind, a.col, a.op, a.values)
        if k in seen:  # keep the most demanding copy (required / strict wins)
            b = seen[k]
            b.required |= a.required
            b.strict |= a.strict
            continue
        seen[k] = a
        out.append(a)
    # a column already required by a cmp/in/between/null atom does not need its own col atom
    have = {a.col for a in out if a.kind in ("cmp", "in", "between", "null")}
    return [a for a in out if not (a.kind in ("col", "mention") and a.col in have)]


# ---------------------------------------------------------------------------------------------------- SQL side
@dataclass
class Facts:
    columns: set = field(default_factory=set)       # bare column names, lower
    qcolumns: set = field(default_factory=set)      # table.column, lower (aliases resolved)
    tables: set = field(default_factory=set)
    literals: set = field(default_factory=set)      # norm_lit
    preds: list = field(default_factory=list)       # (col, op, (values...)) col = bare lower name, or None
    qpreds: list = field(default_factory=list)      # (table.column | None, op, values) aliases resolved
    ops: set = field(default_factory=set)           # / * - +
    aggs: set = field(default_factory=set)          # avg sum count max min
    select_tables: set = field(default_factory=set)  # tables of the SELECT list's columns
    parsed: bool = True


_PRED_OPS = {"EQ": "=", "NEQ": "!=", "GT": ">", "GTE": ">=", "LT": "<", "LTE": "<=", "Like": "like", "ILike": "like"}
_FLIP = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "=", "!=": "!="}


@functools.lru_cache(maxsize=200_000)
def sql_facts(sql: str) -> Facts:
    import sqlglot
    from sqlglot import exp
    f = Facts()
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return _regex_facts(sql)
    if tree is None:
        return _regex_facts(sql)
    alias = {}
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        f.tables.add(name)
        alias[(t.alias_or_name or name).lower()] = name
        alias[name] = name

    def qual(c) -> str:
        t = (c.table or "").lower()
        return f"{alias.get(t, t)}.{c.name.lower()}" if t else c.name.lower()

    for c in tree.find_all(exp.Column):
        f.columns.add(c.name.lower())
        f.qcolumns.add(qual(c))
    for lit in tree.find_all(exp.Literal):
        f.literals.add(norm_lit(lit.this if lit.is_string else lit.this))
    for node, op in ((exp.Div, "/"), (exp.Mul, "*"), (exp.Sub, "-"), (exp.Add, "+"), (exp.Avg, "avg"),
                     (exp.Sum, "sum"), (exp.Count, "count"), (exp.Max, "max"), (exp.Min, "min")):
        if any(True for _ in tree.find_all(node)):
            (f.aggs if op.isalpha() else f.ops).add(op)
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is not None:
        for e in sel.expressions:
            for c in e.find_all(exp.Column):
                q = qual(c)
                if "." in q:
                    f.select_tables.add(q.split(".", 1)[0])
        if not f.select_tables and len(f.tables) == 1:
            f.select_tables = set(f.tables)

    def side_col(e):
        if isinstance(e, exp.Column):
            return e
        cs = list(e.find_all(exp.Column)) if e is not None else []
        return cs[0] if len(cs) == 1 else None

    def lit_of(e):
        if isinstance(e, exp.Literal):
            return norm_lit(e.this)
        if isinstance(e, exp.Neg) and isinstance(e.this, exp.Literal):
            return norm_lit("-" + e.this.this)
        if isinstance(e, exp.Cast):
            return lit_of(e.this)
        return None

    def add(col, op, vals):
        f.preds.append((col.name.lower() if col is not None else None, op, tuple(vals)))
        f.qpreds.append((qual(col) if col is not None else None, op, tuple(vals)))

    for node in tree.walk():
        name = type(node).__name__
        if name in _PRED_OPS:
            l, r = node.this, node.expression
            lv, rv = lit_of(l), lit_of(r)
            op = _PRED_OPS[name]
            neg = isinstance(node.parent, exp.Not)
            if op == "like" and neg:
                op = "not_like"
            if rv is not None and lv is None:
                add(side_col(l), op, [rv])
            elif lv is not None and rv is None:
                add(side_col(r), _FLIP.get(op, op), [lv])
        elif isinstance(node, exp.In):
            vals = [v for v in (lit_of(x) for x in node.expressions) if v is not None]
            if vals:
                add(side_col(node.this), "not_in" if isinstance(node.parent, exp.Not) else "in", vals)
        elif isinstance(node, exp.Between):
            a, b = lit_of(node.args.get("low")), lit_of(node.args.get("high"))
            if a is not None and b is not None:
                add(side_col(node.this), "between", [a, b])
        elif isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
            add(side_col(node.this), "not_null" if isinstance(node.parent, exp.Not) else "is_null", [])
    return f


def _regex_facts(sql: str) -> Facts:
    f = Facts(parsed=False)
    s = sql or ""
    f.literals = {norm_lit(m.group(0)) for m in re.finditer(rf"'(?:[^']|'')*'|(?<![\w.]){_NUM}(?![\w.])", s)}
    f.columns = {norm_col(m.group(0)) for m in re.finditer(r'"[^"]+"|`[^`]+`|\b[A-Za-z_]\w*\b', re.sub(r"'(?:[^']|'')*'", " ", s))}
    for op in "/*+":
        if op in s:
            f.ops.add(op)
    if re.search(r"\s-\s", s):
        f.ops.add("-")
    for m in RX_CMP.finditer(s):
        c = _lhs_col(m.group("lhs"), None)
        f.preds.append((c, norm_op(m.group("op")), (norm_lit(m.group("lit")),)))
    for m in RX_NULL.finditer(s):
        f.preds.append((_lhs_col(m.group("lhs"), None), "not_null" if m.group("neg") else "is_null", ()))
    f.qpreds = list(f.preds)
    return f


# ---------------------------------------------------------------------------------------------------- coverage
_OP_OK = {  # evidence op -> SQL ops that implement it
    "=": {"=", "in", "like"}, "!=": {"!=", "not_in", "not_like"}, ">": {">"}, ">=": {">=", "between"},
    "<": {"<"}, "<=": {"<=", "between"}, "like": {"like", "="}, "not_like": {"not_like", "!="},
}


def _num(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_YEAR_LIKE = re.compile(r"^(\d{4}(?:-\d{2})?)%$")


def _equiv(v: str, op: str) -> list[tuple[str, str]]:
    """(op, value) forms that implement evidence `col op v` besides itself: LIKE '2012%' == STRFTIME('%Y') = '2012'
    (and '2012-05%' == '%Y-%m' = '2012-05'); < '1970-01-01' == < 1970; >= '1970-01-01' == >= 1970."""
    out = []
    m = _YEAR_LIKE.match(v)
    if op == "like" and m:
        out.append(("=", m.group(1)))
    m = re.match(r"^(\d{4})-01-01$", v)
    if m and op in ("<", ">="):
        out.append((op, m.group(1)))
    m = re.match(r"^(\d{4})-12-31$", v)
    if m and op in (">", "<="):
        out.append((op, m.group(1)))
    return out


def atom_covered(a: Atom, f: Facts) -> bool:
    preds = [p for p in f.preds if a.col is None or p[0] == a.col]
    if a.kind == "cmp":
        v = a.values[0]
        if a.col is not None and a.col not in f.columns:
            return False
        for c, op, vals in preds:
            if op in _OP_OK.get(a.op, {a.op}) and (v in vals or (op in ("like", "not_like") and any(
                    x.strip("%") == v.strip("%") for x in vals))):
                if op == "between" and a.op in (">=", "<=") and v != (vals[0] if a.op == ">=" else vals[-1]):
                    continue
                return True
            # x > 5 written as x >= 6 (integers) is still right
            if a.op in (">", "<") and op in (">=", "<=") and _num(v) is not None and vals and _num(vals[0]) is not None:
                if (a.op == ">" and op == ">=" and _num(vals[0]) == _num(v) + 1) or (
                        a.op == "<" and op == "<=" and _num(vals[0]) == _num(v) - 1):
                    return True
        for op2, v2 in _equiv(v, a.op):
            if any(op == op2 and v2 in vals for c, op, vals in preds):
                return True
        if a.col is None:
            return v in f.literals or any(v2 in f.literals for _, v2 in _equiv(v, a.op))
        return False
    if a.kind == "in":
        if a.col is not None and a.col not in f.columns:
            return False
        got = set()
        for c, op, vals in preds:
            if (a.op == "in" and op in ("in", "=")) or (a.op == "not_in" and op in ("not_in", "!=")):
                got |= set(vals)
        return set(a.values) <= got
    if a.kind == "between":
        lo, hi = a.values
        if any(op == "between" and tuple(vals) == (lo, hi) for c, op, vals in preds):
            return True
        has_lo = any(op in (">=",) and lo in vals for c, op, vals in preds)
        has_hi = any(op in ("<=",) and hi in vals for c, op, vals in preds)
        if has_lo and has_hi:
            return True
        # a year range written as STRFTIME('%Y', d) BETWEEN '1990' AND '1995' (dates in the evidence)
        ylo, yhi = lo[:4], hi[:4]
        return any(op == "between" and tuple(vals) == (ylo, yhi) for c, op, vals in preds)
    if a.kind == "null":
        return any(op == a.op for c, op, vals in preds)
    if a.kind == "arith":
        if a.op in f.ops:
            return True
        # DIVIDE(SUM(x), COUNT(y)) written as AVG(x)
        return a.op == "/" and "avg" in f.aggs and re.search(r"\bsum\s*\(", a.text, re.I) is not None and \
            re.search(r"\bcount\s*\(", a.text, re.I) is not None
    if a.kind == "const":
        v = a.values[0]
        if v in f.literals:
            return True
        x = _num(v)
        return x is not None and x != 0 and any(_num(l) is not None and abs(_num(l) - 1.0 / x) < 1e-9 for l in f.literals)
    if a.kind in ("col", "mention"):
        return a.col in f.columns
    return False


def coverage(atoms: list[Atom], sql: str | None) -> dict:
    """{"covered": [bool per atom], "frac": share of required strict atoms covered (None when there are none),
    "covers": all of them covered, "missing": [atom index]}."""
    strict = [i for i, a in enumerate(atoms) if a.strict and a.required]
    if not sql:
        return {"covered": [False] * len(atoms), "frac": 0.0 if strict else None, "covers": not strict,
                "missing": strict}
    f = sql_facts(sql)
    cov = [atom_covered(a, f) for a in atoms]
    miss = [i for i in strict if not cov[i]]
    return {"covered": cov, "frac": (1 - len(miss) / len(strict)) if strict else None, "covers": not miss,
            "missing": miss}


# ---------------------------------------------------------------------------------------------------- metrics
def auc(scores: list[float], labels: list[bool]) -> float | None:
    """Mann-Whitney AUC with ties counted 1/2."""
    pos = sorted(s for s, l in zip(scores, labels) if l)
    neg = sorted(s for s, l in zip(scores, labels) if not l)
    if not pos or not neg:
        return None
    import bisect
    tot = 0.0
    for s in pos:
        lo = bisect.bisect_left(neg, s)
        hi = bisect.bisect_right(neg, s)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


@functools.lru_cache(maxsize=None)
def db_columns(db_dir: str, db_id: str) -> frozenset:
    return db_columns_path(str(Path(db_dir) / db_id / f"{db_id}.sqlite"))


@functools.lru_cache(maxsize=None)
def db_columns_path(path: str) -> frozenset:
    """Lower-case column names of every table of a SQLite file (read-only); empty if it does not exist."""
    p = Path(path)
    if not p.exists():
        return frozenset()
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        cols = set()
        for t in tabs:
            for r in con.execute(f'PRAGMA table_info("{t}")'):
                cols.add(str(r[1]).lower())
        return frozenset(cols)
    finally:
        con.close()


def load_pool(paths: list[Path]) -> dict:
    """Union of candidate files (rerank_sq2 gen format), deduplicated by SQL; src = file index of first occurrence;
    per-file rank kept as `rank`."""
    by = {}
    for i, p in enumerate(paths):
        for line in open(guard(p)):
            r = json.loads(line)
            if r["id"] not in by:
                by[r["id"]] = dict(r, cands=[])
            seen = {c["sql"] for c in by[r["id"]]["cands"]}
            for c in r["cands"]:
                if c.get("sql") and c["sql"] not in seen:
                    seen.add(c["sql"])
                    by[r["id"]]["cands"].append(dict(c, src=c.get("src", i)))
    return by


def report(a) -> dict:
    pool = load_pool(a.pools)
    ids = None
    if a.ids:
        d = json.load(open(guard(a.ids)))
        ids = set(d["ids"] if isinstance(d, dict) else d)
    if a.skip_ids:
        d = json.load(open(guard(a.skip_ids)))
        skip = set(d["ids"] if isinstance(d, dict) else d)
    else:
        skip = set()
    rows = [r for r in pool.values() if (ids is None or r["id"] in ids) and r["id"] not in skip]
    res = {"questions": len(rows)}
    for variant in ("schema+question", "schema", "plain"):
        cand_s, cand_l, within, n_ev, n_atoms = [], [], [], 0, 0
        flag_any = flag_top = flag_any_ok = flag_top_ok = unflag_ok = unflag = 0
        cov_ok_hit = cov_ok_tot = 0
        cv_n = cv_ok = nc_n = nc_ok = 0
        kinds = {}
        for r in rows:
            cols = set(db_columns(str(a.dbs), r["db_id"])) if (a.dbs and variant != "plain") else None
            atoms = parse_evidence(r.get("context") or "", cols, r["question"] if variant == "schema+question" else "")
            strict = [x for x in atoms if x.strict and x.required]
            if not strict:
                continue
            n_ev += 1
            n_atoms += len(strict)
            for x in strict:
                kinds[x.kind] = kinds.get(x.kind, 0) + 1
            cands = [c for c in r["cands"] if c.get("sql")]
            covs = [coverage(atoms, c["sql"]) for c in cands]
            sc = [c["frac"] for c in covs]
            ok = [bool(c.get("ok")) for c in cands]
            cand_s += sc
            cand_l += ok
            for o, c in zip(ok, covs):
                if c["covers"]:
                    cv_n += 1
                    cv_ok += o
                else:
                    nc_n += 1
                    nc_ok += o
            w = auc(sc, ok)
            if w is not None:
                within.append(w)
            top = [cv for c, cv in zip(cands, covs) if c.get("rank", 0) < a.top]
            any_cov = any(c["covers"] for c in covs)
            top_cov = any(c["covers"] for c in top)
            has_ok = any(ok)
            if r.get("covered"):
                cov_ok_tot += sum(ok)
                cov_ok_hit += sum(o and c["covers"] for o, c in zip(ok, covs))
            if not any_cov:
                flag_any += 1
                flag_any_ok += has_ok
            if not top_cov:
                flag_top += 1
                flag_top_ok += has_ok
            else:
                unflag += 1
                unflag_ok += has_ok
        res[variant] = {
            "questions_with_strict_atoms": n_ev, "strict_atoms": n_atoms, "atom_kinds": kinds,
            "candidates": len(cand_s), "auc_pooled": _r(auc(cand_s, cand_l)),
            "auc_within_q_mean": _r(sum(within) / len(within)) if within else None, "within_q_n": len(within),
            "ok_candidates_that_cover": f"{cov_ok_hit}/{cov_ok_tot}",
            "auc_covers_binary": _r(auc([1.0 if f == 1.0 else 0.0 for f in cand_s], cand_l)),
            "p_ok_if_covers": _r(cv_ok / cv_n) if cv_n else None, "p_ok_if_not": _r(nc_ok / nc_n) if nc_n else None,
            "no_cand_covers": flag_any, "no_cand_covers_oracle": _r(flag_any_ok / flag_any) if flag_any else None,
            f"no_top{a.top}_covers": flag_top, f"no_top{a.top}_covers_oracle": _r(flag_top_ok / flag_top) if flag_top else None,
            f"top{a.top}_covers_oracle": _r(unflag_ok / unflag) if unflag else None,
        }
    return res


def _r(x):
    return None if x is None else round(x, 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="AUC of evidence coverage vs ok on candidate pools")
    r.add_argument("pools", nargs="+", type=Path)
    r.add_argument("--dbs", type=Path, help="BIRD databases dir (<db>/<db>.sqlite) for column names")
    r.add_argument("--ids", type=Path, help="restrict to these ids (json list or {ids: [...]})")
    r.add_argument("--skip-ids", type=Path, help="drop these ids (e.g. RESERVED)")
    r.add_argument("--top", type=int, default=8, help="'top candidate' = per-file rank below this")
    r.add_argument("--out", type=Path)
    p = sub.add_parser("parse", help="print the atoms of one evidence string")
    p.add_argument("evidence")
    p.add_argument("--sql", default="")
    p.add_argument("--question", default="")
    a = ap.parse_args(argv)
    if a.cmd == "parse":
        atoms = parse_evidence(a.evidence, None, a.question)
        for x in atoms:
            print(x.to_dict())
        if a.sql:
            print(coverage(atoms, a.sql))
        return
    res = report(a)
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(s + "\n")


if __name__ == "__main__":
    sys.exit(main())
