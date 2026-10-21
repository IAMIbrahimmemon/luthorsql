"""Shared convention feature extraction (wording keys + SQL convention values). Mac-only, no I/O besides schema."""
from __future__ import annotations
import json, re
from pathlib import Path
import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[3]
TABLES = json.load(open(ROOT / "data/raw/bird_train/train_tables.json"))
PK = {}  # db -> set of lowercase pk col names (single-col pks)
for t in TABLES:
    s = set()
    for p in t["primary_keys"]:
        ps = p if isinstance(p, list) else [p]
        if len(ps) == 1:
            s.add(t["column_names_original"][ps[0]][1].lower())
    PK[t["db_id"]] = s

SUPER = re.compile(r"\b(highest|lowest|most|least|largest|smallest|biggest|best|worst|maximum|minimum|longest|shortest|"
                   r"oldest|youngest|earliest|latest|greatest|fewest|newest|heaviest|lightest|tallest|top|max|min|"
                   r"most recent|last|first)\b", re.I)
TOPN = re.compile(r"\btop\s+(\d+|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
HOWMANY = re.compile(r"\b(how many|number of|count)\b", re.I)
PCT = re.compile(r"\bpercent(age)?\b|%", re.I)
LISTQ = re.compile(r"^\s*(list|name|which|what are|give|provide|please list|state|identify|indicate|mention|show)\b", re.I)
FULLNAME = re.compile(r"\bfull name", re.I)
NAMEW = re.compile(r"\bnames?\b", re.I)


def wording(q: str, ev: str) -> dict:
    qe = q + " " + (ev or "")
    k = {}
    k["super"] = bool(SUPER.search(q)) and not TOPN.search(q)
    k["topn"] = bool(TOPN.search(q))
    k["howmany"] = bool(re.search(r"\bhow many\b", q, re.I))
    k["pct"] = bool(PCT.search(q))
    k["ev100"] = bool(re.search(r"100|percent|%", ev or "", re.I))
    k["ev_mult100"] = bool(re.search(r"\*\s*100|100\s*\*|MULTIPLY\s*\(.*100", ev or "", re.I))
    k["ev_distinct"] = bool(re.search(r"distinct", ev or "", re.I))
    k["ev_count"] = bool(re.search(r"count\s*\(", ev or "", re.I))
    k["ev_max"] = bool(re.search(r"\bmax\s*\(|\bmin\s*\(", ev or "", re.I))
    k["list"] = bool(LISTQ.search(q))
    k["fullname"] = bool(FULLNAME.search(q))
    k["name"] = bool(NAMEW.search(q))
    k["plural_q"] = bool(re.search(r"\b(which|what) (\w+ )?\w+s\b|\blist\b|\ball\b", q, re.I))
    k["n_and"] = len(re.findall(r"\b(and|,)\b|,", q))
    return k


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.replace("`", '"')).strip()


def sql_conv(sql: str, db: str) -> dict:
    s = _norm(sql or "")
    u = s.upper()
    c = {}
    # ties form
    lim = re.findall(r"\bLIMIT\s+(\d+)(?:\s*,\s*(\d+)|\s+OFFSET\s+\d+)?", u)
    maxsub = bool(re.search(r"=\s*\(\s*SELECT\s+(MAX|MIN)\s*\(", u))
    if lim:
        n = int(lim[-1][1] or lim[-1][0])
        c["tie"] = "limit1" if n == 1 else "limitn"
    elif maxsub:
        c["tie"] = "maxsub"
    elif re.search(r"SELECT\s+(MAX|MIN)\s*\(", u):
        c["tie"] = "aggval"
    else:
        c["tie"] = "none"
    # count form (outermost COUNT/SUM)
    m = re.search(r"\b(COUNT|SUM)\s*\(\s*(DISTINCT\s+)?([^()]*?)\)", u)
    if not m:
        c["cnt"] = "none"; c["cnt_pk"] = None
    else:
        f, d, arg = m.group(1), m.group(2), m.group(3).strip()
        if f == "SUM":
            c["cnt"] = "sum"
        elif d:
            c["cnt"] = "distinct"
        elif arg == "*" or arg == "1":
            c["cnt"] = "star"
        else:
            c["cnt"] = "col"
        col = arg.split(".")[-1].strip().strip('"').strip("'").lower() if arg not in ("*", "1") else "*"
        c["cnt_pk"] = col in PK.get(db, set()) if col != "*" else None
    c["join"] = " JOIN " in u or bool(re.search(r"FROM\s+[^()]*?,", u.split(" WHERE ")[0]))
    c["x100"] = bool(re.search(r"\*\s*100\b|\b100(\.0)?\s*\*|\*\s*100\.0", u))
    c["cast_real"] = bool(re.search(r"CAST\s*\(.*AS\s+(REAL|FLOAT|DOUBLE|DECIMAL|NUMERIC)", u))
    c["sel_distinct"] = bool(re.match(r"^SELECT\s+DISTINCT\b", u))
    # projection
    ncol, order_in_sel = None, None
    try:
        t = sqlglot.parse_one(s, read="sqlite")
        sel = t if isinstance(t, exp.Select) else t.find(exp.Select)
        ncol = len(sel.expressions)
        o = sel.args.get("order")
        if o is not None and o.expressions:
            oe = o.expressions[0].this.sql().lower().replace('"', "")
            sels = [e.sql().lower().replace('"', "") for e in sel.expressions]
            selsu = [e.unalias().sql().lower().replace('"', "") for e in sel.expressions]
            aliases = [e.alias.lower() for e in sel.expressions if e.alias]
            order_in_sel = oe in sels or oe in selsu or oe in aliases
    except Exception:
        pass
    c["ncol"] = ncol
    c["order_in_sel"] = order_in_sel
    return c
