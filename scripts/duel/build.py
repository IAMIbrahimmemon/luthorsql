"""Contrastive duel inputs for the frozen gated pairs (memo 2026-10-10 §4 item 5).

  uv run python scripts/duel/build.py            # -> data/duel/decision_inputs.jsonl (+ .sha256)
  uv run python scripts/duel/build.py show ID..  # print the rendered (A=pick, B=runner) state for spot checks

Per pair: both SQLs re-executed read-only on the held-out BIRD-train SQLite file (30 s cap, first 100k rows):
column names, row counts, NULL counts, up to 5 rows in A-not-B and B-not-A (multiset difference), a clause-level
SQL diff (sqlglot), and which evidence clauses each SQL implements. Plus schema (ddl6 with the card, as
scripts/decider/dataset.py row()), evidence and question. No gold SQL anywhere; labels stay in decision_pairs.jsonl.
"""
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from phase1_eval import load  # noqa: E402

PAIRS = ROOT / "data/duel/decision_pairs.jsonl"
OUT = ROOT / "data/duel/decision_inputs.jsonl"
W = ROOT / "runs/phase1/p1w/p1w"
CARDS = ROOT / "data/sys1/bird_train/cards"
MAXROWS, SECONDS, NSHOW = 100_000, 30.0, 5


def dbpath(db):
    from luthor.teacher.verify import db_paths
    return db_paths([ROOT / "data/raw/bird_train/train_databases"])[db]


def execute(path, sql):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > SECONDS else 0, 10000)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(MAXROWS)
        return {"cols": cols, "rows": rows, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"cols": [], "rows": [], "error": str(e)[:200]}
    finally:
        con.close()


def cell(v):
    if isinstance(v, float):
        v = round(v, 6)
    s = "NULL" if v is None else str(v)
    return s[:60]


def fmt_rows(rows):
    return "\n".join(" | ".join(cell(v) for v in r) for r in rows) if rows else "(none)"


def summary(ex):
    if ex["error"]:
        return f"error: {ex['error']}"
    rows, cols = ex["rows"], ex["cols"]
    nulls = [sum(1 for r in rows if r[k] is None) for k in range(len(cols))]
    capped = " (capped)" if len(rows) >= MAXROWS else ""
    return (f"columns ({len(cols)}): {' | '.join(cols)}\nrow count: {len(rows)}{capped}\n"
            f"NULL count per column: {' | '.join(str(n) for n in nulls)}\nfirst rows:\n{fmt_rows(rows[:3])}")


def norm(r):
    return tuple(round(v, 6) if isinstance(v, float) else v for v in r)


def multiset_diff(a, b):
    ca, cb = Counter(map(norm, a["rows"])), Counter(map(norm, b["rows"]))
    only_a, only_b = ca - cb, cb - ca
    na, nb = sum(only_a.values()), sum(only_b.values())
    return list(only_a.elements())[:NSHOW], na, list(only_b.elements())[:NSHOW], nb


CLAUSES = [("DISTINCT", "distinct"), ("SELECT", "expressions"), ("FROM", "from"), ("JOIN", "joins"),
           ("WHERE", "where"), ("GROUP BY", "group"), ("HAVING", "having"), ("ORDER BY", "order"), ("LIMIT", "limit")]


def clauses(sql):
    import sqlglot
    try:
        t = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return None
    sel = t if isinstance(t, sqlglot.exp.Select) else t.find(sqlglot.exp.Select)
    if sel is None:
        return None
    out = {}
    for name, k in CLAUSES:
        v = sel.args.get(k)
        if v is None or v == []:
            out[name] = ""
        elif isinstance(v, list):
            out[name] = ", ".join(x.sql(dialect="sqlite") for x in v) if name == "SELECT" else \
                " ".join(x.sql(dialect="sqlite") for x in v)
        else:
            s = v.sql(dialect="sqlite")
            out[name] = re.sub(r"^(WHERE|GROUP BY|HAVING|ORDER BY|LIMIT|FROM)\s+", "", s)
            if name == "DISTINCT":
                out[name] = "yes"
    # subqueries / set operations: flag them as a whole
    out["OTHER"] = "" if t is sel else t.sql(dialect="sqlite")
    return out


def sql_diff(a, b):
    ca, cb = clauses(a), clauses(b)
    if ca is None or cb is None:
        import difflib
        ta, tb = a.split(), b.split()
        ops = [f"{op}: A[{' '.join(ta[i1:i2])}] B[{' '.join(tb[j1:j2])}]"
               for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb).get_opcodes() if op != "equal"]
        return "token diff:\n" + "\n".join(ops[:12])
    lines = [f"{k}:\n  A: {ca[k] or '(none)'}\n  B: {cb[k] or '(none)'}" for k in ca if ca[k] != cb[k]]
    return "\n".join(lines) if lines else "(no clause-level difference)"


X100 = re.compile(r"(\*|×)\s*\(?\s*100\b|\b100(\.0)?\s*(\*|×)|MULTIPLY\s*\(.{0,120}?\b100\b", re.I)


def evidence_checks(evidence):
    """Simple checkable clauses from the evidence text: (label, predicate(sql_upper, sql_raw) -> bool)."""
    ev = evidence or ""
    checks = []
    for lit in dict.fromkeys(re.findall(r"'((?:[^']|'')+)'|\"([^\"]+)\"", ev)):
        s = (lit[0] or lit[1]).strip()
        if s:
            checks.append((f"literal '{s}'", lambda u, r, s=s: s.lower() in r.lower() or s.replace("'", "''").lower() in r.lower()))
    for num in dict.fromkeys(re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", ev)):
        if num == "100" and X100.search(ev):
            continue
        checks.append((f"number {num}", lambda u, r, n=num: re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", r) is not None))
    if X100.search(ev):
        checks.append(("x100 (percentage)", lambda u, r: re.search(r"\*\s*100|100(\.0)?\s*\*", u) is not None))
    if re.search(r"DIVIDE|\s/\s|\bdivided\b|\bratio\b|\baverage\b|\bAVG\b", ev, re.I):
        checks.append(("division or AVG", lambda u, r: "/" in u or "AVG(" in u.replace(" ", "")))
    if re.search(r"IS\s+NOT\s+NULL|not\s+null", ev, re.I):
        checks.append(("IS NOT NULL", lambda u, r: re.search(r"IS\s+NOT\s+NULL", u) is not None))
    for fn in ("COUNT", "SUM", "AVG", "MAX", "MIN", "DISTINCT", "SUBTRACT", "CAST"):
        if re.search(rf"\b{fn}\b", ev):
            pat = {"SUBTRACT": r"-", "DISTINCT": r"\bDISTINCT\b"}.get(fn, rf"\b{fn}\s*\(")
            checks.append((f"{fn}", lambda u, r, p=pat: re.search(p, u) is not None))
    for col in dict.fromkeys(re.findall(r"([A-Za-z_][\w ]{0,40}?)\s*(?:=|>=|<=|>|<|!=)\s*", ev)):
        c = col.strip().split()[-1] if col.strip() else ""
        if c and c.upper() not in {"AND", "OR", "NOT", "WHERE"} and not c.isdigit():
            checks.append((f"column {c} used", lambda u, r, c=c: c.lower() in r.lower()))
    return checks


def coverage(evidence, sql):
    u = sql.upper()
    return {lab: bool(f(u, sql)) for lab, f in evidence_checks(evidence)}


def build_one(p, row, card, path):
    from luthor.sqlw.prompt6 import ddl6
    q, ev = row["question"], (row.get("context") or "")
    ea, eb = execute(path, p["sql_pick"]), execute(path, p["sql_runner"])
    oa, na, ob, nb = multiset_diff(ea, eb)
    return {"id": p["id"], "db_id": p["db_id"], "question": q, "evidence": ev.strip() or "(none)",
            "schema": ddl6(card, q, ev),
            "pick": {"sql": p["sql_pick"], "summary": summary(ea), "only_rows": fmt_rows(oa), "n_only": na,
                     "coverage": coverage(ev, p["sql_pick"])},
            "runner": {"sql": p["sql_runner"], "summary": summary(eb), "only_rows": fmt_rows(ob), "n_only": nb,
                       "coverage": coverage(ev, p["sql_runner"])},
            "same_cols": ea["cols"] == eb["cols"]}


def render(inp, first):
    """State for one order: first in {'pick','runner'} is shown as candidate A."""
    a, b = (inp["pick"], inp["runner"]) if first == "pick" else (inp["runner"], inp["pick"])

    def cov(x):
        return "; ".join(f"{k}: {'yes' if v else 'no'}" for k, v in x["coverage"].items()) or "(no checkable clauses)"
    return {
        "database_schema": inp["schema"], "evidence": inp["evidence"], "question": inp["question"],
        "candidates": {
            "A": {"sql": a["sql"], "result": a["summary"], "evidence_clauses_implemented": cov(a)},
            "B": {"sql": b["sql"], "result": b["summary"], "evidence_clauses_implemented": cov(b)}},
        "contrast": {
            "sql_differences": sql_diff(a["sql"], b["sql"]),
            "rows_in_A_not_in_B": f"{a['n_only']} rows; up to {NSHOW} shown:\n{a['only_rows']}",
            "rows_in_B_not_in_A": f"{b['n_only']} rows; up to {NSHOW} shown:\n{b['only_rows']}",
            "same_column_names": "yes" if inp["same_cols"] else "no"}}


def main():
    from luthor.sq.schema_card import SchemaCard
    pairs = [json.loads(l) for l in open(PAIRS)]
    b2 = load(W / "heldout_union_by_r2.jsonl")
    cards = {}
    with open(OUT, "w") as f:
        for p in pairs:
            row = b2[p["id"]]
            assert row["cands"][p["pick_idx"]]["sql"] == p["sql_pick"]
            if p["db_id"] not in cards:
                cards[p["db_id"]] = SchemaCard.load(CARDS / f"{p['db_id']}.json")
            inp = build_one(p, row, cards[p["db_id"]], dbpath(p["db_id"]))
            f.write(json.dumps(inp, sort_keys=True) + "\n")
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / (OUT.name + ".sha256")).write_text(f"{h}  {OUT.name}\n")
    print(f"{len(pairs)} inputs -> {OUT.relative_to(ROOT)} sha256 {h}")


def show(ids):
    inps = {r["id"]: r for r in map(json.loads, open(OUT))}
    for i in ids:
        st = render(inps[i], "pick")
        print(f"===== {i} ({inps[i]['db_id']})  [schema {len(st['database_schema'])} chars omitted]")
        st = {k: v for k, v in st.items() if k != "database_schema"}
        print(json.dumps(st, indent=1, ensure_ascii=False).replace("\\n", "\n    "))


if __name__ == "__main__":
    show(sys.argv[2:]) if sys.argv[1:2] == ["show"] else main()
