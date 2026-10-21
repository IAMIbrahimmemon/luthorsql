"""Result-aware validator input study (research note 02). Mac only, DECISION only, diagnostics (no fitting).

For every DECISION question in the R1 k=32 pool: A = R1 validator pick (max rr over live), B = best-rr live candidate
whose result signature differs from A's (the validator's #2 distinct result). Both are re-executed read-only and
rich result features are computed (row/col count, column names, types, NULL counts, duplicate rows, first rows,
COUNT->COUNT(DISTINCT) grain probe, pool support). Then simple question-conditioned result cues are checked: how often
does a cue point at B when B is right (rescue) vs when A is right (break). Also token lengths of the input formats.

Out: runs/analysis/validator/02_pairs.jsonl, runs/analysis/validator/02_result_features.out
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/analysis/validator"
import os
POOL = ROOT / os.environ.get("VAL02_POOL", "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl")
TAG = os.environ.get("VAL02_TAG", "")
DEC = ROOT / "data/sys1/heldout_v2/decision.json"
MISSES = ROOT / "runs/analysis/system1_pick_gap_misses.jsonl"
MANUAL = ROOT / "scripts/analysis/system1_pick_gap_manual.json"


def connect(db):
    con = sqlite3.connect(f"file:{ROOT}/data/sys1/dbs/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)

    def run(sql, seconds=20.0):
        box["t"] = time.time() + seconds
        cur = con.execute(sql)
        rows = cur.fetchmany(100_000)
        names = [d[0] for d in cur.description] if cur.description else []
        return names, rows
    return run


def kind(v):
    return "null" if v is None else "int" if isinstance(v, int) else "real" if isinstance(v, float) else "text"


def features(names, rows):
    n = len(rows)
    m = len(names)
    cols = []
    for j in range(m):
        vals = [r[j] for r in rows]
        ks = Counter(kind(v) for v in vals if v is not None)
        nn = sum(v is None for v in vals)
        num = [v for v in vals if isinstance(v, (int, float))]
        cols.append({"name": names[j], "type": ks.most_common(1)[0][0] if ks else "null", "nulls": nn,
                     "distinct": len(set(vals)), "min": min(num) if num else None, "max": max(num) if num else None})
    return {"n": n, "m": m, "cols": cols, "dup": n - len(set(rows)), "head": [list(r) for r in rows[:5]],
            "scalar": rows[0][0] if n == 1 and m == 1 else None}


COUNT_RE = re.compile(r"COUNT\(\s*(?!DISTINCT\b|\*)", re.I)


def grain_probe(run, sql):
    """COUNT(x) -> COUNT(DISTINCT x): the number of distinct things counted. None when not applicable."""
    head = re.split(r"\bFROM\b", sql, maxsplit=1, flags=re.I)[0]  # only a COUNT in the outer SELECT list
    if not COUNT_RE.search(head) or re.search(r"\bGROUP BY\b", sql, re.I) and not re.search(r"\(\s*SELECT", sql, re.I):
        return None
    try:
        _, rows = run(COUNT_RE.sub("COUNT(DISTINCT ", head, count=1) + sql[len(head):], 10.0)
        return rows[0][0] if len(rows) == 1 and len(rows[0]) == 1 else None
    except Exception:
        return None


HOWMANY = re.compile(r"^\s*(how many|how much|what is the (total )?(number|count|amount)|count the|calculate the (total )?number)", re.I)
PCT = re.compile(r"percent|percentage|ratio|proportion|rate\b", re.I)
LIST = re.compile(r"\b(list|name (all|the)|give (all|the)|what are|which (are|of)|show)\b", re.I)


def cues(q, a, b):
    """Question-conditioned result cues. Each returns 'A', 'B' or None (which side looks right)."""
    out = {}
    # 1. "how many" wants one number
    if HOWMANY.search(q):
        ok = lambda f: f["n"] == 1 and f["m"] == 1 and isinstance(f["scalar"], (int, float))
        if ok(a) != ok(b):
            out["howmany_shape"] = "A" if ok(a) else "B"
    # 2. percentage: value in [0,100] and not tiny fraction when question says percent
    if re.search(r"percent", q, re.I) and a["scalar"] is not None and b["scalar"] is not None:
        try:
            fa, fb = float(a["scalar"]), float(b["scalar"])
            ina, inb = 0 <= fa <= 100, 0 <= fb <= 100
            if ina != inb:
                out["pct_range"] = "A" if ina else "B"
        except Exception:
            pass
    # 3. NULLs in the answer
    na = sum(c["nulls"] for c in a["cols"])
    nb = sum(c["nulls"] for c in b["cols"])
    if (na > 0) != (nb > 0):
        out["nulls"] = "B" if na > 0 else "A"
    # 4. extra column not named in the question (column superset)
    an = {c["name"].lower() for c in a["cols"]}
    bn = {c["name"].lower() for c in b["cols"]}
    if a["m"] != b["m"]:
        big, small, side = (a, b, "B") if a["m"] > b["m"] else (b, a, "A")
        extra = [c["name"] for c in big["cols"] if c["name"].lower() not in {x["name"].lower() for x in small["cols"]}]
        words = set(re.findall(r"[a-z]+", q.lower()))
        if extra and not any(set(re.findall(r"[a-z]+", e.lower())) & words for e in extra):
            out["extra_col"] = side
    # 5. huge result for a question that names a specific thing (quoted literal or proper noun mid-sentence)
    specific = bool(re.search(r"'[^']+'|\b[a-z]+ [A-Z][a-z]+", q))
    if specific and not HOWMANY.search(q):
        if a["n"] >= 100 and 0 < b["n"] <= 20:
            out["huge"] = "B"
        elif b["n"] >= 100 and 0 < a["n"] <= 20:
            out["huge"] = "A"
    # 6. grain probe: a count that double-counts entities (COUNT != COUNT DISTINCT) vs one whose value is the
    #    distinct count. Direction is NOT decidable from results alone; logged as 'visible conflict'.
    if a.get("probe") is not None and b.get("probe") is not None or a.get("probe") is not None or b.get("probe") is not None:
        def dbl(f):
            return f.get("probe") is not None and f["scalar"] is not None and f["probe"] != f["scalar"]
        if dbl(a) != dbl(b):
            out["grain_visible"] = "B" if dbl(a) else "A"  # direction = 'prefer the non-double-counting one'
    return out


def main():
    ids = set(json.load(open(DEC))["ids"])
    misses = {json.loads(l)["id"]: json.loads(l) for l in open(MISSES)}
    manual = json.load(open(MANUAL))
    pairs = []
    runs = {}
    t0 = time.time()
    for line in open(POOL):
        r = json.loads(line)
        if r["id"] not in ids:
            continue
        cands = [c for c in r["cands"] if c["src"] in ("r1", "both")]
        live = [c for c in cands if c["runs"] and c["nonnull"]] or cands
        if not live:
            continue
        A = max(live, key=lambda c: c["rr"])
        others = [c for c in live if c["sig"] != A["sig"]]
        oracle = any(c["ok"] for c in cands)
        rec = {"id": r["id"], "db": r["db_id"], "q": r["question"], "ev": r["context"], "A_ok": A["ok"],
               "oracle": oracle, "n_classes": len({c["sig"] for c in live}), "miss": r["id"] in misses}
        rec["A_support"] = sum(c["sig"] == A["sig"] for c in cands)
        if others:
            B = max(others, key=lambda c: c["rr"])
            rec.update(B_ok=B["ok"], B_support=sum(c["sig"] == B["sig"] for c in cands), margin=A["rr"] - B["rr"])
        else:
            B = None
        run = runs.get(r["db_id"]) or runs.setdefault(r["db_id"], connect(r["db_id"]))
        for side, c in (("A", A), ("B", B)):
            if c is None:
                continue
            rec[side + "_sql"] = c["sql"]
            rec[side + "_res"] = c["res"]
            try:
                names, rows = run(c["sql"])
                f = features(names, rows)
                f["probe"] = grain_probe(run, c["sql"])
            except Exception as e:
                f = {"err": str(e)[:80]}
            rec[side] = f
        if B is not None and "err" not in rec.get("A", {"err": 1}) and "err" not in rec.get("B", {"err": 1}):
            rec["cues"] = cues(r["question"] + " " + r["context"], rec["A"], rec["B"])
        m = misses.get(r["id"])
        if m:
            rec["theme"], rec["verdict"] = m["theme"], m["verdict"]
            rec["right_sql"] = m["right_sql"]
            try:
                names, rows = run(m["right_sql"])
                f = features(names, rows)
                f["probe"] = grain_probe(run, m["right_sql"])
                rec["R"] = f
                if "err" not in rec.get("A", {"err": 1}):
                    rec["cues_R"] = cues(r["question"] + " " + r["context"], rec["A"], f)
            except Exception as e:
                rec["R"] = {"err": str(e)[:80]}
        pairs.append(rec)
    print(f"executed {len(pairs)} questions in {time.time() - t0:.0f}s", file=sys.stderr)
    with open(OUT / f"02_pairs{TAG}.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, default=str) + "\n")


if __name__ == "__main__":
    main()
