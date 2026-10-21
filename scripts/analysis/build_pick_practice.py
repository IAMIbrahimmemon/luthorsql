"""Build the practice-set data file for "The Pick Problem" (selection over the R1 ∪ R2 pool, research problem 08-03 /
01-03). Held-out BIRD training DBs only (data/sys1/dev_dbs_bird.json); no dev data, no gold SQL, no label-derived
fields other than the label itself.

  python scripts/analysis/build_pick_practice.py OUT_DIR

Writes OUT_DIR/pick_practice.json (1,000 questions) and OUT_DIR/schemas.json (CREATE statements of the 9 DBs).
Generator A = R1, generator B = R2 (B's own list = the published R2 pick).
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def own(path):
    """id -> {sql: (best rank, cost at that rank)} from a model's own top-8 file."""
    out = {}
    for i, r in load(path).items():
        m = {}
        for c in r["cands"]:
            if c["sql"] not in m or c["rank"] < m[c["sql"]][0]:
                m[c["sql"]] = (c["rank"], c["cost"])
        out[i] = m
    return out


def status(c):
    return "error" if not c.get("runs") else ("rows" if c.get("nonnull") else "empty_or_null")


def shape(res):
    m = re.match(r"(\d+) rows? x (\d+) cols? \(([^)]*)\)", res or "")
    return (int(m[1]), int(m[2]), [t.strip() for t in m[3].split(",")]) if m else (None, None, None)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    byA = load(ROOT / "data/ens1/out/heldout_by_r1.jsonl")
    byB = load(ROOT / "data/ens1/out/heldout_by_r2.jsonl")
    ownA = own(ROOT / "runs/r1r2/r1/rr/cands_dev.jsonl")
    ownB = own(ROOT / "runs/r1r2/r2/rr/cands_dev.jsonl")
    held = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
    qs = []
    for i, rb in byB.items():
        ra = byA[i]
        assert rb["db_id"] in held
        assert [c["sql"] for c in ra["cands"]] == [c["sql"] for c in rb["cands"]]
        cands = []
        for ca, cb in zip(ra["cands"], rb["cands"]):
            sql = cb["sql"]
            rA, gA = ownA[i].get(sql, (None, None))
            rB, gB = ownB[i].get(sql, (None, None))
            assert rA is not None or rB is not None
            n_rows, n_cols, types = shape(cb.get("res"))
            cands.append({"sql": sql, "s_A": ca["rr"], "s_B": cb["rr"], "rank_A": rA, "rank_B": rB,
                          "cost_A": gA, "cost_B": gB, "status": status(cb), "result": cb.get("res"),
                          "n_rows": n_rows, "n_cols": n_cols, "col_types": types, "class": cb["sig"],
                          "y": int(bool(cb.get("ok")))})
        qs.append({"id": i, "db_id": rb["db_id"], "question": rb["question"], "evidence": rb["context"],
                   "cands": cands})
    qs.sort(key=lambda q: (q["db_id"], int(q["id"].rsplit("_", 1)[-1]) if q["id"].rsplit("_", 1)[-1].isdigit() else 0))
    json.dump({"generator_A": "R1", "generator_B": "R2", "n_questions": len(qs), "questions": qs},
              open(out / "pick_practice.json", "w"), indent=1)
    schemas = {}
    for d in sorted(held):
        con = sqlite3.connect(f"file:{ROOT}/data/raw/bird_train/train_databases/{d}/{d}.sqlite?mode=ro", uri=True)
        schemas[d] = [s for (s,) in con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")]
        con.close()
    json.dump(schemas, open(out / "schemas.json", "w"), indent=1)
    print(len(qs), "questions,", sum(len(q["cands"]) for q in qs), "candidates ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
