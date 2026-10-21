"""Build the clean sandbox for the System 2 BIRD dev run (runs/phase2/system2_dev/PREREG.md), OUTSIDE the repo:
  <box>/dbs/<db>/<db>.sqlite + database_description/*.csv   (copies; no dev.json, no gold anywhere in the box)
  <box>/TOOLS/sql.py                                        (copy of scripts/system2/sql_tool.py)
  <box>/chunks/A0/cNN.jsonl   question_id, db_id, question, evidence                 (Opus alone)
  <box>/chunks/A1/cNN.jsonl   + candidates[{label, sql, result, n_of_32}] (top 8)  (Opus + System 1 handoff)
  <box>/answers/A0, <box>/answers/A1, <box>/work                                  (agent outputs)
Candidates mirror scripts/analysis/hybrid/s1_candidates.py (the 2026-10-12 DECISION builder) on look #1's frozen pool
runs/phase1/devlook/devlook/poolA_by_valhard.jsonl, label-free (no ok / gold field is read or written).
Order: all 1,534 questions shuffled with seed 20261017, chunks of 25; the same chunks for A0 and A1.

  .venv/bin/python scripts/system2/build_dev_sandbox.py --box <dir outside the repo>
"""
import argparse, hashlib, json, random, shutil, string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "data/bench/dev_20240627"
POOL = ROOT / "runs/phase1/devlook/devlook/poolA_by_valhard.jsonl"
TOOL = ROOT / "scripts/system2/sql_tool.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261017)
    a = ap.parse_args()
    box = a.box.resolve()
    if str(box).startswith(str(ROOT)):
        raise SystemExit("the sandbox must be outside the repo")
    if box.exists():
        raise SystemExit(f"{box} exists; refusing to overwrite")
    for sub in ("dbs", "TOOLS", "chunks/A0", "chunks/A1", "answers/A0", "answers/A1", "work"):
        (box / sub).mkdir(parents=True)
    for d in sorted((DEV / "dev_databases").iterdir()):
        if d.is_dir():
            (box / "dbs" / d.name).mkdir()
            shutil.copy2(d / f"{d.name}.sqlite", box / "dbs" / d.name / f"{d.name}.sqlite")
            if (d / "database_description").is_dir():
                shutil.copytree(d / "database_description", box / "dbs" / d.name / "database_description")
    shutil.copy2(TOOL, box / "TOOLS/sql.py")
    q = json.load(open(DEV / "dev.json"))
    rows = [{"question_id": f"bird_dev_{i}", "db_id": t["db_id"], "question": t["question"],
             "evidence": t.get("evidence") or ""} for i, t in enumerate(q)]          # no SQL field copied
    have = {}
    for ln in open(POOL):
        r = json.loads(ln)
        c = [x for x in r["cands"] if x.get("sql") and x.get("src") in ("r1", "both", "k32", None)]  # dev pool tags R1's own list "k32"
        lv = [x for x in c if x.get("runs") and x.get("nonnull")] or c
        groups = {}
        for x in sorted(lv, key=lambda x: -x.get("rr", -1e9)):
            groups.setdefault(x.get("sig") or x["sql"], []).append(x)
        have[r["id"]] = [{"label": string.ascii_uppercase[k], "sql": g[0]["sql"], "result": g[0].get("res", ""),
                          "n_of_32": len(g)} for k, g in enumerate(list(groups.values())[:8])]
    order = rows[:]
    random.Random(a.seed).shuffle(order)
    chunks = [order[i:i + a.chunk] for i in range(0, len(order), a.chunk)]
    for k, ch in enumerate(chunks):
        name = f"c{k:02d}.jsonl"
        (box / "chunks/A0" / name).write_text("".join(json.dumps(r) + "\n" for r in ch))
        (box / "chunks/A1" / name).write_text("".join(json.dumps({**r, "candidates": have.get(r["question_id"], [])}) + "\n" for r in ch))
    text = "".join(p.read_text() for p in sorted((box / "chunks").rglob("*.jsonl")))
    leak = [k for k in ('"SQL"', '"ok"', '"gold_sql"', '"gold"', '"difficulty"') if k in text]
    assert not leak, f"label field in chunks: {leak}"
    n_c = sum(1 for r in rows if have.get(r["question_id"]))
    print(f"sandbox {box}: {len(rows)} questions, {len(chunks)} chunks of <= {a.chunk}; {n_c} with candidates; "
          f"dbs {len(list((box / 'dbs').iterdir()))}")


if __name__ == "__main__":
    main()
