"""Sandbox for the Opus + System 1 handoff on the 396 DECISION questions outside the 2026-10-12 sample
(runs/phase2/decision_rest/PREREG.md), OUTSIDE the repo:
  <box>/dbs/<db>/<db>.sqlite + database_description/*.csv   (the 8 DECISION BIRD-train DBs)
  <box>/TOOLS/sql.py                                        (copy of scripts/system2/sql_tool.py)
  <box>/chunks/A1/cNN.jsonl   rows of runs/analysis/hybrid/s1_cands_decision.jsonl (label-free), DECISION order
  <box>/answers/A1, <box>/work

  .venv/bin/python scripts/system2/build_decision_sandbox.py --box <dir outside the repo>
"""
import argparse, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/raw/bird_train/train_databases"
CANDS = ROOT / "runs/analysis/hybrid/s1_cands_decision.jsonl"
SAMPLE = ROOT / "runs/analysis/hybrid/opus_s1_raw/sample_ids.json"
TOOL = ROOT / "scripts/system2/sql_tool.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=25)
    a = ap.parse_args()
    box = a.box.resolve()
    if str(box).startswith(str(ROOT)):
        raise SystemExit("the sandbox must be outside the repo")
    if box.exists():
        raise SystemExit(f"{box} exists; refusing to overwrite")
    sample = set(json.load(open(SAMPLE)))
    rows = [r for r in map(json.loads, open(CANDS)) if r["question_id"] not in sample]
    for sub in ("dbs", "TOOLS", "chunks/A1", "answers/A1", "work"):
        (box / sub).mkdir(parents=True)
    for db in sorted({r["db_id"] for r in rows}):
        (box / "dbs" / db).mkdir()
        shutil.copy2(TRAIN / db / f"{db}.sqlite", box / "dbs" / db / f"{db}.sqlite")
        if (TRAIN / db / "database_description").is_dir():
            shutil.copytree(TRAIN / db / "database_description", box / "dbs" / db / "database_description")
    shutil.copy2(TOOL, box / "TOOLS/sql.py")
    chunks = [rows[i:i + a.chunk] for i in range(0, len(rows), a.chunk)]
    for k, ch in enumerate(chunks):
        (box / "chunks/A1" / f"c{k:02d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ch))
    for r in rows:
        assert set(r) == {"question_id", "db_id", "question", "evidence", "candidates"}, r.keys()
        for c in r["candidates"]:
            assert set(c) == {"label", "sql", "result", "n_of_32"}, c.keys()
    print(f"sandbox {box}: {len(rows)} questions, {len(chunks)} chunks; "
          f"{sum(1 for r in rows if r['candidates'])} with candidates; dbs {sorted(p.name for p in (box / 'dbs').iterdir())}")


if __name__ == "__main__":
    main()
