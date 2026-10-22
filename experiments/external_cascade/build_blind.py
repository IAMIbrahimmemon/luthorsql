"""Blind System 1 inputs for the external cascade test (PREREG.md): questions without SQL, Spider-format tables, and
the databases (KaggleDBQA's official column descriptions written as BIRD-style database_description CSVs, the same
files the agent's schema tool reads). Gold stays on the Mac.

  python3 experiments/external_cascade/build_blind.py --out data/external_blind
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = ("kaggledbqa_test", "spider_realistic")
DBS = ROOT / "data/sys1/sealed/dbs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for s in SETS for l in open(ROOT / f"data/eval/sealed_{s}.jsonl")]
    kag = {t["db_id"]: t for t in json.load(open(ROOT / "data/raw/kaggledbqa/repo/KaggleDBQA_tables.json"))}
    spi = {t["db_id"]: t for t in json.load(open(ROOT / "data/raw/spider_data/tables.json"))}
    need = sorted({r["db_id"] for r in rows})
    tabs = [kag[d] if d in kag else spi[d] for d in need]
    (a.out / "blind").mkdir(parents=True, exist_ok=True)
    (a.out / "blind/questions.json").write_text(json.dumps(
        [{"question_id": i, "db_id": r["db_id"], "question": r["question"], "evidence": ""} for i, r in enumerate(rows)], indent=1))
    keep = ("db_id", "table_names_original", "table_names", "column_names_original", "column_names", "column_types",
            "primary_keys", "foreign_keys")
    (a.out / "blind/tables.json").write_text(json.dumps([{k: t[k] for k in keep} for t in tabs]))
    (a.out / "index_to_id.json").write_text(json.dumps([r["id"] for r in rows]))
    for d in need:
        dst = a.out / "databases" / d
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DBS / d / f"{d}.sqlite", dst / f"{d}.sqlite")
        if d in kag:
            t = kag[d]
            (dst / "database_description").mkdir(exist_ok=True)
            for ti, tn in enumerate(t["table_names_original"]):
                with open(dst / "database_description" / f"{tn}.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["original_column_name", "column_name", "column_description", "data_format", "value_description"])
                    for (tj, cn), desc in zip(t["column_names_original"], t["column_descriptions"]):
                        if tj == ti:
                            w.writerow([cn, "", desc or "", "", ""])
    print(len(rows), "questions,", len(need), "databases")


if __name__ == "__main__":
    main()
