"""Blind System 1 inputs for the fresh unseen set (PREREG.md): questions with evidence and no SQL, Spider-format
tables built from the SQLite catalogue, and the databases with their description CSVs. Gold stays on the Mac.

  python3 experiments/fresh_unseen/build_blind.py --items data/sys1/fresh/items.jsonl --dbs data/sys1/fresh/dbs \
      --out data/sys1/fresh_blind
"""
import argparse
import json
import shutil
import sqlite3
from pathlib import Path


def spider_tables(path: Path, db: str) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
    cols, types, pks, fks, idx = [[-1, "*"]], ["text"], [], [], {}
    for ti, t in enumerate(tabs):
        for cid, name, typ, _nn, _d, pk in con.execute(f'PRAGMA table_info("{t}")'):
            idx[(t.lower(), name.lower())] = len(cols)
            cols.append([ti, name])
            ty = (typ or "").lower()
            types.append("number" if any(k in ty for k in ("int", "real", "num", "dec", "double", "float")) else "text")
            if pk:
                pks.append(idx[(t.lower(), name.lower())])
    for t in tabs:
        for row in con.execute(f'PRAGMA foreign_key_list("{t}")'):
            _id, _seq, ref, frm, to = row[:5]
            a, b = idx.get((t.lower(), frm.lower())), idx.get((ref.lower(), (to or "").lower()))
            if a is not None and b is not None:
                fks.append([a, b])
    con.close()
    nice = lambda s: s.replace("_", " ").lower()
    return {"db_id": db, "table_names_original": tabs, "table_names": [nice(t) for t in tabs],
            "column_names_original": cols, "column_names": [[i, nice(c) if i >= 0 else "*"] for i, c in cols],
            "column_types": types, "primary_keys": pks, "foreign_keys": fks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--dbs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.items)]
    need = sorted({r["db_id"] for r in rows})
    (a.out / "blind").mkdir(parents=True, exist_ok=True)
    (a.out / "blind/questions.json").write_text(json.dumps(
        [{"question_id": i, "db_id": r["db_id"], "question": r["question"], "evidence": r.get("evidence", "")}
         for i, r in enumerate(rows)], indent=1))
    (a.out / "blind/tables.json").write_text(json.dumps([spider_tables(a.dbs / d / f"{d}.sqlite", d) for d in need]))
    (a.out / "index_to_id.json").write_text(json.dumps([r["id"] for r in rows]))
    for d in need:
        dst = a.out / "databases" / d
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.dbs / d / f"{d}.sqlite", dst / f"{d}.sqlite")
        if (a.dbs / d / "database_description").is_dir():
            shutil.copytree(a.dbs / d / "database_description", dst / "database_description", dirs_exist_ok=True)
    assert "SQL" not in (a.out / "blind/questions.json").read_text() and '"sql"' not in (a.out / "blind/questions.json").read_text()
    print(len(rows), "questions,", len(need), "databases")


if __name__ == "__main__":
    main()
