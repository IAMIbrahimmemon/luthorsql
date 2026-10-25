"""Convert BIRD's dev SQLite databases to native DuckDB files (one per database), for run_dialects_dev.py --engines duckdb.

Each table is copied with DuckDB's sqlite scanner into a native DuckDB table, so the DuckDB run uses DuckDB's own
types and functions, not SQLite read through an extension. A table whose stored values do not fit its declared
types is copied with every column as text (sqlite_all_varchar), and is listed in the output.

  uv run --with duckdb python experiments/dialects/build_duckdb.py --sqlite-dbs <dev_databases> --out <dir>
"""
import argparse
import json
from pathlib import Path

import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite-dbs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fallback = {}
    for d in sorted(p for p in a.sqlite_dbs.iterdir() if p.is_dir()):
        dst = a.out / f"{d.name}.duckdb"
        if dst.exists():
            dst.unlink()
        con = duckdb.connect(str(dst))
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"ATTACH '{d / (d.name + '.sqlite')}' AS src (TYPE sqlite, READ_ONLY)")
        tables = [r[0] for r in con.execute("SELECT table_name FROM duckdb_tables() WHERE database_name='src'").fetchall()]
        for t in tables:
            q = t.replace('"', '""')
            try:
                con.execute("SET sqlite_all_varchar=false")
                con.execute(f'CREATE TABLE main."{q}" AS SELECT * FROM src."{q}"')
            except Exception:
                con.execute(f'DROP TABLE IF EXISTS main."{q}"')
                con.execute("SET sqlite_all_varchar=true")
                con.execute(f'CREATE TABLE main."{q}" AS SELECT * FROM src."{q}"')
                fallback.setdefault(d.name, []).append(t)
        con.execute("DETACH src")
        con.close()
        print(d.name, len(tables), "tables", flush=True)
    (a.out / "fallback_tables.json").write_text(json.dumps(fallback, indent=1))
    print("all-text fallback:", fallback)


if __name__ == "__main__":
    main()
