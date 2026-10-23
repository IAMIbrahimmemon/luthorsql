#!/usr/bin/env python3
"""Read-only SQLite tool for the Opus SQL agent (System 2), rebuilt 2026-10-17 to match the tool used in the
2026-10-10/12 runs (verified by replaying the logged calls: scripts/system2/replay_check.py).

  python3 TOOLS/sql.py <db_id> --schema      # CREATE statements, 3 sample rows per table, column descriptions
  python3 TOOLS/sql.py <db_id> "<SQL>"       # read-only query; prints columns and up to 20 rows

The databases live under DB_ROOT (default: <this file's dir>/../dbs): <db_id>/<db_id>.sqlite and
<db_id>/database_description/*.csv. The tool refuses db ids that are not plain names, opens SQLite read-only, and
(with LUTHOR_SQL_BUDGET_DIR set) counts executions per question id (LUTHOR_SQL_QID) and refuses the 9th.
"""
import csv
import io
import os
import re
import sqlite3
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_ROOT = Path(os.environ.get("LUTHOR_SQL_DB_ROOT", HERE.parent / "dbs"))
MAX_ROWS = 20
TIMEOUT = float(os.environ.get("LUTHOR_SQL_TIMEOUT", "30"))
DESC_MAX = int(os.environ.get("LUTHOR_SQL_DESC_MAX", "300"))


def connect(db):
    if not re.fullmatch(r"[A-Za-z0-9_]+", db):
        sys.exit(f"ERROR: bad db id {db!r}")
    p = DB_ROOT / db / f"{db}.sqlite"
    if not p.exists():
        sys.exit(f"ERROR: unknown database {db}")
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    return con


def schema(db):
    con = connect(db)
    out = []
    for name, sql in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"):
        out.append(sql + ";")
        try:
            for r in con.execute(f'SELECT * FROM "{name}" LIMIT 3'):
                out.append("  -- " + repr(tuple(r)))
        except sqlite3.Error as e:
            out.append(f"  -- (sample rows unavailable: {e})")
    d = DB_ROOT / db / "database_description"
    if d.is_dir():
        for f in sorted(d.glob("*.csv")):
            out.append("")
            out.append(f"# description: {f.name}")
            raw = f.read_bytes().decode("utf-8", errors="replace")
            rows = list(csv.reader(io.StringIO(raw)))
            for row in rows[1:]:
                cells = [c.replace("\r", "").replace("\n", " ").strip() for c in row if c and c.strip()]
                out.append(" | ".join(cells)[:DESC_MAX])  # a blank CSV row prints as an empty line (as the original)
    print("\n".join(out))


def budget():
    bd, q = os.environ.get("LUTHOR_SQL_BUDGET_DIR"), os.environ.get("LUTHOR_SQL_QID")
    if not (bd and q):
        return
    p = Path(bd) / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', q)}.count"
    n = int(p.read_text()) if p.exists() else 0
    if n >= 8:
        sys.exit("ERROR: execution limit reached (8 per question)")
    p.write_text(str(n + 1))


def query(db, sql):
    budget()
    con = connect(db)
    timer = threading.Timer(TIMEOUT, con.interrupt)
    timer.start()
    try:
        cur = con.execute(sql)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        print(f"ERROR: {e}")
        return
    finally:
        timer.cancel()
    print(f"columns: {cols}")
    for r in rows[:MAX_ROWS]:
        print(repr(tuple(r)))
    if len(rows) > MAX_ROWS:
        print("... (more rows)")


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: sql.py <db_id> --schema | sql.py <db_id> "<SQL>"')
    db, arg = sys.argv[1], sys.argv[2]
    schema(db) if arg == "--schema" else query(db, arg)


if __name__ == "__main__":
    main()
