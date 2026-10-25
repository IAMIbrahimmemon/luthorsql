"""Load BIRD's dev SQLite databases into Snowflake (one schema per database, names kept exactly, quoted).

Credentials come from ~/.config/luthor/snowflake.json (account, user, private_key_file, warehouse, database); nothing
is written to the repository. Only BIRD's public dev data is uploaded.

  uv run --with "snowflake-connector-python[pandas]" python experiments/dialects/load_snowflake.py --sqlite-dbs <dev_databases>
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import snowflake.connector as sf
from snowflake.connector.pandas_tools import write_pandas


def connect():
    c = json.load(open(os.path.expanduser("~/.config/luthor/snowflake.json")))
    con = sf.connect(account=c["account"], user=c["user"], authenticator="SNOWFLAKE_JWT",
                     private_key_file=c["private_key_file"], warehouse=c["warehouse"], database=c["database"])
    return con, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite-dbs", type=Path, required=True)
    a = ap.parse_args()
    con, c = connect()
    cur = con.cursor()
    for d in sorted(p for p in a.sqlite_dbs.iterdir() if p.is_dir()):
        cur.execute(f'CREATE OR REPLACE SCHEMA "{c["database"]}"."{d.name}"')
        src = sqlite3.connect(f"file:{d / (d.name + '.sqlite')}?mode=ro", uri=True)
        src.text_factory = lambda b: b.decode(errors="replace")
        tables = [r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for t in tables:
            df = pd.read_sql_query(f'SELECT * FROM "{t}"', src)
            for col in df.columns:  # columns SQLite stores with mixed types become text, as SQLite would compare them
                if df[col].dtype == object:
                    df[col] = df[col].map(lambda v: None if v is None else str(v))
            ok, _, n, _ = write_pandas(con, df, table_name=t, database=c["database"], schema=d.name,
                                       auto_create_table=True, overwrite=True, quote_identifiers=True)
            print(d.name, t, n, "rows", "ok" if ok else "FAILED", flush=True)
        src.close()


if __name__ == "__main__":
    main()
