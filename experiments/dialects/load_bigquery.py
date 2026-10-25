"""Load BIRD's dev SQLite databases into BigQuery (project from --project, one dataset per database).

Uses Application Default Credentials (gcloud auth application-default login); only BIRD's public dev data is uploaded.
Column names are kept when BigQuery's flexible column names allow them; any other name is mapped to a safe one and
the mapping is written to bigquery_names.json for the scorer.

  uv run --with google-cloud-bigquery --with pandas --with pyarrow python experiments/dialects/load_bigquery.py \
      --sqlite-dbs <dev_databases> --project luthorsql
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

OK = re.compile(r"^[^.!\"$()*,/;?@\[\\\]^`{}~]+$")  # characters BigQuery's flexible column names do not allow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite-dbs", type=Path, required=True)
    ap.add_argument("--project", required=True)
    a = ap.parse_args()
    bq = bigquery.Client(project=a.project)
    names = {}
    for d in sorted(p for p in a.sqlite_dbs.iterdir() if p.is_dir()):
        ds = bigquery.Dataset(f"{a.project}.{d.name}")
        ds.location = "US"
        bq.create_dataset(ds, exists_ok=True)
        src = sqlite3.connect(f"file:{d / (d.name + '.sqlite')}?mode=ro", uri=True)
        src.text_factory = lambda b: b.decode(errors="replace")
        tables = [r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for t in tables:
            df = pd.read_sql_query(f'SELECT * FROM "{t}"', src)
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].map(lambda v: None if v is None else str(v))
            mapping = {}
            for col in df.columns:
                safe = col if OK.match(col) else re.sub(r"[^A-Za-z0-9_ -]", "_", col)
                mapping[col] = safe
            df = df.rename(columns=mapping)
            names.setdefault(d.name, {})[t] = {k: v for k, v in mapping.items() if k != v}
            job = bq.load_table_from_dataframe(df, f"{a.project}.{d.name}.{t}",
                                               job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
            job.result()
            print(d.name, t, len(df), "rows ok", flush=True)
        src.close()
    Path(__file__).with_name("bigquery_names.json").write_text(json.dumps(names, indent=1))


if __name__ == "__main__":
    main()
