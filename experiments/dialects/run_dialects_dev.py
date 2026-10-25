"""System 1 in MySQL and PostgreSQL on all 1,534 BIRD dev questions.

BIRD publishes MySQL and PostgreSQL gold only for Mini-Dev (run_dialects.py). Its MySQL and PostgreSQL dumps hold all
11 dev databases, so here the reference is the original SQLite gold, executed on the SQLite databases: a prediction
is right when its rows, executed in the target dialect, equal the gold rows as sets after normalising values
(numbers to floats rounded to 6 places, dates and times to ISO strings, bytes to text), since the engines return
different Python types for the same value. The SQLite row uses the same normalisation, so all three columns are
scored alike. System 1's candidates and validator scores come from the frozen package run on dev (no new model run);
translation and pick rule as in run_dialects.py.

  uv run --with sqlglot==30.21.0 --with psycopg2-binary --with pymysql python experiments/dialects/run_dialects_dev.py \
      --dev <dev.json> --cands <predict_dev.json.cands.jsonl> --sqlite-dbs <dev_databases> \
      --mysql-socket <sock> --out experiments/dialects/result_dev.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import decimal
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_dialects import Engine, pick, sqlite_run  # noqa: E402

import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402


class DuckEngine:
    """Native DuckDB files built by build_duckdb.py, one per dev database; names matched per database."""
    kind = "duckdb"

    def __init__(self, a):
        import duckdb
        self.cons, self.cat = {}, {}
        for f in sorted(a.duckdb_dir.glob("*.duckdb")):
            con = duckdb.connect(str(f), read_only=True)
            tables, columns = {}, {}
            for t, c in con.execute("SELECT table_name, column_name FROM information_schema.columns").fetchall():
                tables[t.lower()] = t
                columns.setdefault(c.lower(), c)
            self.cons[f.stem], self.cat[f.stem] = con, (tables, columns)
        self.db = None

    def use(self, db):
        self.db = db

    def translate(self, sql):
        tables, columns = self.cat[self.db]
        tree = sqlglot.parse_one(sql, read="sqlite")
        for t in tree.find_all(exp.Table):
            if t.name.lower() in tables:
                t.set("this", exp.to_identifier(tables[t.name.lower()], quoted=True))
        for c in tree.find_all(exp.Column):
            if c.name.lower() in columns:
                c.set("this", exp.to_identifier(columns[c.name.lower()], quoted=True))
        return tree.sql(dialect="duckdb")

    def run(self, sql):
        import threading
        con = self.cons[self.db]
        timer = threading.Timer(30, con.interrupt)
        timer.start()
        try:
            return con.execute(sql).fetchall()
        except Exception:
            return None
        finally:
            timer.cancel()


class SnowEngine:
    """Snowflake: one schema per dev database (load_snowflake.py); tables qualified with their schema, names quoted
    exactly. Credentials from ~/.config/luthor/snowflake.json (key pair)."""
    kind = "snowflake"

    def __init__(self, a, catalog=None):
        import os
        import snowflake.connector as sf
        c = json.load(open(os.path.expanduser("~/.config/luthor/snowflake.json")))
        self.con = sf.connect(account=c["account"], user=c["user"], authenticator="SNOWFLAKE_JWT",
                              private_key_file=c["private_key_file"], warehouse=c["warehouse"], database=c["database"])
        cur = self.con.cursor()
        cur.execute("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 30")
        if catalog is None:
            catalog = {}
            cur.execute("SELECT table_schema, table_name, column_name FROM information_schema.columns")
            for sch, t, col in cur.fetchall():
                tables, columns = catalog.setdefault(sch, ({}, {}))
                tables[t.lower()] = t
                columns.setdefault(col.lower(), col)
        self.cat, self.db = catalog, None

    def use(self, db):
        self.db = db

    def translate(self, sql):
        tables, columns = self.cat[self.db]
        tree = sqlglot.parse_one(sql, read="sqlite")
        for t in tree.find_all(exp.Table):
            if t.name.lower() in tables:
                t.set("this", exp.to_identifier(tables[t.name.lower()], quoted=True))
                t.set("db", exp.to_identifier(self.db, quoted=True))
        for c in tree.find_all(exp.Column):
            if c.name.lower() in columns:
                c.set("this", exp.to_identifier(columns[c.name.lower()], quoted=True))
        return tree.sql(dialect="snowflake")

    def run(self, sql):
        try:
            cur = self.con.cursor()
            cur.execute(sql)
            return cur.fetchall()
        except Exception:
            return None


class BQEngine:
    """BigQuery: one dataset per dev database (load_bigquery.py), tables qualified project.dataset.table; column
    names BigQuery would not accept were renamed at load time (bigquery_names.json) and are mapped here."""
    kind = "bigquery"

    def __init__(self, a, client=None, catalog=None):
        from google.cloud import bigquery
        self.bq = client or bigquery.Client(project=a.bq_project)
        self.project = a.bq_project
        self.cfg = bigquery.QueryJobConfig(job_timeout_ms=30000, use_query_cache=True)
        if catalog is None:
            renamed = json.load(open(Path(__file__).with_name("bigquery_names.json")))
            catalog = {}
            for ds in self.bq.list_datasets():
                db = ds.dataset_id
                tables, columns = {}, {}
                q = f"SELECT table_name, column_name FROM `{self.project}.{db}.INFORMATION_SCHEMA.COLUMNS`"
                for t, col in self.bq.query(q).result():
                    tables[t.lower()] = t
                    columns.setdefault(col.lower(), col)
                for t, m in renamed.get(db, {}).items():
                    for orig, safe in m.items():
                        columns[orig.lower()] = safe
                catalog[db] = (tables, columns)
        self.cat, self.db = catalog, None

    def use(self, db):
        self.db = db

    def translate(self, sql):
        tables, columns = self.cat[self.db]
        tree = sqlglot.parse_one(sql, read="sqlite")
        for t in tree.find_all(exp.Table):
            if t.name.lower() in tables:
                t.set("this", exp.to_identifier(tables[t.name.lower()], quoted=True))
                t.set("db", exp.to_identifier(self.db, quoted=True))
                t.set("catalog", exp.to_identifier(self.project, quoted=True))
        for c in tree.find_all(exp.Column):
            if c.name.lower() in columns:
                c.set("this", exp.to_identifier(columns[c.name.lower()], quoted=True))
        return tree.sql(dialect="bigquery")

    def run(self, sql):
        try:
            return [tuple(r.values()) for r in self.bq.query(sql, job_config=self.cfg).result(timeout=40)]
        except Exception:
            return None


def norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float, decimal.Decimal)):
        return round(float(v), 6)
    if isinstance(v, dt.datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, (dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, dt.timedelta):
        return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).decode(errors="replace")
    if isinstance(v, dict):  # BigQuery STRUCT
        return tuple(norm(x) for x in v.values())
    if isinstance(v, (list, tuple)):  # BigQuery ARRAY
        return tuple(norm(x) for x in v)
    return v


def rows(res):
    return None if res is None else {tuple(norm(x) for x in r) for r in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--sqlite-dbs", type=Path, required=True)
    ap.add_argument("--pg", default="dbname=BIRD")
    ap.add_argument("--mysql-socket", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engines", nargs="+", default=["postgres", "mysql"])
    ap.add_argument("--duckdb-dir", type=Path)
    ap.add_argument("--bq-project", default="luthorsql")
    a = ap.parse_args()
    dev = json.load(open(a.dev))
    s1 = {int(json.loads(l)["index"]): json.loads(l) for l in open(a.cands)}
    gold = [rows(sqlite_run(a.sqlite_dbs, q["db_id"], q["SQL"])) for q in dev]
    out = {"n": len(dev), "gold_failed_sqlite": sum(g is None for g in gold),
           "rule": "normalised row sets equal to the SQLite gold rows"}
    ok = 0
    by = Counter()
    tot = Counter(q.get("difficulty") for q in dev)
    for i, q in enumerate(dev):
        p = pick(s1[i]["cands"])
        r = rows(sqlite_run(a.sqlite_dbs, q["db_id"], p["sql"])) if p else None
        right = r is not None and gold[i] is not None and r == gold[i]
        ok += right
        by[q.get("difficulty")] += right
    out["sqlite"] = {"ex": round(100 * ok / len(dev), 2), "by_difficulty": {d: round(100 * by[d] / tot[d], 2) for d in tot}}
    print("sqlite", out["sqlite"], flush=True)
    prev = json.loads(a.out.read_text()) if a.out.exists() else {}
    out = {**prev, **out}
    for kind in a.engines:
        if kind in ("snowflake", "bigquery"):
            out[kind] = run_parallel(a, dev, s1, gold, tot, kind)
            print(kind, out[kind], flush=True)
            continue
        e = DuckEngine(a) if kind == "duckdb" else Engine(kind, a)
        ok, n_c, n_run = 0, 0, 0
        by = Counter()
        for i, q in enumerate(dev):
            if hasattr(e, "use"):
                e.use(q["db_id"])
            cands = []
            for c in s1[i]["cands"]:
                if not c.get("sql"):
                    continue
                n_c += 1
                try:
                    t = e.translate(c["sql"])
                except Exception:
                    continue
                res = e.run(t)
                n_run += res is not None
                cands.append({"sql": t, "rr": c.get("rr", -1e9), "runs": res is not None,
                              "nonnull": bool(res) and any(v is not None for row in res for v in row), "res": res})
            p = pick(cands)
            right = p is not None and p["res"] is not None and gold[i] is not None and rows(p["res"]) == gold[i]
            ok += right
            by[q.get("difficulty")] += right
        out[kind] = {"ex": round(100 * ok / len(dev), 2), "candidates": n_c, "candidates_ran": n_run,
                     "by_difficulty": {d: round(100 * by[d] / tot[d], 2) for d in tot}}
        print(kind, out[kind], flush=True)
    a.out.write_text(json.dumps(out, indent=1))


def judge(e, i, q, s1, gold):
    e.use(q["db_id"])
    cands, n_c, n_run = [], 0, 0
    for c in s1[i]["cands"]:
        if not c.get("sql"):
            continue
        n_c += 1
        try:
            t = e.translate(c["sql"])
        except Exception:
            continue
        res = e.run(t)
        n_run += res is not None
        cands.append({"sql": t, "rr": c.get("rr", -1e9), "runs": res is not None,
                      "nonnull": bool(res) and any(v is not None for row in res for v in row), "res": res})
    p = pick(cands)
    right = p is not None and p["res"] is not None and gold[i] is not None and rows(p["res"]) == gold[i]
    return right, n_c, n_run


def run_parallel(a, dev, s1, gold, tot, kind):
    workers = 32 if kind == "bigquery" else 8  # BigQuery allows 100 concurrent interactive queries
    import threading
    from concurrent.futures import ThreadPoolExecutor
    first = SnowEngine(a) if kind == "snowflake" else BQEngine(a)
    local = threading.local()

    def eng():
        if not hasattr(local, "e"):
            local.e = SnowEngine(a, first.cat) if kind == "snowflake" else BQEngine(a, first.bq, first.cat)
        return local.e

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda iq: judge(eng(), iq[0], iq[1], s1, gold), enumerate(dev)))
    by = Counter()
    for (right, _, _), q in zip(res, dev):
        by[q.get("difficulty")] += right
    return {"ex": round(100 * sum(r[0] for r in res) / len(dev), 2), "candidates": sum(r[1] for r in res),
            "candidates_ran": sum(r[2] for r in res), "by_difficulty": {d: round(100 * by[d] / tot[d], 2) for d in tot}}


if __name__ == "__main__":
    main()
