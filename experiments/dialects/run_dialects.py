"""System 1 on other SQL dialects: BIRD Mini-Dev (500 dev questions) in SQLite, MySQL and PostgreSQL.

No retraining and no new model run. System 1's frozen package output on BIRD dev (every candidate with its validator
score; the validator reads SQL text, not results) is reused. For a target dialect, every candidate is translated
from SQLite with sqlglot, table and column names are matched to the target catalog case-insensitively, the
translated candidates are executed on the target database, and the pick rule is the package's own: the highest
validator score among candidates that run and return a non-null result. Scoring follows BIRD's Mini-Dev rule: the
set of result rows equals the set returned by the official gold query for that dialect (30 s per query).

  uv run --with sqlglot==30.21.0 --with psycopg2-binary --with pymysql python experiments/dialects/run_dialects.py \
      --minidev <MINIDEV dir> --cands <predict_dev.json.cands.jsonl> --sqlite-dbs <dev_databases> \
      --pg "dbname=BIRD" --mysql-socket <sock> --out experiments/dialects/result.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp


def pick(cands):
    u = [c for c in cands if c.get("sql")]
    if not u:
        return None
    live = [c for c in u if c.get("runs") and c.get("nonnull")] or u
    return max(live, key=lambda c: c.get("rr", -1e9))


class Engine:
    def __init__(self, kind, a):
        self.kind = kind
        if kind == "postgres":
            import psycopg2
            self.con = psycopg2.connect(a.pg)
            self.con.autocommit = True
            cur = self.con.cursor()
            cur.execute("SET statement_timeout = 30000")
            cur.execute("select table_name, column_name, data_type from information_schema.columns where table_schema='public'")
        else:
            import pymysql
            self.con = pymysql.connect(unix_socket=a.mysql_socket, user="root", database="BIRD", autocommit=True)
            cur = self.con.cursor()
            cur.execute("SET SESSION max_execution_time = 30000")
            cur.execute("SET SESSION sql_mode = (SELECT REPLACE(@@sql_mode, 'ONLY_FULL_GROUP_BY', ''))")
            cur.execute("select table_name, column_name, data_type from information_schema.columns where table_schema='BIRD'")
        self.tables, self.columns, types = {}, {}, {}
        for t, c, ty in cur.fetchall():
            self.tables[t.lower()] = t
            self.columns.setdefault(c.lower(), c)
            types.setdefault(c.lower(), set()).add(ty.lower())
        self.types = {c: next(iter(v)) for c, v in types.items() if len(v) == 1}  # unambiguous column types only

    def translate(self, sql):
        tree = sqlglot.parse_one(sql, read="sqlite")
        for t in tree.find_all(exp.Table):
            real = self.tables.get(t.name.lower())
            if real:
                t.set("this", exp.to_identifier(real, quoted=True))
        for c in tree.find_all(exp.Column):
            real = self.columns.get(c.name.lower())
            if real:
                c.set("this", exp.to_identifier(real, quoted=True))
        if self.kind == "postgres":
            tree = self.pg_rules(tree)
        return tree.sql(dialect=self.kind)

    TEXT = ("text", "character varying", "character")
    TIME = ("date", "timestamp without time zone", "timestamp with time zone", "time without time zone")

    def pg_rules(self, tree):
        """SQLite coerces types where PostgreSQL does not: compare text columns with string literals, read dates as
        text for LIKE / SUBSTRING / comparisons with numbers or strings, LIKE ignores case, JULIANDAY via epoch."""
        def ty(node):
            return self.types.get(node.name.lower()) if isinstance(node, exp.Column) else None

        def as_text(node):
            return exp.Cast(this=node.copy(), to=exp.DataType.build("text"))

        for cmp in list(tree.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            for side, other in (("this", "expression"), ("expression", "this")):
                col, lit = cmp.args.get(side), cmp.args.get(other)
                if isinstance(col, exp.Column) and isinstance(lit, exp.Literal):
                    t = ty(col)
                    if t in self.TEXT and not lit.is_string:
                        cmp.set(other, exp.Literal.string(lit.this))
                    elif t in self.TIME:
                        cmp.set(side, as_text(col))
                        cmp.set(other, exp.Literal.string(lit.this))
        for b in list(tree.find_all(exp.Between)):
            t = ty(b.this)
            if t in self.TEXT or t in self.TIME:
                if t in self.TIME:
                    b.set("this", as_text(b.this))
                for k in ("low", "high"):
                    v = b.args.get(k)
                    if isinstance(v, exp.Literal) and not v.is_string:
                        b.set(k, exp.Literal.string(v.this))
        for like in list(tree.find_all(exp.Like)):
            if ty(like.this) not in (None,) + self.TEXT:
                like.set("this", as_text(like.this))
            like.replace(exp.ILike(this=like.this, expression=like.expression))
        for sub in list(tree.find_all(exp.Substring)):
            if ty(sub.this) not in (None,) + self.TEXT:
                sub.set("this", as_text(sub.this))
        for f in list(tree.find_all(exp.Anonymous)):
            if f.name.upper() == "JULIANDAY" and f.expressions:
                arg = exp.Cast(this=f.expressions[0].copy(), to=exp.DataType.build("timestamp"))
                f.replace(exp.paren(exp.Add(this=exp.Div(this=exp.Extract(this=exp.var("EPOCH"), expression=arg),
                                                         expression=exp.Literal.number("86400.0")),
                                            expression=exp.Literal.number("2440587.5"))))
        return tree

    def run(self, sql):
        cur = self.con.cursor()
        try:
            cur.execute(sql)
            return cur.fetchall()
        except Exception:
            if self.kind == "postgres":
                self.con.rollback() if not self.con.autocommit else None
            return None


def sqlite_run(db_root, db, sql):
    con = sqlite3.connect(f"file:{db_root}/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    t = threading.Timer(30, con.interrupt)
    t.start()
    try:
        return con.execute(sql).fetchall()
    except Exception:
        return None
    finally:
        t.cancel()
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minidev", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--sqlite-dbs", type=Path, required=True)
    ap.add_argument("--pg", default="dbname=BIRD")
    ap.add_argument("--mysql-socket", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    s1 = {}
    for ln in open(a.cands):
        r = json.loads(ln)
        s1[int(r["index"])] = r
    out = {"n": 500, "rule": "set(pred rows) == set(gold rows), per dialect, official Mini-Dev gold"}
    sq = json.load(open(a.minidev / "mini_dev_sqlite.json"))
    ok = [0, 0]
    for q in sq:
        p = pick(s1[q["question_id"]]["cands"])
        g = sqlite_run(a.sqlite_dbs, q["db_id"], q["SQL"])
        r = sqlite_run(a.sqlite_dbs, q["db_id"], p["sql"]) if p else None
        ok[0] += g is not None and r is not None and set(r) == set(g)
        ok[1] += g is None
    out["sqlite"] = {"ex": round(100 * ok[0] / len(sq), 2), "gold_failed": ok[1]}
    print("sqlite", out["sqlite"], flush=True)
    for kind, fname in (("postgres", "mini_dev_postgresql.json"), ("mysql", "mini_dev_mysql.json")):
        e = Engine(kind, a)
        qs = json.load(open(a.minidev / fname))
        n_ok = n_gold_fail = n_tr_fail = n_cands = n_cands_run = 0
        per_diff = Counter()
        tot_diff = Counter()
        for q in qs:
            tot_diff[q["difficulty"]] += 1
            g = e.run(q["SQL"])
            if g is None:
                n_gold_fail += 1
            cands = []
            for c in s1[q["question_id"]]["cands"]:
                if not c.get("sql"):
                    continue
                n_cands += 1
                try:
                    t = e.translate(c["sql"])
                except Exception:
                    n_tr_fail += 1
                    continue
                res = e.run(t)
                n_cands_run += res is not None
                cands.append({"sql": t, "rr": c.get("rr", -1e9), "runs": res is not None,
                              "nonnull": bool(res) and any(v is not None for row in res for v in row), "res": res})
            p = pick(cands)
            right = p is not None and p["res"] is not None and g is not None and set(p["res"]) == set(g)
            n_ok += right
            per_diff[q["difficulty"]] += right
        out[kind] = {"ex": round(100 * n_ok / len(qs), 2), "gold_failed": n_gold_fail,
                     "candidates": n_cands, "translate_failed": n_tr_fail, "candidates_ran": n_cands_run,
                     "by_difficulty": {d: round(100 * per_diff[d] / tot_diff[d], 2) for d in tot_diff}}
        print(kind, out[kind], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
