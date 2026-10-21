"""How often does the FK-graph compiler pick a different join path than the gold SQL, how often
are there several shortest paths, and does it change the result? Run on sets whose SQLite
files are local (BIRD dev, Spider train/dev)."""
import collections
import json
import sqlite3
import sys
from collections import deque
from pathlib import Path

import sqlglot
from sqlglot import exp

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from luthor.sq.convert import convert
from luthor.sq.plan import compile_plan, join_tree
from luthor.sq.schema_card import card_from_spider_tables, infer_fks
from prepare_plans import load_questions, load_tables, LAYOUTS, _pick, read_descriptions


def n_shortest(graph, a, b):
    """number of distinct shortest paths a->b"""
    dist, cnt, dq = {a: 0}, {a: 1}, deque([a])
    while dq:
        n = dq.popleft()
        for m, _, _, _ in graph.adj.get(n, []):
            if m not in dist:
                dist[m] = dist[n] + 1; cnt[m] = cnt[n]; dq.append(m)
            elif dist[m] == dist[n] + 1:
                cnt[m] += cnt[n]
    return cnt.get(b, 0)


def gold_tables(sql):
    return {t.name.lower() for t in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table)}


def run(fmt, root, split):
    qs = load_questions(fmt, root, split); tabs = load_tables(fmt, root, split)
    by = collections.defaultdict(list)
    for q in qs: by[q["db_id"]].append(q)
    S = collections.Counter()
    for db, items in by.items():
        lay = LAYOUTS[fmt]
        p = root / _pick(lay["db"], split).format(split=split, db=db)
        if not p.exists() or db not in tabs: continue
        desc = read_descriptions(root / lay["desc"].format(split=split, db=db)) if lay["desc"] else {}
        card = card_from_spider_tables(tabs[db], p, desc); g = card.graph()
        declared = {(a, b, c, d) for a, b, c, d in [(tabs[db]["table_names_original"][tabs[db]["column_names_original"][x][0]], tabs[db]["column_names_original"][x][1], tabs[db]["table_names_original"][tabs[db]["column_names_original"][y][0]], tabs[db]["column_names_original"][y][1]) for x, y in tabs[db]["foreign_keys"]]}
        declared |= {(c, d, a, b) for a, b, c, d in declared}
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True); con.text_factory = lambda b: b.decode("utf-8", "replace")
        for q in items:
            c = convert(q["sql"], g)
            if c.plan is None or len(c.plan.tables()) < 2: continue
            S["multi_table_plans"] += 1
            try:
                tables, edges = join_tree(g, c.plan.tables(), c.plan.via, {(x.table, x.column) for x in c.plan.join_cols})
                sql = compile_plan(c.plan, g)
                ok = set(map(tuple, con.execute(sql).fetchmany(100000))) == set(map(tuple, con.execute(q["sql"]).fetchmany(100000)))
            except Exception:
                S["compile_or_exec_error"] += 1; continue
            same_tables = {t.lower() for t in tables} == gold_tables(q["sql"])
            ambiguous = any(n_shortest(g, tables[0], t) > 1 for t in c.plan.tables()[1:])
            uses_inferred = any(e not in declared for e in edges)
            S[f"tables_same_as_gold={same_tables}"] += 1
            S[f"tables_same_as_gold={same_tables}:exec_ok={ok}"] += 1
            S[f"ambiguous={ambiguous}:exec_ok={ok}"] += 1
            S[f"uses_inferred_fk={uses_inferred}:exec_ok={ok}"] += 1
        con.close()
    return S


for fmt, root, split in [("bird", Path("data/bench/dev_20240627"), "dev"), ("spider", Path("data/raw/spider_data"), "train"),
                         ("spider", Path("data/raw/spider_data"), "dev")]:
    S = run(fmt, root, split)
    print(f"== {fmt} {split}"); [print(f"  {k}: {v}") for k, v in sorted(S.items())]
