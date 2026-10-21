"""Round-6 step 0 (redo): schema size, schema-linking recall and undocumented joins on BIRD dev (dev set).

  - size: full DDL (our format) per database, in characters and Qwen tokens, against an 8k-token window
  - linking: share of questions whose gold tables / columns survive (a) the round-5 table cut at several
    budgets and (b) column pruning that keeps every table, its keys, and the top-k question-matched columns
  - joins: gold join conditions that are not declared foreign keys (per database's schema card)

  .venv/bin/python scripts/schema6_stats.py   # -> data/eval/schema6_stats.json
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.prompt import _words, schema_ddl, table_ddl  # noqa: E402

DEV = Path("data/eval/dev6_bird.jsonl")
CARDS = Path("data/eval_plans_v2/bird_dev/cards")


def gold_refs(card, sql):
    """Gold tables, (table, column) pairs and join conditions, resolved against the card (best effort)."""
    names = {t.name.lower(): t for t in card.tables}
    cols = {(t.name.lower(), c.name.lower()) for t in card.tables for c in t.columns}
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    alias = {}
    for tb in tree.find_all(exp.Table):
        n = tb.name.lower()
        if n in names:
            alias[(tb.alias or tb.name).lower()] = n
            alias[n] = n
    tables = set(alias.values())
    used = set()
    for c in tree.find_all(exp.Column):
        cn, tq = c.name.lower(), (c.table or "").lower()
        if tq and tq in alias:
            if (alias[tq], cn) in cols:
                used.add((alias[tq], cn))
        else:
            for t in tables:
                if (t, cn) in cols:
                    used.add((t, cn))
    joins = []
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            l, r = eq.left, eq.right
            if isinstance(l, exp.Column) and isinstance(r, exp.Column) and l.table and r.table:
                a, b = alias.get(l.table.lower()), alias.get(r.table.lower())
                if a and b and a != b:
                    joins.append(((a, l.name.lower()), (b, r.name.lower())))
    return tables, used, joins


def pruned_columns(card, question, context, k):
    """Every table kept; per table its keys, FK columns and the k columns best matching the question."""
    qw = _words(question + " " + context)
    fkc = {(t1.lower(), c1.lower()) for t1, c1, t2, c2 in card.fks} | {(t2.lower(), c2.lower()) for t1, c1, t2, c2 in card.fks}
    keep = set()
    for t in card.tables:
        sc = sorted(t.columns, key=lambda c: -len(qw & _words(c.name + " " + (c.description or ""))))
        for i, c in enumerate(sc):
            key = (t.name.lower(), c.name.lower())
            if c.pk or key in fkc or i < k:
                keep.add(key)
    return keep


def main():
    dev = [json.loads(l) for l in open(DEV)]
    cards = {}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        ntok = lambda s: len(tok(s)["input_ids"])
    except Exception:
        ntok = lambda s: round(len(s) / 3.3)
    size = {}
    for db in sorted({d["db_id"] for d in dev}):
        c = cards[db] = SchemaCard.load(CARDS / f"{db}.json")
        full = "\n\n".join(table_ddl(c, t) for t in c.tables)
        size[db] = {"tables": len(c.tables), "columns": sum(len(t.columns) for t in c.tables), "chars": len(full),
                    "tokens": ntok(full), "declared_fks": len(c.fks)}
    budgets = [9000, 16000, 24000]
    tab_ok = collections.Counter()
    col_ok = collections.Counter()
    col_size = collections.defaultdict(list)
    joins_total = joins_undeclared = q_with_undeclared = parsed = 0
    und_examples = []
    for d in dev:
        c = cards[d["db_id"]]
        refs = gold_refs(c, d["gold_sql"])
        if refs is None:
            continue
        parsed += 1
        tables, used, joins = refs
        for b in budgets:
            shown = set(t.lower() for t in re.findall(r'CREATE TABLE "?([^"\s(]+)', schema_ddl(c, d["question"], d["context"], budget=b)))
            tab_ok[b] += tables <= shown
        for k in (5, 10, 15):
            keep = pruned_columns(c, d["question"], d["context"], k)
            col_ok[k] += used <= keep
            col_size[k].append(len(keep))
        declared = {frozenset([(a.lower(), b.lower()), (x.lower(), y.lower())]) for a, b, x, y in c.fks}
        und = [j for j in joins if frozenset(j) not in declared]
        joins_total += len(joins)
        joins_undeclared += len(und)
        q_with_undeclared += bool(und)
        if und and len(und_examples) < 8:
            und_examples.append({"id": d["id"], "db": d["db_id"], "join": [".".join(x) for x in und[0]]})
    n = parsed
    rep = {"questions_parsed": n, "of": len(dev), "db_size": size,
           "gold_tables_all_shown_pct": {f"table_cut_{b}": round(100 * tab_ok[b] / n, 1) for b in budgets},
           "gold_columns_all_kept_pct": {f"all_tables_top{k}_cols": round(100 * col_ok[k] / n, 1) for k in col_ok},
           "mean_columns_kept": {f"top{k}": round(sum(v) / len(v), 1) for k, v in col_size.items()},
           "joins": {"gold_join_conditions": joins_total, "not_declared_fk": joins_undeclared,
                     "pct_undeclared": round(100 * joins_undeclared / max(joins_total, 1), 1),
                     "questions_with_undeclared_join": q_with_undeclared, "examples": und_examples}}
    Path("data/eval/schema6_stats.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
