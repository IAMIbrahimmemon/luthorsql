"""Convert BIRD / Spider / SynSQL gold SQL into Luthor's bounded query plans.

For every question:
  1. parse the gold SQL and convert it to a Plan (luthor.sq.convert), or record why not;
  2. find literal candidates in question + context and ground every filter value / LIMIT on
     one (a value the model can actually point at); ungroundable -> dropped, reason kept;
  3. compile the plan with the FK-graph compiler and EXECUTE both it and the gold SQL on the
     SQLite database; keep the plan only if the result sets match (BIRD's EX rule);
  4. write minimal-pair twins (value swap, order-direction swap) whose compiled plans run.
Rows the grammar cannot express become outcome=unsupported examples (the model learns to
abstain). Nothing here reads any client data; inputs are the public dataset folders.

  python scripts/prepare_plans.py --format bird   --root data/bench/dev_20240627 --split dev   --out data/plans
  python scripts/prepare_plans.py --format spider --root data/spider             --split train --out data/plans
  python scripts/prepare_plans.py --format synsql --root data/synsql             --split train --out data/plans --limit 200000
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luthor.sq.candidates import ValueIndex, find_candidates, ground  # noqa: E402
from luthor.sq.convert import convert  # noqa: E402
from luthor.sq.plan import CompileError, Plan, compile_plan  # noqa: E402
from luthor.sq.schema_card import SchemaCard, card_from_spider_tables  # noqa: E402

_LOG = logging.getLogger("luthorsql")

# ---------------------------------------------------------------- dataset layouts
# Paths are relative to --root. `qfield` / `sql` / `context` name the question-file fields.
LAYOUTS = {
    "bird": {"questions": "{split}.json", "tables": "{split}_tables.json", "db": "{split}_databases/{db}/{db}.sqlite",
             "qfield": "question", "sql": "SQL", "context": "evidence", "desc": "{split}_databases/{db}/database_description"},
    "spider": {"questions": {"train": ["train_spider.json", "train_others.json"], "dev": ["dev.json"], "test": ["test.json"]},
               "tables": {"test": "test_tables.json", "*": "tables.json"},
               "db": {"test": "test_database/{db}/{db}.sqlite", "*": "database/{db}/{db}.sqlite"},
               "qfield": "question", "sql": "query", "context": None, "desc": None},
    # Spider questions rewritten with schema-word synonyms; root = the folder holding spider_data/ and spider_syn/
    "spider_syn": {"questions": {"train": ["spider_syn/train_spider.json"], "dev": ["spider_syn/dev.json"]},
                   "tables": "spider_data/tables.json", "db": "spider_data/database/{db}/{db}.sqlite",
                   "qfield": "SpiderSynQuestion", "sql": "query", "context": None, "desc": None},
    "synsql": {"questions": "data.json", "tables": "tables.json", "db": "databases/{db}/{db}.sqlite",
               "qfield": "question", "sql": "sql", "context": "external_knowledge", "desc": None},
    # sealed evaluation sets (never trained or tuned on); root = data/raw
    "spider_dk": {"questions": ["spider_dk/repo/Spider-DK.json"], "tables": "spider_dk/repo/tables.json",
                  "db": "spider_dk/repo/database/{db}/{db}.sqlite", "qfield": "question", "sql": "query", "context": None, "desc": None},
    "spider_realistic": {"questions": ["spider_realistic/spider-realistic.json"], "tables": "spider_data/tables.json",
                         "db": "spider_data/database/{db}/{db}.sqlite", "qfield": "question", "sql": "query", "context": None, "desc": None},
    "kaggledbqa": {"questions": sorted(f"kaggledbqa/repo/examples/{d}_test.json" for d in (
                       "GeoNuclearData", "GreaterManchesterCrime", "Pesticide", "StudentMathScore", "TheHistoryofBaseball",
                       "USWildFires", "WhatCDHipHop", "WorldSoccerDataBase")),
                   "tables": "kaggledbqa/repo/KaggleDBQA_tables.json", "db": "kaggledbqa/databases/databases/{db}/{db}.sqlite",
                   "qfield": "question", "sql": "query", "context": None, "desc": None},
    "wikisql": {"special": "wikisql", "db": "{split}.db", "context": None, "desc": None},
}


def _pick(v, split):
    if isinstance(v, dict):
        return v.get(split, v.get("*"))
    return v


def load_questions(fmt: str, root: Path, split: str) -> list[dict]:
    """Uniform rows: {db_id, question, context, sql, question_id}."""
    lay = LAYOUTS[fmt]
    if lay.get("special") == "wikisql":
        return _wikisql(root, split)[0]
    q = _pick(lay["questions"], split)
    files = q if isinstance(q, list) else [q.format(split=split)]
    rows = []
    for f in files:
        for i, r in enumerate(json.loads((root / f).read_text())):
            rows.append({"db_id": r["db_id"], "question": r[lay["qfield"]], "sql": r[lay["sql"]],
                         "context": (r.get(lay["context"]) or "") if lay["context"] else "",
                         "question_id": r.get("question_id", f"{Path(f).stem}_{i}")})
    return rows


def load_tables(fmt: str, root: Path, split: str) -> dict[str, dict]:
    lay = LAYOUTS[fmt]
    if lay.get("special") == "wikisql":
        return _wikisql(root, split)[1]
    t = _pick(lay["tables"], split).format(split=split)
    return {e["db_id"]: e for e in json.loads((root / t).read_text())}


_WIKI_CACHE: dict = {}


def _wikisql(root: Path, split: str):
    """WikiSQL: one table per 'database'; columns are col0..colN in SQLite and the real
    header names become the column descriptions (the model must read them)."""
    if split in _WIKI_CACHE:
        return _WIKI_CACHE[split]
    AGG = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    OPS = ["=", ">", "<"]
    tables, rows = {}, []
    for line in open(root / f"{split}.tables.jsonl"):
        t = json.loads(line)
        name = "table_" + t["id"].replace("-", "_")
        tables[t["id"]] = {"db_id": t["id"], "table_names_original": [name],
                           "column_names_original": [[-1, "*"]] + [[0, f"col{i}"] for i in range(len(t["header"]))],
                           "column_types": ["text"] + t["types"], "primary_keys": [], "foreign_keys": [],
                           "_descriptions": {name.lower(): {f"col{i}": h for i, h in enumerate(t["header"])}},
                           "_types": t["types"]}
    for i, line in enumerate(open(root / f"{split}.jsonl")):
        r = json.loads(line)
        t = tables[r["table_id"]]
        name, q = t["table_names_original"][0], r["sql"]
        sel = f"col{q['sel']}"
        sel = f"{AGG[q['agg']]}({sel})" if q["agg"] else sel
        conds = []
        for c, op, v in q["conds"]:
            if op >= len(OPS):
                conds = None
                break
            lit = v if t["_types"][c] == "real" and isinstance(v, (int, float)) else "'" + str(v).replace("'", "''") + "'"
            conds.append(f"col{c} {OPS[op]} {lit}")
        if conds is None:
            continue
        sql = f"SELECT {sel} FROM {name}" + (" WHERE " + " AND ".join(conds) if conds else "")
        rows.append({"db_id": r["table_id"], "question": r["question"], "sql": sql, "context": "", "question_id": i})
    _WIKI_CACHE[split] = (rows, tables)
    return rows, tables


def read_descriptions(folder: Path) -> dict[str, dict[str, str]]:
    """BIRD database_description/*.csv -> {table: {column: text}}."""
    out: dict[str, dict[str, str]] = {}
    if not folder or not folder.exists():
        return out
    for f in folder.glob("*.csv"):
        raw = f.read_bytes()
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                rows = list(csv.DictReader(io.StringIO(raw.decode(enc))))
                break
            except UnicodeDecodeError:
                rows = []
        cols = {}
        for r in rows:
            r = {(k or "").strip().lower(): " ".join((v or "").split()) for k, v in r.items()}
            name = r.get("original_column_name", "")
            full, desc, vals = r.get("column_name", ""), r.get("column_description", ""), r.get("value_description", "")
            text = desc if desc and desc.lower() != name.lower() else ""
            if full and full.lower() != name.lower() and full.lower() not in text.lower():
                text = f"{full}. {text}".strip(". ") if text else full
            if vals and vals.lower() not in ("not useful",):
                text = f"{text} Values: {vals[:240]}".strip()
            if name:
                cols[name.lower()] = text[:400]
        out[f.stem.lower()] = cols
    return out


UNSUP_SAMPLE = 1.0  # share of unsupported rows written (all are counted); --unsup-sample
MISMATCH_LOG = None  # --mismatch-log: write every plan that compiled but returned different rows

# ---------------------------------------------------------------- execution check
def run(con: sqlite3.Connection, sql: str, timeout: float = 20.0):
    deadline = time.perf_counter() + timeout
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 20_000)
    try:
        return con.execute(sql).fetchmany(200_001)
    finally:
        con.set_progress_handler(None, 0)


def norm_rows(rows) -> set:
    return {tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows}


# ---------------------------------------------------------------- per-database worker
_SWAP = [("highest", "lowest"), ("most", "least"), ("largest", "smallest"), ("maximum", "minimum"),
         ("biggest", "smallest"), ("greatest", "least"), ("top", "bottom")]


_VINDEX = [None]  # this worker's current database value index (twins re-match their new question)


def process_db(args):
    fmt, root, split, db, entry, items = args
    lay = LAYOUTS[fmt]
    stats = collections.Counter()
    out: list[dict] = []
    db_path = root / _pick(lay["db"], split).format(split=split, db=db)
    if not db_path.exists():
        stats["missing_db"] += len(items)
        return out, stats, None
    desc = entry.get("_descriptions") or (read_descriptions(root / lay["desc"].format(split=split, db=db)) if lay["desc"] else {})
    try:
        card = card_from_spider_tables(entry, db_path, desc)
    except Exception as e:  # one unreadable database file must not stop a 16k-database run
        stats[f"db_error:{type(e).__name__}"] += len(items)
        return out, stats, None
    graph = card.graph()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    rnd = random.Random(hash(db) & 0xFFFF)
    try:
        vindex = ValueIndex.build(con, card)
    except Exception:
        _LOG.warning("%s: value index failed", db, exc_info=True)
        vindex = None
    _VINDEX[0] = vindex
    for qid, q in items:
        try:
            _one(fmt, split, db, qid, q, card, graph, con, rnd, stats, out, vindex)
        except Exception as e:  # a bad row must not stop a 400k-question run
            stats["error:" + type(e).__name__] += 1
    con.close()
    return out, stats, card


def _one(fmt, split, db, qid, q, card, graph, con, rnd, stats, out, vindex=None):
    if True:
        question = q["question"].strip()
        context = (q.get("context") or "").strip()
        gold_sql = q["sql"]
        text = question + (f"\nContext: {context}" if context else "")
        base = {"id": f"{fmt}_{split}_{qid}", "source": fmt, "split": split, "db_id": db,
                "question": question, "context": context, "gold_sql": gold_sql}
        conv = convert(gold_sql, graph)
        if conv.plan is None:
            stats["unsupported"] += 1
            stats["reason:" + re.sub(r":.*", "", conv.reason)] += 1
            if rnd.random() < UNSUP_SAMPLE:
                out.append({**base, "outcome": "unsupported", "reason": conv.reason})
            return
        plan: Plan = conv.plan
        db_values = vindex.match(text) if vindex is not None else []
        cands = find_candidates(text, card, extra=db_values)
        # ground literals: every literal the plan needs (inner plans, HAVING and set-operation
        # sides included) must be a candidate the model could point at
        value_labels: list = []
        ok = _ground_plan(plan, cands, value_labels, top=True)
        limit_label = None
        if ok and plan.limit is not None and plan.limit != 1:
            limit_label = ground(plan.limit, "", cands)
            ok = limit_label is not None
        if not ok:
            stats["ungrounded_value"] += 1
            if os.environ.get("LUTHOR_UNGROUNDED_LOG"):  # which literal failed, for value-grounding work
                miss = []
                def walk(pl):
                    for f in (pl.filters + [pr.cond for pr in pl.projections if pr.cond]
                          + [pr.cond2 for pr in pl.projections if getattr(pr, "cond2", None)] + list(pl.having)):
                        for v in getattr(f, "values", []) or []:
                            if ground(v, f.col.id, cands) is None:
                                miss.append({"col": f.col.id, "value": v})
                        if getattr(f, "sub", None) is not None:
                            walk(f.sub)
                    if pl.other is not None:
                        walk(pl.other)
                walk(plan)
                if plan.limit not in (None, 1) and ground(plan.limit, "", cands) is None:
                    miss.append({"col": "LIMIT", "value": plan.limit})
                with open(os.environ["LUTHOR_UNGROUNDED_LOG"], "a") as fh:
                    fh.write(json.dumps({"db": db, "question": question, "missing": miss}, default=str) + "\n")
            return
        try:
            sql = compile_plan(plan, graph)
        except CompileError as e:
            stats["compile:" + str(e).split(" ")[0]] += 1
            return
        try:
            gold_rows = run(con, gold_sql)
            got = run(con, sql)
        except sqlite3.Error as e:
            stats["exec_error"] += 1
            return
        if norm_rows(got) != norm_rows(gold_rows):
            stats["exec_mismatch"] += 1
            if MISMATCH_LOG is not None:
                with open(MISMATCH_LOG, "a") as fh:
                    fh.write(json.dumps({"db_id": db, "question": question, "gold_sql": gold_sql, "compiled_sql": sql,
                                         "gold_n": len(gold_rows), "got_n": len(got),
                                         "gold_head": [list(map(str, r)) for r in gold_rows[:3]],
                                         "got_head": [list(map(str, r)) for r in got[:3]]}) + "\n")
            return
        empty = len(gold_rows) == 0 or all(all(v in (None, 0) for v in r) for r in gold_rows)
        stats["plan_empty" if empty else "plan_ok"] += 1
        row = {**base, "outcome": "plan", "plan": plan.to_dict(), "compiled_sql": sql,
               "value_labels": value_labels,
               "limit_label": limit_label, "empty_result": empty, "db_values": db_values}
        out.append(row)
        out += twins(row, plan, cands, card, graph, con, rnd, stats)


def _ground_plan(plan: Plan, cands, value_labels: list, top: bool) -> bool:
    """Ground every literal of a plan (recursively) on a candidate; rewrite it to the candidate's
    value. value_labels collects the top-level filters' candidate indices (what v1 trains on)."""
    for f in plan.filters:
        if f.sub is not None:
            if not _ground_plan(f.sub, cands, [], top=False):
                return False
            if top:
                value_labels.append([])
            continue
        idx = []
        for v in f.values:
            i = ground(v, f.col.id, cands)
            if i is None:
                return False
            idx.append(i)
        f.values = [cands[i].value for i in idx]
        if top:
            value_labels.append(idx)
    for pr in plan.projections:
        for cf in (pr.cond, getattr(pr, "cond2", None)):  # grammar v4.1: a second condition
            if cf is None:
                continue
            idx = [ground(v, cf.col.id, cands) for v in cf.values]
            if any(i is None for i in idx):
                return False
            cf.values = [cands[i].value for i in idx]
    for h in plan.having:
        idx = [ground(v, "", cands) for v in h.values]
        if any(i is None for i in idx):
            return False
        h.values = [cands[i].value for i in idx]
    for c in plan.columns():  # grammar v3: string values inside expression templates
        if any(ground(v, c.id, cands) is None for v in c.vals):
            return False
    if plan.other is not None and not _ground_plan(plan.other, cands, [], top=False):
        return False
    return True


def twins(row, plan: Plan, cands, card: SchemaCard, graph, con, rnd, stats) -> list[dict]:
    """Minimal pairs: one word changes, one slot changes, the compiled plan still runs."""
    out = []
    # value swap: a grounded stored value that appears in the question text
    for fi, f in enumerate(plan.filters):
        for vi, ci in enumerate(row["value_labels"][fi]):
            c = cands[ci]
            if c.kind != "value" or c.span is None or c.span[1] > len(row["question"]):
                continue
            pool = [v for v in card.values.get(f.col.id, []) if v != c.value and 2 < len(v) < 40]
            if not pool:
                continue
            nv = rnd.choice(pool)
            q2 = row["question"][:c.span[0]] + nv + row["question"][c.span[1]:]
            p2 = Plan.from_dict(plan.to_dict())
            p2.filters[fi].values[vi] = nv
            try:
                sql2 = compile_plan(p2, graph)
                if not run(con, sql2, timeout=5.0):
                    continue
            except Exception:
                _LOG.debug("value twin skipped", exc_info=True)
                continue
            out.append({**{k: row[k] for k in ("source", "split", "db_id", "context")},
                        "id": row["id"] + f"_twin_v{fi}{vi}", "question": q2, "outcome": "plan",
                        "plan": p2.to_dict(), "compiled_sql": sql2, "twin_of": row["id"], "twin_slot": "value",
                        "gold_sql": None, "empty_result": False,
                        "db_values": _VINDEX[0].match(q2) if _VINDEX[0] is not None else []})
            stats["twin_value"] += 1
            return out  # one value twin per row is enough
    # order-direction swap
    if plan.order_dir in ("asc", "desc"):
        low = row["question"].lower()
        for a, b in _SWAP:
            for x, y in ((a, b), (b, a)):
                m = re.search(rf"\b{x}\b", low)
                if m:
                    q2 = row["question"][:m.start()] + y + row["question"][m.end():]
                    p2 = Plan.from_dict(plan.to_dict())
                    p2.order_dir = "asc" if plan.order_dir == "desc" else "desc"
                    try:
                        sql2 = compile_plan(p2, graph)
                    except Exception:
                        _LOG.debug("order twin skipped", exc_info=True)
                        return out
                    out.append({**{k: row[k] for k in ("source", "split", "db_id", "context")},
                                "id": row["id"] + "_twin_o", "question": q2, "outcome": "plan",
                                "plan": p2.to_dict(), "compiled_sql": sql2, "twin_of": row["id"],
                                "twin_slot": "order_dir", "gold_sql": None, "empty_result": False,
                                "value_labels": row["value_labels"],
                                "limit_label": row["limit_label"], "db_values": row.get("db_values", [])})
                    stats["twin_order"] += 1
                    return out
    return out


# ---------------------------------------------------------------- leak control
def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def dedup_against(rows: list[dict], eval_questions: list[str], threshold: float = 0.5,
                  max_df: int = 300) -> list[dict]:
    """Drop training rows whose word overlap with any eval question is >= threshold (Jaccard).
    Candidates come only from words that occur in <= max_df eval questions (two questions
    sharing half their words almost always share one of those), which keeps it fast."""
    inv: dict[str, list[int]] = collections.defaultdict(list)
    ev = [_words(q) for q in eval_questions]
    for i, w in enumerate(ev):
        for t in w:
            inv[t].append(i)
    rare = {t: ids for t, ids in inv.items() if len(ids) <= max_df}
    keep = []
    for r in rows:
        w = _words(r["question"])
        cand = set()
        for t in w:
            if t in rare:
                cand.update(rare[t])
        if any(len(w & ev[i]) / max(1, len(w | ev[i])) >= threshold for i in cand):
            continue
        keep.append(r)
    return keep


def _set_unsup(p: float, mlog=None, allowed=None, nums=None) -> None:
    global UNSUP_SAMPLE, MISMATCH_LOG
    UNSUP_SAMPLE = p
    MISMATCH_LOG = mlog
    import luthor.sq.convert as cv
    cv.ALLOWED_TEMPLATES = allowed
    cv.ALLOWED_NUMS = nums


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=LAYOUTS, required=True)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exclude-eval", nargs="*", default=[], help="plans.jsonl of eval sets to dedup against")
    ap.add_argument("--exclude-dbs", nargs="*", default=[], help="db_ids that must never appear (eval databases)")
    ap.add_argument("--unsup-sample", type=float, default=1.0)
    ap.add_argument("--mismatch-log", default=None)
    ap.add_argument("--allowed-templates", type=Path, default=None,
                    help="grammar v3: JSON list of expression templates learned on training data (default: any)")
    a = ap.parse_args()
    allowed = set(json.loads(a.allowed_templates.read_text())) if a.allowed_templates else None
    nf = a.allowed_templates.with_name(a.allowed_templates.stem + "_nums.json") if a.allowed_templates else None
    nums = set(json.loads(nf.read_text())) if nf and nf.exists() else None
    global UNSUP_SAMPLE
    UNSUP_SAMPLE = a.unsup_sample

    qs = load_questions(a.format, a.root, a.split)
    if a.limit:
        random.Random(0).shuffle(qs)
        qs = qs[: a.limit]
    tables = load_tables(a.format, a.root, a.split)
    by_db: dict[str, list] = collections.defaultdict(list)
    skipped_eval_db = 0
    for i, q in enumerate(qs):
        if q["db_id"] in a.exclude_dbs:
            skipped_eval_db += 1
            continue
        by_db[q["db_id"]].append((q["question_id"], q))
    jobs = [(a.format, a.root, a.split, db, tables[db], items) for db, items in by_db.items() if db in tables]

    outdir = a.out / f"{a.format}_{a.split}"
    (outdir / "cards").mkdir(parents=True, exist_ok=True)
    rows, stats = [], collections.Counter({"skipped_eval_db": skipped_eval_db})
    t0 = time.time()
    with Pool(a.workers, initializer=_set_unsup, initargs=(a.unsup_sample, a.mismatch_log, allowed, nums)) as pool:
        for n, (r, s, card) in enumerate(pool.imap_unordered(process_db, jobs), 1):
            rows += r
            stats.update(s)
            if card is not None:
                card.save(outdir / "cards" / f"{card.db_id}.json")
            if n % 20 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} databases, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    if a.exclude_eval:
        evq = []
        for f in a.exclude_eval:
            evq += [json.loads(l)["question"] for l in open(f)]
        before = len(rows)
        rows = dedup_against(rows, evq)
        stats["dedup_removed"] = before - len(rows)
    with open(outdir / "plans.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    total = len(qs) - skipped_eval_db
    ok = stats["plan_ok"] + stats["plan_empty"]
    summary = {"questions": total, "plans_verified": ok, "coverage": round(ok / max(1, total), 4),
               "unsupported": stats["unsupported"], "twins": stats["twin_value"] + stats["twin_order"],
               "stats": dict(stats.most_common())}
    (outdir / "stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
