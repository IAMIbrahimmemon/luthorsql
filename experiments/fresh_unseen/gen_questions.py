"""Generate the fresh unseen question set (experiments/fresh_unseen/PREREG.md).

Per database: deepseek-v4.1-flash writes 40 BIRD-style items (question, evidence, gold SQLite, difficulty) in 4
batches; automatic filters (executes < 30 s, non-empty, <= 1000 rows, deterministic, not a near-duplicate); an
independent glm-5.3 judge removes items whose gold does not answer the question exactly; the first 30 survivors in
generation order are kept. Then the owner's 50-item check sample is drawn (seed 20261021).

  .venv/bin/python experiments/fresh_unseen/gen_questions.py --dbs data/sys1/fresh/dbs --out data/sys1/fresh
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = "deepseek-v4.1-flash:cloud"
JUDGE = "glm-5.3:cloud"
PER_DB, KEEP, SEED = 40, 30, 20261021
MIX = [("simple", 3), ("moderate", 4), ("challenging", 3)]  # per batch of 10 -> 30/40/30, close to the 30/45/25 aim


def llm(model, prompt, temperature=0.7, tries=3):
    body = json.dumps({"model": model, "stream": False, "format": "json", "options": {"temperature": temperature},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                txt = json.loads(r.read())["message"]["content"].strip()
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)  # glm wraps JSON in a fence even in JSON mode
            return json.loads(txt)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"{model} failed: {last}")


def schema_text(dbs: Path, db: str) -> str:
    env = {**os.environ, "LUTHOR_SQL_DB_ROOT": str(dbs)}
    tool = ROOT / "experiments/fresh_unseen/sql_tool.py"
    return subprocess.run(["python3", str(tool), db, "--schema"], capture_output=True, text=True, env=env).stdout


def run(path: Path, sql: str):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    t = threading.Timer(30, con.interrupt)
    t.start()
    try:
        return con.execute(sql).fetchall()
    except Exception:  # noqa: BLE001
        return None
    finally:
        t.cancel()
        con.close()


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def lcs_ratio(a, b):
    a, b = words(a), words(b)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return 2 * prev[-1] / (len(a) + len(b))


def preview(rows, k=5):
    return f"{len(rows)} rows; first {min(k, len(rows))}: " + "; ".join(repr(tuple(r))[:160] for r in rows[:k])


AUTHOR_PROMPT = """You are writing a new text-to-SQL benchmark in the style of BIRD for the SQLite database below.

Write exactly {n} items: {mix}. Each item has:
- "question": a natural, specific question a data analyst would ask, answerable from this database. Use real values
  that appear in the data (see the sample rows; you may use other values only if you are sure they exist).
- "evidence": one or two short notes in BIRD style that define domain terms, formulas or value mappings the question
  needs (e.g. "severe refers to classification = 'Class I'; rate = DIVIDE(COUNT(...), COUNT(...))"). May be "" for
  simple items.
- "sql": the single correct SQLite query (gold). Select exactly the columns the question asks for. Make the answer
  deterministic: avoid LIMIT with ties unless the question says how to break them.
- "difficulty": "simple" (one table, simple filter/aggregate), "moderate" (joins and/or grouping), or
  "challenging" (several joins, subqueries, ratios/percentages, window functions or date arithmetic).
Do not repeat or paraphrase any of these earlier questions: {prev}

Return JSON {{"items": [{{"question": ..., "evidence": ..., "sql": ..., "difficulty": ...}}, ...]}}.

DATABASE `{db}`:
{schema}"""

JUDGE_PROMPT = """You check items of a text-to-SQL benchmark. Decide whether the gold SQL answers the question
EXACTLY as asked, given the evidence notes (which define terms as the benchmark intends): the right rows, the right
columns (no extra or missing columns), correct joins, filters, aggregation and value spellings. Be strict; a gold that
is only approximately right, returns extra columns, or depends on an arbitrary tie-break is NO.

Database schema:
{schema}

Question: {question}
Evidence: {evidence}
Gold SQL: {sql}
Gold result: {preview}

Return JSON {{"verdict": "YES" or "NO", "reason": "<one sentence>"}}."""


def build_db(dbs: Path, db: str, log):
    schema = schema_text(dbs, db)[:24000]
    path = dbs / db / f"{db}.sqlite"
    cands, kept = [], []
    for b in range(PER_DB // 10):
        prev = json.dumps([c["question"] for c in cands])[:6000]
        mix = ", ".join(f"{k} {d}" for d, k in MIX)
        try:
            got = llm(AUTHOR, AUTHOR_PROMPT.format(n=10, mix=mix, prev=prev, db=db, schema=schema)).get("items", [])
        except Exception as e:  # noqa: BLE001
            log.append(f"{db} batch {b}: author failed {e}")
            continue
        for it in got[:10]:
            if isinstance(it, dict) and it.get("question") and it.get("sql"):
                cands.append({"question": str(it["question"]).strip(), "evidence": str(it.get("evidence") or "").strip(),
                              "sql": str(it["sql"]).strip(), "difficulty": str(it.get("difficulty", "")).strip().lower(),
                              "gen_order": len(cands)})
    stats = {"generated": len(cands), "exec_fail": 0, "empty": 0, "too_many_rows": 0, "nondeterministic": 0,
             "duplicate": 0, "judge_no": 0, "judge_error": 0}
    for c in cands:
        r1 = run(path, c["sql"])
        if r1 is None:
            stats["exec_fail"] += 1
            continue
        if not r1 or all(all(v is None for v in row) for row in r1):
            stats["empty"] += 1
            continue
        if len(r1) > 1000:
            stats["too_many_rows"] += 1
            continue
        r2 = run(path, c["sql"])
        if r2 is None or frozenset(r2) != frozenset(r1):
            stats["nondeterministic"] += 1
            continue
        if any(lcs_ratio(c["question"], k["question"]) > 0.8 for k in kept):
            stats["duplicate"] += 1
            continue
        try:
            v = llm(JUDGE, JUDGE_PROMPT.format(schema=schema, question=c["question"], evidence=c["evidence"],
                                               sql=c["sql"], preview=preview(r1)), temperature=0.0)
        except Exception:  # noqa: BLE001
            stats["judge_error"] += 1
            continue
        if str(v.get("verdict", "")).upper() != "YES":
            stats["judge_no"] += 1
            continue
        c["judge_reason"] = str(v.get("reason", ""))
        c["gold_rows"] = len(r1)
        c["gold_preview"] = preview(r1)
        kept.append(c)
        if len(kept) == KEEP:
            break
    stats["kept"] = len(kept)
    return db, kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dbs = sorted(p.name for p in a.dbs.iterdir() if (p / f"{p.name}.sqlite").exists())
    log = []
    with ThreadPoolExecutor(len(dbs)) as ex:
        res = list(ex.map(lambda d: build_db(a.dbs, d, log), dbs))
    items, stats = [], {}
    for db, kept, st in res:
        stats[db] = st
        for k, c in enumerate(kept):
            items.append({"id": f"fresh_{db}_{k}", "db_id": db, **c})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    rng = random.Random(SEED)
    sample = sorted(rng.sample(range(len(items)), min(50, len(items))))
    (a.out / "human_check_ids.json").write_text(json.dumps([items[i]["id"] for i in sample]))
    (a.out / "gen_stats.json").write_text(json.dumps({"per_db": stats, "total": len(items), "log": log}, indent=1))
    print(json.dumps({"per_db": stats, "total": len(items), "log": log}, indent=1))


if __name__ == "__main__":
    main()
