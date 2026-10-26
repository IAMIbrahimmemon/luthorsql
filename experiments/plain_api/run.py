"""Plain Anthropic Messages API agent on RESERVED (experiments/plain_api/PREREG.md): no Claude Code harness.

One conversation per question, the paper's API-agent prompts (scripts/system2/api_runner.py, paper/prompts/api_loop.txt),
the read-only SQL tool with the 8-execution limit enforced in code, a submit tool, prompt caching on the system prompt
and the schema block. Resumable (answers appended per question; finished ids are skipped). Stops when the run's
list-price cost passes --max-dollars.

  python3 experiments/plain_api/run.py --s1 runs/phase2/cost_confirmation/s1_reserved.jsonl \
      --dbs data/raw/bird_train/train_databases --out runs/phase2/plain_api/r1      # key from ~/.config/luthor/anthropic_key
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "system2"))
from api_runner import MAX_SQL, MAX_TURNS, SYS_A0, SYS_A1  # noqa: E402  (the paper's API-agent prompts and limits)

TOOL = ROOT / "scripts" / "system2" / "sql_tool.py"
MODEL = "claude-opus-5-5"
PRICE = {"input_tokens": 4.0, "cache_creation_input_tokens": 5.0, "cache_read_input_tokens": 0.20, "output_tokens": 20.0}
RUN_SQL = {"name": "run_sql", "description": "Run one read-only SQLite query on the question's database; returns columns and up to 20 rows.",
           "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}
SUBMIT = {"A0": {"name": "submit", "description": "Submit the final SQLite query.",
                 "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
          "A1": {"name": "submit", "description": "Submit the final answer: a candidate label, or NONE with your own SQL.",
                 "input_schema": {"type": "object", "properties": {"pick": {"type": "string"}, "sql": {"type": "string"}}, "required": ["pick"]}}}
CACHE = {"type": "ephemeral"}


def dollars(u: dict) -> float:
    return sum(u.get(k, 0) * p for k, p in PRICE.items()) / 1e6


def tool(dbs: Path, db: str, arg: str) -> str:
    env = {k: v for k, v in os.environ.items() if "KEY" not in k.upper()}  # the tool never sees a key
    env["LUTHOR_SQL_DB_ROOT"] = str(dbs)
    try:
        p = subprocess.run([sys.executable, str(TOOL), db, arg], capture_output=True, text=True, timeout=60, env=env)
        out = (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        out = "ERROR: query timed out"
    return out[:12000]


def call(body: dict, key: str, dry: bool) -> dict:
    if dry:  # pipeline test: submit candidate A (or System 1's SQL) without any API call
        return {"content": [{"type": "tool_use", "id": "t0", "name": "submit", "input": {"pick": "A", "sql": "SELECT 1"}}],
                "usage": {}, "stop_reason": "tool_use"}
    data = json.dumps(body).encode()
    last = ""
    for attempt in range(8):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=data, headers={
                "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=600))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code in (400, 401, 402, 403, 404):  # bad request / auth / credit: retrying cannot help
                raise RuntimeError(last)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(90, 3 * 2 ** attempt))
    raise RuntimeError(f"API failed after retries ({last})")


def solve(q: dict, arm: str, a, key: str, schema: str) -> dict:
    user = [{"type": "text", "text": f"Database: {q['db_id']}\n\nSCHEMA\n{schema}", "cache_control": CACHE},
            {"type": "text", "text": f"QUESTION: {q['question']}\nEVIDENCE: {q.get('evidence') or ''}"
             + ("\n\nCANDIDATES\n" + (json.dumps(q['candidates'], indent=1) if q.get('candidates') else "(none)") if arm == "A1" else "")}]
    msgs = [{"role": "user", "content": user}]
    system = [{"type": "text", "text": SYS_A1 if arm == "A1" else SYS_A0, "cache_control": CACHE}]
    tools = [RUN_SQL, SUBMIT[arm]]
    usage, n_sql, ans, turn, stops = defaultdict(int), 0, None, 0, []
    t0 = time.time()
    for turn in range(MAX_TURNS):
        # rolling breakpoint on the newest message (the schema and system breakpoints stay fixed)
        for m in msgs[1:]:
            if isinstance(m["content"], list):
                for b in m["content"]:
                    b.pop("cache_control", None)
        if len(msgs) > 1 and isinstance(msgs[-1]["content"], list) and msgs[-1]["content"]:
            msgs[-1]["content"][-1]["cache_control"] = CACHE
        r = call({"model": MODEL, "max_tokens": 4096, "system": system, "tools": tools, "messages": msgs}, key, a.dry_run)
        for k in PRICE:
            usage[k] += (r.get("usage") or {}).get(k) or 0
        stops.append(r.get("stop_reason"))
        content = r.get("content") or []
        msgs.append({"role": "assistant", "content": content})
        uses = [b for b in content if b.get("type") == "tool_use"]
        if not uses:
            msgs.append({"role": "user", "content": [{"type": "text", "text": "Call run_sql or submit."}]})
            continue
        results = []
        for b in uses:
            if b["name"] == "submit":
                ans = b.get("input") or {}
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": "submitted"})
                continue
            if n_sql >= MAX_SQL:
                out = "ERROR: execution limit reached (8 per question); call submit now."
            else:
                n_sql += 1
                out = tool(a.dbs, q["db_id"], str((b.get("input") or {}).get("sql") or ""))
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out})
        if ans is not None:
            break
        if turn == MAX_TURNS - 2:
            results.append({"type": "text", "text": "Turn limit: call submit now."})
        msgs.append({"role": "user", "content": results})
    row = {"question_id": q["id"], "db_id": q["db_id"], "arm": arm, "turns": turn + 1, "sql_executions": n_sql,
           "usage": dict(usage), "dollars": round(dollars(usage), 6), "seconds": round(time.time() - t0, 2),
           "stop_reasons": stops, "submitted": ans is not None, "sql": (ans or {}).get("sql") or ""}
    if arm == "A1":
        row["pick"] = str((ans or {}).get("pick") or "NONE").strip()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=Path, required=True)
    ap.add_argument("--dbs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="run directory: answers_A0.jsonl, answers_A1.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-dollars", type=float, default=40.0)
    ap.add_argument("--limit-per-db", type=int, help="smoke test: first N questions of each database")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", nargs="+", choices=["A0", "A1"], default=["A0", "A1"])
    a = ap.parse_args()
    key = "" if a.dry_run else (os.environ.get("ANTHROPIC_API_KEY") or open(os.path.expanduser("~/.config/luthor/anthropic_key")).read().strip())
    qs = [json.loads(l) for l in open(a.s1)]
    by_db = defaultdict(list)
    for q in sorted(qs, key=lambda q: (q["db_id"], q["id"])):
        by_db[q["db_id"]].append(q)
    if a.limit_per_db:
        by_db = {d: v[:a.limit_per_db] for d, v in by_db.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    files = {arm: a.out / f"answers_{arm}.jsonl" for arm in a.arms}
    done = {arm: set() for arm in files}
    spent = 0.0
    for arm, f in files.items():
        if f.exists():
            for l in open(f):
                r = json.loads(l)
                done[arm].add(r["question_id"])
                spent += r["dollars"]
    print(f"resume: {', '.join(f'{k} {len(v)}' for k, v in done.items())} done, ${spent:.2f} spent", flush=True)
    lock = threading.Lock()
    state = {"spent": spent, "stop": False}
    errlog = a.out / "errors.log"

    def work(q, arm, schema):
        if state["stop"]:
            return
        try:
            row = solve(q, arm, a, key, schema)
        except Exception as e:  # logged; retried on the next resume
            with lock, open(errlog, "a") as f:
                f.write(f"{arm}\t{q['id']}\t{type(e).__name__}: {str(e)[:300]}\n")
            return
        with lock:
            with open(files[arm], "a") as f:
                f.write(json.dumps(row) + "\n")
            state["spent"] += row["dollars"]
            if state["spent"] > a.max_dollars:
                state["stop"] = True

    for d, dq in by_db.items():
        schema = tool(a.dbs, d, "--schema")
        for arm in a.arms:
            todo = [q for q in dq if q["id"] not in done[arm]]
            if not todo or state["stop"]:
                continue
            work(todo[0], arm, schema)  # first question alone writes the cache
            with ThreadPoolExecutor(a.workers) as ex:
                list(ex.map(lambda q: work(q, arm, schema), todo[1:]))
            print(f"{d} {arm}: {len(todo)} run, ${state['spent']:.2f} total", flush=True)
    n = {arm: sum(1 for _ in open(f)) if f.exists() else 0 for arm, f in files.items()}
    print(f"{'STOPPED at $' + format(state['spent'], '.2f') if state['stop'] else 'done'}: {n} answered, ${state['spent']:.2f}", flush=True)


if __name__ == "__main__":
    main()
