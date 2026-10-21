"""System 2 as a plain API tool-use loop (no Claude Code harness): one question per conversation, one read-only SQL
tool with the 8-execution limit enforced in code, a submit tool for the final answer. Resumable (answers appended
per question; finished ids are skipped) and logged (per-question tokens, turns, executions, errors).

Providers: `ollama` (Ollama Cloud /api/chat, key in $OLLAMA_API_KEY). The same loop is the base for the BIRD
test submission runner.

  OLLAMA_API_KEY=... python3 scripts/system2/api_runner.py --provider ollama --model glm-5.3-flash --arm A1 \
      --chunks <box>/chunks/A1 --dbs <box>/dbs --out <dir>/answers.jsonl --workers 8
Arms: A0 = model alone (no candidates shown); A1 = model + System 1 handoff (candidates shown, pick or own SQL).
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "sql_tool.py"
MAX_SQL = 8
MAX_TURNS = 14

RULES = """Rules (strict — this is a benchmark):
- At most 8 SQL executions. Use the evidence: it defines terms and formulas exactly as the benchmark intends.
- Return exactly the columns the question asks for, in the order asked, nothing extra (no extra id/name columns
  unless asked). For "which/what X has the highest ..." return X with ORDER BY ... LIMIT 1 unless ties are clearly
  intended. Percentages/ratios: follow the evidence formula; cast to REAL for division.
- Scoring = the result set of your SQL equals the gold result set (row order ignored, values exact)."""

SYS_A0 = """You answer one BIRD text-to-SQL question (SQLite). You get the database schema (tables, 3 sample rows per
table, column descriptions), the question and the evidence. Tool run_sql runs read-only SQL and returns up to 20 rows.
When done, call submit with your one final SQLite query. If unsure, still submit your best SQL.
""" + RULES

SYS_A1 = """You answer one BIRD text-to-SQL question (SQLite) together with a fast text-to-SQL system. That system
already proposed candidate queries, grouped by identical execution result and listed best first by its own
confidence (n_of_32 = how many of its 32 samples gave that result; "result" = a short preview of the rows). There may
be no candidates. Tool run_sql runs read-only SQL and returns up to 20 rows.
When done, call submit with pick="<label>" if one candidate answers the question correctly, or pick="NONE" and your
own sql if none is correct or there are none. When a candidate is clearly right, pick it without extra exploration;
write your own SQL only when you are confident every candidate is wrong.
""" + RULES

TOOLS_SQL = {"type": "function", "function": {"name": "run_sql", "description": "Run one read-only SQLite query on the question's database; returns columns and up to 20 rows.",
             "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}}
SUBMIT_A0 = {"type": "function", "function": {"name": "submit", "description": "Submit the final SQLite query.",
             "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}}
SUBMIT_A1 = {"type": "function", "function": {"name": "submit", "description": "Submit the final answer: a candidate label, or NONE with your own SQL.",
             "parameters": {"type": "object", "properties": {"pick": {"type": "string"}, "sql": {"type": "string"}}, "required": ["pick"]}}}


def tool(dbs: Path, db: str, arg: str) -> str:
    env = {**os.environ, "LUTHOR_SQL_DB_ROOT": str(dbs)}
    env.pop("OLLAMA_API_KEY", None)  # the tool never sees the key
    try:
        p = subprocess.run([sys.executable, str(TOOL), db, arg], capture_output=True, text=True, timeout=60, env=env)
        out = (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        out = "ERROR: query timed out"
    return out[:12000]


def chat_ollama(model, messages, tools, key):
    body = json.dumps({"model": model, "stream": False, "messages": messages, "tools": tools,
                       "options": {"temperature": 0}}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request("https://ollama.com/api/chat", data=body,
                                         headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=300))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
            code = getattr(e, "code", None)
            last = f"{type(e).__name__} {code}: {e}"
            if code in (400, 401, 402, 403):  # bad request / auth / out of credit: retrying cannot help
                raise
            time.sleep(min(60, 2 ** attempt * 3))
    raise RuntimeError(f"API failed after retries (last: {last})")


def solve(q, a, key, schemas, lock):
    db = q["db_id"]
    with lock:
        if db not in schemas:
            schemas[db] = tool(a.dbs, db, "--schema")
    user = f"Database: {db}\n\nSCHEMA\n{schemas[db]}\n\nQUESTION: {q['question']}\nEVIDENCE: {q.get('evidence') or ''}"
    if a.arm == "A1":
        cands = q.get("candidates") or []
        user += "\n\nCANDIDATES\n" + (json.dumps(cands, indent=1) if cands else "(none)")
    msgs = [{"role": "system", "content": SYS_A1 if a.arm == "A1" else SYS_A0}, {"role": "user", "content": user}]
    tools = [TOOLS_SQL, SUBMIT_A1 if a.arm == "A1" else SUBMIT_A0]
    n_sql, tok_in, tok_out, ans = 0, 0, 0, None
    for turn in range(MAX_TURNS):
        r = chat_ollama(a.model, msgs, tools, key)
        tok_in += r.get("prompt_eval_count") or 0
        tok_out += r.get("eval_count") or 0
        m = r.get("message") or {}
        calls = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        if not calls:
            msgs.append({"role": "user", "content": "Call run_sql or submit."})
            continue
        for c in calls:
            f = c.get("function") or {}
            args = f.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if f.get("name") == "submit":
                ans = args
                break
            if n_sql >= MAX_SQL:
                out = "ERROR: execution limit reached (8 per question); call submit now."
            else:
                n_sql += 1
                out = tool(a.dbs, db, str(args.get("sql") or ""))
            msgs.append({"role": "tool", "content": out, "tool_name": "run_sql"})
        if ans is not None:
            break
        if turn == MAX_TURNS - 2:
            msgs.append({"role": "user", "content": "Turn limit: call submit now."})
    row = {"question_id": q["question_id"], "turns": turn + 1, "sql_executions": n_sql, "tok_in": tok_in, "tok_out": tok_out}
    if a.arm == "A1":
        pick = str((ans or {}).get("pick") or "NONE").strip()
        row.update({"pick": pick, "sql": (ans or {}).get("sql") or ""})
    else:
        row["sql"] = (ans or {}).get("sql") or ""
    row["submitted"] = ans is not None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["ollama"], default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=["A0", "A1"], required=True)
    ap.add_argument("--chunks", type=Path, required=True, help="dir of cNN.jsonl (A1 rows carry candidates)")
    ap.add_argument("--dbs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    key = os.environ["OLLAMA_API_KEY"]
    qs = [json.loads(l) for f in sorted(a.chunks.glob("*.jsonl")) for l in open(f)]
    if a.limit:
        qs = qs[:a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["question_id"] for l in open(a.out)} if a.out.exists() else set()
    todo = [q for q in qs if q["question_id"] not in done]
    print(f"{a.model} {a.arm}: {len(qs)} questions, {len(done)} done, {len(todo)} to run", flush=True)
    lock, wlock, schemas = threading.Lock(), threading.Lock(), {}
    log = a.out.with_suffix(".errors.log")

    def work(q):
        try:
            row = solve(q, a, key, schemas, lock)
        except Exception as e:  # logged; the question is retried on the next resume
            with wlock, open(log, "a") as f:
                f.write(f"{q['question_id']}\t{type(e).__name__}: {str(e)[:300]}\n")
            return
        with wlock, open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    n = sum(1 for _ in open(a.out)) if a.out.exists() else 0
    print(f"done: {n}/{len(qs)} answered", flush=True)


if __name__ == "__main__":
    main()
