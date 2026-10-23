"""LuthorSQL cascade on BIRD, stage 2 (API): send System 1's unsure questions to Claude Opus 5.5.

Stage 1 (GPU, run_bird.sh) writes <pred>.json and <pred>.json.cands.jsonl. This stage reads only the .cands.jsonl file
(question, evidence, System 1's candidates with their validator scores and result previews; no gold) and the
databases, and writes the final BIRD prediction file.

  prepare   margin per question (top-2 distinct-result validator gap); questions with margin < 3.0 or no margin are
            routed; routed questions are grouped by database into chunks of at most 25, each with System 1's top 8
            result groups, inside a sandbox holding only the databases, the SQL tool and the chunks.
  run       one Claude Agent SDK session (the Claude Code agent loop) per chunk, with the prompt used in the paper's
            held-out tests, verbatim. Resumable: finished chunks are skipped, unfinished questions are retried.
  merge     kept questions keep System 1's pick; routed questions take the agent's pick (a candidate label) or its own
            SQL; a routed question with no agent answer falls back to System 1's pick (counted in the report).
  tokens    prompt and output tokens and list-price cost from the session logs (for BIRD's advance token count).

The agent runs with no user or project settings, no CLAUDE.md, no MCP servers, no skills or plugins, and only the
Bash, Read, Write, Edit, Glob and Grep tools; a PreToolUse guard refuses any network command, package install or path
outside the sandbox, and Bash runs in the OS sandbox with no network where the platform supports it.

  python cascade.py all --cands predict_test.json.cands.jsonl --db-root test_databases --work cascade_work \
      --out predict_test_cascade.json [--jobs 4] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import re
import shutil
import string
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = 3.0
MODEL = "claude-opus-5-5"
TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
PROMPT = (HERE / "PROMPT_routed.md").read_text()
RUN_BLOCK = ("\nYour run:\n"
             "  Working directory: {box}  (run every command as: cd {box} && ...)\n"
             "  CHUNK = {chunk}\n"
             "  OUT   = {out}\n"
             "  Tool  = python3 TOOLS/sql.py (inside the working directory)\n"
             "Your final reply must be only the final message line described above.")
PRICE = {"input_tokens": 4.0, "cache_creation_input_tokens": 5.0, "cache_read_input_tokens": 0.20}  # $/M, Opus 5.5
PRICE_OUT = 20.0
FILLER = "SELECT 1"


# ---------------------------------------------------------------------------------------------------- System 1 side
def margin(cands):
    """Top-2 distinct-result validator margin over live candidates (paper, Section 4); None when there is no margin."""
    live = [c for c in cands if c.get("sql") and c.get("runs") and c.get("nonnull")]
    if not live:
        return None
    best = {}
    for c in live:
        s = c.get("sig") or c["sql"]
        if s not in best or c["rr"] > best[s]["rr"]:
            best[s] = c
    cls = sorted(best.values(), key=lambda c: -c["rr"])
    return None if len(cls) == 1 else cls[0]["rr"] - cls[1]["rr"]


def groups(cands, top=8):
    lv = [x for x in cands if x.get("sql") and x.get("runs") and x.get("nonnull")] or [x for x in cands if x.get("sql")]
    g = {}
    for x in sorted(lv, key=lambda x: -x.get("rr", -1e9)):
        g.setdefault(x.get("sig") or x["sql"], []).append(x)
    return [{"label": string.ascii_uppercase[k], "sql": v[0]["sql"], "result": v[0].get("res", ""), "n_of_32": len(v)}
            for k, v in enumerate(list(g.values())[:top])]


def s1_pick(cands):
    u = [c for c in cands if c.get("sql")]
    if not u:
        return None
    live = [c for c in u if c.get("runs") and c.get("nonnull")] or u
    return max(live, key=lambda c: c.get("rr", -1e9))["sql"]


def chunks(rows, size=25):
    by = defaultdict(list)
    for r in rows:
        by[r["db_id"]].append(r)
    for db, rs in sorted(by.items()):
        k = -(-len(rs) // size)
        n = -(-len(rs) // k)
        for j in range(k):
            yield f"{db}_{j + 1}", rs[j * n:(j + 1) * n]


def load_s1(path):
    rows = []
    for ln in open(path):
        r = json.loads(ln)
        rows.append({"index": int(r["index"]), "question_id": f"q{int(r['index'])}", "db_id": r["db_id"],
                     "question": r.get("question", ""), "evidence": (r.get("context") or "").strip(),
                     "cands": r.get("cands") or []})
    rows.sort(key=lambda r: r["index"])
    return rows


def prepare(a):
    work = a.work.resolve()
    box = work / "box"
    rows = load_s1(a.cands)
    routed, kept = [], []
    for r in rows:
        m = margin(r["cands"])
        r["margin"] = None if m is None else round(m, 4)
        (routed if m is None or m < T else kept).append(r)
    if box.exists():
        old = json.loads((work / "routing.json").read_text())
        assert old["routed_ids"] == [r["question_id"] for r in routed], "routing changed since the last prepare"
        print(f"prepare: {box} exists and matches; nothing to do")
        return
    for sub in ("dbs", "TOOLS", "chunks/C", "answers/C", "work"):
        (box / sub).mkdir(parents=True, exist_ok=True)
    db_root = a.db_root.resolve()
    for db in sorted({r["db_id"] for r in rows}):
        src = db_root / db
        assert (src / f"{db}.sqlite").exists(), f"missing database {src}/{db}.sqlite"
        (box / "dbs" / db).symlink_to(src, target_is_directory=True)
    shutil.copy2(HERE / "sql_tool.py", box / "TOOLS/sql.py")
    names = []
    for name, ch in chunks(routed):
        (box / "chunks/C" / f"{name}.jsonl").write_text("".join(json.dumps(
            {"question_id": r["question_id"], "db_id": r["db_id"], "question": r["question"], "evidence": r["evidence"],
             "candidates": groups(r["cands"])}) + "\n" for r in ch))
        names.append(name)
    text = "".join(p.read_text() for p in (box / "chunks").rglob("*.jsonl"))
    leak = [k for k in ('"SQL"', '"gold"', '"gold_sql"', '"ok"', '"difficulty"') if k in text]
    assert not leak, f"label field in chunks: {leak}"
    summary = {"n": len(rows), "routed": len(routed), "kept": len(kept), "agent_share": round(len(routed) / len(rows), 4),
               "no_margin": sum(r["margin"] is None for r in rows), "threshold": T, "chunks": len(names),
               "routed_ids": [r["question_id"] for r in routed]}
    (work / "routing.json").write_text(json.dumps(summary, indent=1))
    print(f"prepare: {len(rows)} questions, routed {len(routed)} ({summary['agent_share']:.1%}; "
          f"{summary['no_margin']} with no margin), {len(names)} chunks, sandbox {box}")


# ---------------------------------------------------------------------------------------------------- agent side
CMD = r"(?:^|[;&|`(\n]|\$\()\s*(?:\S*/)?"
NET = re.compile(CMD + r"(curl|wget|nc|ncat|netcat|ssh|scp|sftp|rsync|ftp|telnet|pip3?|uv|uvx|npm|npx|node|git|brew|"
                 r"apt(?:-get)?|yum|dnf|docker|sudo|osascript)(?=\s|$)"
                 r"|https?://|\b(?:import|from)\s+(?:socket|urllib|requests|httpx|http|ftplib|smtplib|aiohttp)\b")
ABS_PATH = re.compile(r"(?<![\w.)\]])/(?:[\w.-]+/)+[\w.-]*")
SAFE_ABS = ("/dev/null", "/usr/bin/", "/bin/", "/usr/local/bin/", "/opt/homebrew/bin/")


def make_guard(box: Path, log, harness_dir: Path | None = None):
    """harness_dir: the agent's own Claude Code config folder, where the harness saves long tool outputs that the agent
    then reads back; reading it is allowed, writing is not."""
    box_s = str(box)
    hd = str(harness_dir) if harness_dir else None

    def inside(p, write=False):
        q = os.path.normpath(p if os.path.isabs(p) else os.path.join(box_s, p))
        ok = q == box_s or q.startswith(box_s + os.sep) or (not write and hd is not None and q.startswith(hd + os.sep))
        if write:
            ok = ok and (q.startswith(str(box / "answers") + os.sep) or q.startswith(str(box / "work") + os.sep))
        return ok

    async def guard(inp, tool_use_id, ctx):
        name, ti = inp.get("tool_name"), inp.get("tool_input") or {}
        why = None
        if name == "Bash":
            cmd = ti.get("command", "")
            if NET.search(cmd):
                why = "network, install and external commands are not allowed; use python3 TOOLS/sql.py"
            else:
                for tok in ABS_PATH.findall(cmd):
                    if not (inside(tok) or tok.startswith(SAFE_ABS)):
                        why = f"path outside the working directory: {tok}"
                        break
        elif name in ("Read", "Glob", "Grep"):
            p = ti.get("file_path") or ti.get("path") or box_s
            if not inside(p):
                why = f"path outside the working directory: {p}"
        elif name in ("Write", "Edit"):
            p = ti.get("file_path", "")
            if not inside(p, write=True):
                why = f"writes are allowed only under answers/ and work/: {p}"
        else:
            why = f"tool {name} is not allowed"
        if why:
            log({"guard_deny": name, "reason": why, "input": ti})
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": why}}
        return {}

    return guard


def chunk_ids(p: Path):
    return [json.loads(l)["question_id"] for l in open(p)]


def answered(p: Path):
    out = {}
    if p.exists():
        for ln in open(p):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and r.get("question_id") is not None:
                out.setdefault(str(r["question_id"]), r)
    return out


def to_jsonable(m):
    if dataclasses.is_dataclass(m):
        d = {"_type": type(m).__name__}
        for f in dataclasses.fields(m):
            d[f.name] = to_jsonable(getattr(m, f.name))
        return d
    if isinstance(m, (list, tuple)):
        return [to_jsonable(x) for x in m]
    if isinstance(m, dict):
        return {str(k): to_jsonable(v) for k, v in m.items()}
    return m if isinstance(m, (str, int, float, bool)) or m is None else str(m)


async def agent_session(box: Path, chunk: Path, out: Path, logf: Path, a):
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
    work = box.parent
    home = work / "agent_home"
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    env = {"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"], "HOME": str(home),
           "CLAUDE_CONFIG_DIR": str(home / ".claude"), "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
           "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    fh = open(logf, "a")

    def log(obj):
        fh.write(json.dumps({"t": time.time(), **obj}) + "\n")
        fh.flush()

    prompt = PROMPT + RUN_BLOCK.format(box=box, chunk=chunk, out=out)
    opts = ClaudeAgentOptions(
        model=MODEL, cwd=str(box), tools=TOOLS, allowed_tools=TOOLS, permission_mode="dontAsk",
        system_prompt={"type": "preset", "preset": "claude_code"}, setting_sources=[], skills=[],
        mcp_servers={}, strict_mcp_config=True, max_turns=200, env=env,
        settings=json.dumps({"autoMemoryEnabled": False}),
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[make_guard(box, log, home / ".claude")])]},
        sandbox=a.sandbox_settings)
    log({"event": "start", "chunk": chunk.name, "prompt": prompt})
    result = None
    async for m in query(prompt=prompt, options=opts):
        d = to_jsonable(m)
        log({"message": d})
        if d.get("_type") == "SystemMessage" and d.get("subtype") == "init":
            src = (d.get("data") or {}).get("apiKeySource")
            if src not in (None, "ANTHROPIC_API_KEY"):
                raise SystemExit(f"refusing to run: the agent would authenticate with {src}, not ANTHROPIC_API_KEY")
        if d.get("_type") == "ResultMessage":
            result = d
    log({"event": "end", "result": result and {k: result.get(k) for k in
                                                ("subtype", "is_error", "num_turns", "total_cost_usd", "usage",
                                                 "duration_ms", "api_error_status")}})
    fh.close()
    return result


def fake_session(chunk: Path, out: Path, logf: Path):
    """--dry-run: no API call. Picks candidate A (or NONE with an empty query) so the rest of the pipeline is tested."""
    with open(out, "a") as f:
        for ln in open(chunk):
            r = json.loads(ln)
            f.write(json.dumps({"question_id": r["question_id"], "pick": "A"} if r["candidates"]
                               else {"question_id": r["question_id"], "pick": "NONE", "sql": FILLER}) + "\n")
    logf.write_text(json.dumps({"event": "dry-run", "chunk": chunk.name}) + "\n")


async def run_chunk(name, box, a, sem):
    chunk = box / "chunks/C" / f"{name}.jsonl"
    out = box / "answers/C" / f"{name}.jsonl"
    logs = box.parent / "logs"
    logs.mkdir(exist_ok=True)
    need = chunk_ids(chunk)
    for attempt in range(a.retries + 1):
        missing = [q for q in need if q not in answered(out)]
        if not missing:
            return name, "done"
        cur = chunk
        if attempt or len(missing) < len(need):  # rerun only the unanswered questions, as a smaller chunk
            cur = box / "chunks/C" / f"{name}_r{attempt + 1}.jsonl"
            keep = set(missing)
            cur.write_text("".join(l for l in open(chunk) if json.loads(l)["question_id"] in keep))
        logf = logs / f"{cur.stem}.jsonl"
        async with sem:
            if a.dry_run:
                fake_session(cur, out, logf)
                continue
            try:
                res = await agent_session(box, cur, out, logf, a)
                if res and res.get("is_error") and res.get("api_error_status") in (429, 500, 529):
                    await asyncio.sleep(30 * (attempt + 1))
            except SystemExit:
                raise
            except Exception as e:  # API or CLI failure: log and retry the unanswered questions
                if "Invalid API key" in str(e) or "authentication" in str(e).lower():
                    raise SystemExit(f"stopping: {e}")
                with open(logf, "a") as fh:
                    fh.write(json.dumps({"t": time.time(), "event": "exception", "error": repr(e)}) + "\n")
                await asyncio.sleep(30 * (attempt + 1))
    missing = [q for q in need if q not in answered(out)]
    return name, "done" if not missing else f"{len(missing)} unanswered"


def sandbox_settings(mode):
    if mode == "off":
        return None
    have = sys.platform == "darwin" or shutil.which("bwrap")
    if mode == "auto" and not have:
        print("note: OS sandbox unavailable here (no bwrap); relying on the PreToolUse guard")
        return None
    return {"enabled": True, "autoAllowBashIfSandboxed": True, "allowUnsandboxedCommands": False}


def run(a):
    box = a.work.resolve() / "box"
    if not a.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY (or use --dry-run)")
    a.sandbox_settings = sandbox_settings(a.os_sandbox)
    names = sorted(p.stem for p in (box / "chunks/C").glob("*.jsonl") if not re.search(r"_r\d+$", p.stem))

    async def main():
        sem = asyncio.Semaphore(a.jobs)
        return await asyncio.gather(*(run_chunk(n, box, a, sem) for n in names))

    res = asyncio.run(main())
    bad = [(n, s) for n, s in res if s != "done"]
    print(f"run: {len(res) - len(bad)}/{len(res)} chunks complete" + (f"; incomplete: {bad}" if bad else ""))


# ---------------------------------------------------------------------------------------------------- merge, tokens
def merge(a):
    work = a.work.resolve()
    box = work / "box"
    rows = load_s1(a.cands)
    routed = set(json.loads((work / "routing.json").read_text())["routed_ids"])
    cands, ans = {}, {}
    for p in sorted((box / "chunks/C").glob("*.jsonl")):
        for ln in open(p):
            r = json.loads(ln)
            cands[r["question_id"]] = {c["label"]: c["sql"] for c in r["candidates"]}
    for p in sorted((box / "answers/C").glob("*.jsonl")):
        for q, r in answered(p).items():
            ans.setdefault(q, r)
    preds, n = {}, defaultdict(int)
    for r in rows:
        q, sql = r["question_id"], None
        if q in routed:
            x = ans.get(q)
            pick = str((x or {}).get("pick", "")).strip()
            if x is None:
                n["routed_no_answer_fallback"] += 1
            elif pick.upper() != "NONE" and pick in cands.get(q, {}):
                sql, n["routed_picked"] = cands[q][pick], n["routed_picked"] + 1
            elif (x.get("sql") or "").strip():
                sql, n["routed_own_sql"] = x["sql"], n["routed_own_sql"] + 1
            else:
                n["routed_bad_answer_fallback"] += 1
        else:
            n["kept"] += 1
        if sql is None:
            sql = s1_pick(r["cands"]) or FILLER
        sql = " ".join(str(sql).replace("\t", " ").split())
        preds[str(r["index"])] = f"{sql}\t----- bird -----\t{r['db_id']}"
    Path(a.out).write_text(json.dumps(preds, indent=1))
    rep = {"questions": len(rows), **n, "dry_run": bool(a.dry_run)}
    (work / "merge_report.json").write_text(json.dumps(rep, indent=1))
    print("merge:", json.dumps(rep))


def tokens(a):
    logs = a.work.resolve() / "logs"
    tot, sessions, cost = defaultdict(int), 0, 0.0
    for f in sorted(logs.glob("*.jsonl")):
        seen, s_tok = set(), defaultdict(int)
        for ln in open(f):
            r = json.loads(ln)
            m = r.get("message") or {}
            if m.get("_type") != "AssistantMessage" or not m.get("usage"):
                continue
            mid = m.get("message_id") or id(m)
            if mid in seen:
                continue
            seen.add(mid)
            for k in list(PRICE) + ["output_tokens"]:
                s_tok[k] += int(m["usage"].get(k) or 0)
        if seen:
            sessions += 1
            for k, v in s_tok.items():
                tot[k] += v
    prompt = sum(tot[k] for k in PRICE)
    cost = sum(tot[k] * v for k, v in PRICE.items()) / 1e6 + tot["output_tokens"] * PRICE_OUT / 1e6
    routing = json.loads((a.work.resolve() / "routing.json").read_text())
    rep = {"sessions": sessions, "routed_questions": routing["routed"], "questions": routing["n"],
           "prompt_tokens": prompt, "prompt_tokens_by_kind": {k: tot[k] for k in PRICE},
           "output_tokens": tot["output_tokens"], "list_price_usd": round(cost, 2),
           "prompt_tokens_per_question": round(prompt / max(1, routing["n"])),
           "prompt_tokens_per_routed_question": round(prompt / max(1, routing["routed"]))}
    (a.work.resolve() / "token_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "run", "merge", "tokens", "all"])
    ap.add_argument("--cands", type=Path, help="stage 1's <pred>.json.cands.jsonl")
    ap.add_argument("--db-root", type=Path, help="folder with <db_id>/<db_id>.sqlite and database_description/")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="final BIRD prediction file")
    ap.add_argument("--jobs", type=int, default=4, help="agent sessions in parallel")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--os-sandbox", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; fake answers (pipeline test only)")
    a = ap.parse_args()
    if a.step in ("prepare", "all"):
        prepare(a)
    if a.step in ("run", "all"):
        run(a)
    if a.step in ("merge", "all"):
        merge(a)
    if a.step in ("tokens", "all") and not a.dry_run:
        tokens(a)


if __name__ == "__main__":
    main()
