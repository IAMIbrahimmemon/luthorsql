"""Verify Claude-verify packs with Gemini through the Antigravity CLI (`agy`, Google AI Pro login).

Same judgement as the Claude subagents (faithful / natural / problem). Output goes next to the pack as
pack_NNNN.<tag>.jsonl, so it never mixes with Claude's pack_NNNN.result.jsonl. Resumable: vids already
written are skipped.

  python scripts/gemini_verify.py --packs 270 271 --model gemini-3.1-pro-high --tag gemini
  python scripts/gemini_verify.py --calibrate 300 --model ...   # re-judge items Claude already judged

Before Gemini's verdicts are used for training, --calibrate measures them against Claude's labels.
Only public datasets go through here (SynSQL / BIRD / Spider).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "claude_verify"

RULES = """You are a strict quality verifier for a text-to-SQL training set. For EACH item decide:
- faithful: does the question ask for exactly what the SQL returns? Same shown columns and aggregations (count of rows vs count of distinct things matters; COUNT(*) over a join counts joined rows), same filters and literal values, same grouping, ordering and limit ("top N" needs ORDER BY and LIMIT). A question that is vaguer than the SQL, asks for extra or fewer things, or would make a competent analyst write a different query is NOT faithful. Equivalent phrasings that give the same result ARE faithful. A trailing "Context:" line is part of the question.
- natural: would a real person type this? No raw table/column identifiers (also not in the Context line), no SQL words, not garbled. Formal wording is fine.
- problem: short reason when either is false, else "".
Do not use any tools. Do not read or write files. Answer ONLY with a JSON array, one object per item, in input order:
[{"vid": "...", "faithful": true, "natural": true, "problem": ""}, ...]"""


def render(x: dict) -> dict:
    sch = [{"table": t["table"], "columns": [{k: c.get(k) for k in ("name", "type", "description")} for c in t["columns"]]}
           for t in x.get("schema") or []]
    return {"vid": x["vid"], "question": x["question"], "sql": x["sql"], "schema": sch}


def parse(out: str) -> list[dict]:
    m = re.search(r"\[\s*\{.*\}\s*\]", out, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []


def judge(batch: list[dict], model: str, timeout: str) -> list[dict]:
    prompt = RULES + "\n\nITEMS:\n" + json.dumps([render(x) for x in batch], ensure_ascii=False)
    with tempfile.TemporaryDirectory() as td:  # empty working dir: nothing for the agent to touch
        if model.startswith("codex"):  # ChatGPT Plus via Codex CLI, read-only sandbox; "codex" or "codex:<model>"
            last = Path(td) / "last.txt"
            cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "-o", str(last)]
            if ":" in model:
                cmd += ["-m", model.split(":", 1)[1]]
            subprocess.run(cmd + ["-"], input=prompt, cwd=td, capture_output=True, text=True)
            text = last.read_text() if last.exists() else ""
        else:  # Gemini via Antigravity CLI
            text = subprocess.run(["agy", "-p", prompt, "--model", model, "--print-timeout", timeout,
                                   "--disable-slash-commands"], cwd=td, capture_output=True, text=True).stdout
    got = {g.get("vid"): g for g in parse(text) if isinstance(g, dict)}
    out = []
    for x in batch:
        g = got.get(x["vid"])
        if g is not None and isinstance(g.get("faithful"), bool) and isinstance(g.get("natural"), bool):
            out.append({"vid": x["vid"], "faithful": g["faithful"], "natural": g["natural"], "problem": str(g.get("problem", ""))})
    return out


def run(items: list[dict], out: Path, model: str, batch: int, workers: int, timeout: str):
    done = set()
    if out.exists():
        done = {json.loads(l)["vid"] for l in open(out) if l.strip()}
    todo = [x for x in items if x["vid"] not in done]
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    with ThreadPoolExecutor(workers) as ex, open(out, "a") as f:
        for res in ex.map(lambda b: judge(b, model, timeout), batches):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{out.name}: +{len(res)}", flush=True)


def claude_labels() -> dict[str, dict]:
    lab = {}
    for f in D.glob("pack_*.result.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
                lab[r["vid"]] = r
            except Exception:
                pass
    return lab


def report(pred: Path, lab: dict):
    rows = [json.loads(l) for l in open(pred) if l.strip()]
    rows = [r for r in rows if r["vid"] in lab]
    ok = lambda r: r["faithful"] and r["natural"]
    n = len(rows)
    acc = sum(ok(r) for r in rows)
    both = sum(ok(r) and ok(lab[r["vid"]]) for r in rows)
    c_ok = sum(ok(lab[r["vid"]]) for r in rows)
    agree = sum(ok(r) == ok(lab[r["vid"]]) for r in rows)
    print(json.dumps({"n": n, "gemini_keep": round(acc / n, 3), "claude_keep": round(c_ok / n, 3),
                      "agreement": round(agree / n, 3),
                      "kept_set_claude_ok (precision)": round(both / max(acc, 1), 3),
                      "claude_ok_recovered (recall)": round(both / max(c_ok, 1), 3)}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="*", type=int, default=[])
    ap.add_argument("--calibrate", type=int, default=0, help="re-judge N random items Claude already judged")
    ap.add_argument("--model", default="gemini-3.8-flash-high")
    ap.add_argument("--tag", default="gemini")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", default="15m")
    a = ap.parse_args()
    if a.calibrate:
        lab = claude_labels()
        pool = []
        for f in sorted(D.glob("pack_*.json")):
            n = int(f.stem.split("_")[1])
            if n < 142:  # SynSQL packs only (the source we will use Gemini on)
                continue
            pool += [x for x in json.loads(f.read_text())["items"] if x["vid"] in lab]
        random.Random(0).shuffle(pool)
        out = D / f"calib.{a.tag}.{a.model}.jsonl"
        run(pool[:a.calibrate], out, a.model, a.batch, a.workers, a.timeout)
        report(out, lab)
        return
    for p in a.packs:
        items = json.loads((D / f"pack_{p:04d}.json").read_text())["items"]
        cf = D / f"pack_{p:04d}.result.jsonl"  # items Claude already judged are not re-judged
        if cf.exists():
            seen = {json.loads(l)["vid"] for l in open(cf) if l.strip()}
            items = [x for x in items if x["vid"] not in seen]
        run(items, D / f"pack_{p:04d}.{a.tag}.jsonl", a.model, a.batch, a.workers, a.timeout)


if __name__ == "__main__":
    sys.exit(main())
