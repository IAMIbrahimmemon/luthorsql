"""Run qgen writer / checker batches through a local Ollama endpoint (e.g. DeepSeek V4.1 Flash on
Ollama Cloud) instead of Claude subagents. Same input and output files as the subagents use,
so the two can be mixed: e.g. DeepSeek writes, a Claude subagent checks (different model
families make the back-check more independent).

  python scripts/qgen_ollama.py write --batches data/qgen/batches [--only 0000 0001] [--plans-per-call 6] [--workers 4]
  python scripts/qgen_ollama.py check --batches data/qgen/batches [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://localhost:11434/api/chat"


def _parse(text: str) -> dict:
    """The model's JSON, tolerating a ```json fence or text around the object."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b + 1])
        raise


def chat(model: str, prompt: str, timeout: float = 300, retries: int = 3) -> dict:
    body = json.dumps({"model": model, "stream": False, "format": "json", "think": False,
                       "options": {"temperature": 0.7, "num_predict": 16384},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return _parse(json.load(r)["message"]["content"])
        except Exception as e:
            if k == retries - 1:
                raise
            # rate limit / quota: back off for minutes, not seconds
            time.sleep(120 * (k + 1) if "429" in str(e) else 3 * (k + 1))


def write_batch(path: Path, model: str, per_call: int) -> Path:
    w = json.loads(path.read_text())
    out_items = []
    for i in range(0, len(w["plans"]), per_call):
        chunk = w["plans"][i:i + per_call]
        prompt = (w["instructions"] + "\n\nPLANS (JSON):\n" + json.dumps(chunk, default=str)
                  + "\n\nReturn only the JSON object described above, covering every plan_id in PLANS.")
        try:
            res = chat(model, prompt)
        except Exception as e:
            print(f"  write failed for {path.name} chunk {i}: {e}", flush=True)
            res = {"items": []}
        got = {it.get("plan_id"): it for it in res.get("items", [])}
        for p in chunk:  # a plan the model dropped counts as skipped, never invented
            it = got.get(p["plan_id"]) or {"plan_id": p["plan_id"], "skip": True, "questions": []}
            it["questions"] = [q for q in it.get("questions") or [] if isinstance(q, dict) and q.get("text")]
            out_items.append(it)
    out = path.with_name(path.name.replace("writer_", "questions_"))
    out.write_text(json.dumps({"items": out_items, "writer": model}, indent=1))
    return out


TOPUP = """For each plan below, some questions are already written. Write ONLY the missing questions, in the
styles listed under "need", each phrased differently from the existing ones. Same rules as before: never write
table or column names or SQL words; use literal values exactly as given; ask for exactly the plan's result.
Output: JSON {"items": [{"plan_id": ..., "questions": [{"style": ..., "text": ...}]}]}"""


def topup_batch(qpath: Path, model: str, per_call: int = 5) -> int:
    """Fill plans that have fewer than 2 questions per style (incl. plans lost to failed chunks)."""
    w = json.loads(qpath.with_name(qpath.name.replace("questions_", "writer_")).read_text())
    plans = {p["plan_id"]: p for p in w["plans"]}
    d = json.loads(qpath.read_text())
    todo = []
    for it in d["items"]:
        have = [q.get("style") for q in it.get("questions") or []]
        need = [s for s in ("plain", "casual", "synonym") for _ in range(max(0, 2 - have.count(s)))]
        if need and not it.get("skip_reason_model"):
            todo.append((it, need))
    added = 0
    for i in range(0, len(todo), per_call):
        chunk = todo[i:i + per_call]
        payload = [{"plan_id": it["plan_id"], "schema": plans[it["plan_id"]]["schema"], "plan": plans[it["plan_id"]]["plan"],
                    "existing": [q["text"] for q in it.get("questions") or []], "need": need} for it, need in chunk]
        try:
            res = chat(model, w["instructions"] + "\n\n" + TOPUP + "\n\nPLANS (JSON):\n" + json.dumps(payload, default=str))
        except Exception as e:
            print(f"  topup failed for {qpath.name}: {e}", flush=True)
            continue
        got = {x.get("plan_id"): x for x in res.get("items", []) if isinstance(x, dict)}
        for it, need in chunk:
            new = [q for q in (got.get(it["plan_id"]) or {}).get("questions") or [] if isinstance(q, dict) and q.get("text")]
            seen = {q["text"].strip().lower() for q in it.get("questions") or []}
            for q in new:
                if q["text"].strip().lower() not in seen and q.get("style") in need:
                    need.remove(q["style"])
                    it.setdefault("questions", []).append(q)
                    seen.add(q["text"].strip().lower())
                    added += 1
            if it.get("questions"):
                it["skip"] = False
    qpath.write_text(json.dumps(d, indent=1))
    return added


def check_batch(path: Path, model: str, per_call: int = 4) -> Path:
    """per_call plans (each: one schema + its questions) per request. Resumable: verdicts already in
    checked_NNNN.json are kept and only items with a missing qid are asked again."""
    c = json.loads(path.read_text())
    out = path.with_name(path.name.replace("check_", "checked_"))
    have = {}
    if out.exists():
        try:
            have = {it["qid"]: it for it in json.loads(out.read_text())["items"] if it.get("qid")}
        except Exception:
            have = {}
    todo = [it for it in c["items"] if any(q["qid"] not in have for q in it["questions"])]
    for i in range(0, len(todo), per_call):
        chunk = todo[i:i + per_call]
        prompt = (c["instructions"] + "\n\nITEMS (JSON):\n" + json.dumps(chunk, default=str)
                  + "\n\nReturn only the JSON object described above, one entry per qid.")
        try:
            res = chat(model, prompt)
        except Exception as e:
            print(f"  check failed for {path.name} chunk {i}: {e}", flush=True)
            continue
        for it in res.get("items", []):
            if isinstance(it, dict) and it.get("qid"):
                have[it["qid"]] = it
        out.write_text(json.dumps({"items": list(have.values()), "checker": model}, indent=1))
    out.write_text(json.dumps({"items": list(have.values()), "checker": model}, indent=1))
    return out


def missing_verdicts(path: Path) -> int:
    c = json.loads(path.read_text())
    out = path.with_name(path.name.replace("check_", "checked_"))
    have = set()
    if out.exists():
        try:
            have = {it["qid"] for it in json.loads(out.read_text())["items"] if it.get("qid")}
        except Exception:
            pass
    return sum(q["qid"] not in have for it in c["items"] for q in it["questions"])


VERIFY = """You audit a text-to-SQL training dataset. For EACH item judge:
- faithful: does the question ask for exactly what the SQL returns? Same shown columns and aggregations
  (count of rows vs count of distinct things matters: COUNT(*) over a join counts joined rows, not entities),
  same filters and literal values, same grouping, ordering and limit. A question that is vaguer than the SQL,
  asks for extra or fewer things, misreads what a column means, or would make a competent analyst write a
  different query is NOT faithful. Equivalent phrasings that give the same result ARE faithful.
- natural: would a real person type this? No raw table/column identifiers, no SQL words, not garbled.
- problem: a short reason when either is false, else "".
Be strict. Output: JSON {"items": [{"vid": ..., "faithful": true/false, "natural": true/false, "problem": "..."}]}"""


def _slim(item: dict) -> dict:
    """Question + SQL + plan English + only the schema tables the SQL mentions (keeps prompts short)."""
    sql = item["sql"].lower()
    sch = [t for t in item.get("schema") or [] if t.get("table", "").lower() in sql]
    for t in sch:
        t["columns"] = [{k: c.get(k) for k in ("name", "type", "description", "examples")} for c in t["columns"]]
    return {"vid": item.get("vid") or item.get("aid"), "question": item["question"], "sql": item["sql"],
            "plan_english": item.get("plan_english"), "schema": sch}


def verify_file(path: Path, model: str, per_call: int = 8) -> Path:
    """Resumable: verdicts already in <name>.verified.jsonl are skipped."""
    items = [_slim(x) for x in json.loads(path.read_text())["items"]]
    out = path.with_suffix(".verified.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["vid"] for l in open(out) if l.strip()}
    todo = [x for x in items if x["vid"] not in done]
    for i in range(0, len(todo), per_call):
        chunk = todo[i:i + per_call]
        try:
            res = chat(model, VERIFY + "\n\nITEMS (JSON):\n" + json.dumps(chunk, default=str))
        except Exception as e:
            print(f"  verify failed for {path.name} chunk {i}: {e}", flush=True)
            continue
        ids = {x["vid"] for x in chunk}
        with open(out, "a") as f:
            for v in res.get("items", []):
                if isinstance(v, dict) and v.get("vid") in ids:
                    f.write(json.dumps(v) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "check", "topup", "verify"])
    ap.add_argument("--batches", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-v4.1-flash:cloud")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--plans-per-call", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "verify":  # --batches: a folder of verify_*.json / audit_*.json files
        files = sorted(a.batches.glob("audit_[0-9].json")) + sorted(a.batches.glob("verify_*.json"))
        files = [f for f in files if not f.name.endswith(".verified.jsonl")]
        with ThreadPoolExecutor(a.workers) as ex:
            for out in ex.map(lambda f: verify_file(f, a.model), files):
                print("verified", out, flush=True)
        return
    if a.cmd == "topup":
        files = sorted(a.batches.glob("questions_*.json"))
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            for f, n in zip(files, ex.map(lambda f: topup_batch(f, a.model), files)):
                print(f"topup {f.name}: +{n} questions {time.time() - t0:.0f}s", flush=True)
        return
    pat = "writer_*.json" if a.cmd == "write" else "check_*.json"
    files = [f for f in sorted(a.batches.glob(pat)) if not a.only or f.stem.split("_")[1] in a.only]
    done_pat = "questions_" if a.cmd == "write" else "checked_"
    if a.cmd == "write":
        files = [f for f in files if not (f.parent / f.name.replace(f.name.split("_")[0] + "_", done_pat)).exists()]  # resume
    else:  # checks resume per question: redo a batch only while verdicts are missing
        files = [f for f in files if missing_verdicts(f) > 0]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        fn = (lambda f: write_batch(f, a.model, a.plans_per_call)) if a.cmd == "write" else (lambda f: check_batch(f, a.model))
        for out in ex.map(fn, files):
            print("wrote", out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
