"""Decision-model judge (TypeSafe Jev / SystemOne API) over the wide (k=32) pool. Held-out v2 DECISION only;
eval-only. Never reads BIRD dev. Same shortlist and per-question shuffle as llm_judge.py.

  JEV_KEY_FILE=.env.txt python scripts/analysis/jev_judge.py run   [--model jev-1.13.0] [--k 4] [--workers 8]
  python scripts/analysis/jev_judge.py run --model cloudflare/clef --url https://openrouter.ai/api/v1/systemone \
      --key-name openrouter_api_key
  python scripts/analysis/jev_judge.py score [--model jev-1.13.0] [--k 4] [--thr 0.8]

Pre-registered rule (docs/BUDGET500_PLAN.md log): final pick = the judge's choice when its probability >= 0.8,
else the sum_z pick. Other thresholds are printed for information only.
The key is read from the file named by JEV_KEY_FILE (line jev_api_key=...) and never printed. Every request body
goes through the leak gate first. --url points at any SystemOne-compatible endpoint (e.g. a local Clef server).
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from llm_judge import CARDS, IDS, LETTERS, OUT, W, load, preview, shortlist  # noqa: E402
from phase1_eval import compare  # noqa: E402

INSTR = ("Exactly one candidate SQL query is intended to answer the question correctly on this SQLite database. "
         "Pick the candidate whose result best answers the question as the evidence defines it: the right columns "
         "in the right order (only what is asked), the filters and joins the evidence implies, aggregation and "
         "grouping, DISTINCT vs duplicates, ordering/LIMIT for highest/top/most questions.")


def key(name="jev_api_key"):
    f = os.environ.get("JEV_KEY_FILE", str(ROOT / ".env.txt"))
    for ln in open(f):
        k, _, v = ln.partition("=")
        if k.strip().lower() == name:
            return v.strip().strip("'\"")
    raise SystemExit(f"no {name} in {f}")


def out_path(model, k):
    from phase1_eval import SPLIT
    tag = "" if SPLIT == "decision" else f"_{SPLIT}"
    return OUT / f"jev_{re.sub(r'[^A-Za-z0-9.]+', '_', model)}_k{k}{tag}.jsonl"


def body(row, order, card, model):
    from luthor.sqlw.prompt6 import ddl6
    cands = {}
    for L, j in zip(LETTERS, order):
        c = row["cands"][j]
        cands[L] = {"sql": c["sql"], "result": f"{(c.get('res') or '').split(';')[0]}\n{preview(row['db_id'], c['sql'])}"}
    state = {"database_schema": ddl6(card, row["question"], row.get("context") or ""),
             "evidence": (row.get("context") or "").strip() or "(none)", "question": row["question"],
             "candidates": cands}
    crit = {L: f"Candidate {L} is the correct query" for L in cands}
    return {"model": model, "state": state, "questions": {"correct": {"type": "choice", "instructions": INSTR,
                                                                      "criteria": crit}}}


def run(a):
    from luthor.sq.schema_card import SchemaCard
    from luthor.teacher.leak import LeakGate
    gate, k_ = LeakGate(), (key(a.key_name) if a.auth else "")
    b1, b2 = load(W / "heldout_union_by_r1.jsonl"), load(W / "heldout_union_by_r2.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    op = out_path(a.model, a.k)
    done = {json.loads(l)["id"] for l in open(op)} if op.exists() else set()
    lock, cards, tok = threading.Lock(), {}, [0]
    todo = [i for i in IDS if i in b2 and i not in done]
    print(f"{len(todo)} to judge ({len(done)} done)", flush=True)

    def post(b):
        hdr = {"Content-Type": "application/json"}
        if k_:
            hdr["Authorization"] = "Bearer " + k_
        for t in range(5):
            req = urllib.request.Request(a.url, data=json.dumps(b).encode(), headers=hdr)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(5 * 2 ** t)
                    continue
                raise RuntimeError(f"HTTP {e.code} {msg}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                time.sleep(5 * 2 ** t)
                err = e
        raise RuntimeError(f"gave up: {err if 'err' in dir() else 'retries'}")

    def one(i):
        top = shortlist(b1[i], b2[i], a.k)
        rec = {"id": i, "top": top}
        if len(top) >= 2:
            order = top[:]
            random.Random(int(hashlib.md5(i.encode()).hexdigest()[:8], 16)).shuffle(order)
            db = b2[i]["db_id"]
            with lock:
                if db not in cards:
                    cards[db] = SchemaCard.load(CARDS / f"{db}.json")
            b = body(b2[i], order, cards[db], a.model)
            txt = json.dumps(b)
            rec["order"] = order
            if gate.hit(txt) is not None:
                rec.update(pick=None, error="leak gate: not sent")
            else:
                try:
                    d = post(b)
                    ans = d["answers"]["correct"]
                    L = ans.get("choice")
                    rec.update(pick=order[LETTERS.index(L)] if L in LETTERS[:len(order)] else None,
                               prob={order[LETTERS.index(x)]: p for x, p in (ans.get("probabilities") or {}).items()
                                     if x in LETTERS[:len(order)]},
                               confidence=ans.get("confidence"), served=d.get("model"), usage=d.get("usage"))
                    tok[0] += int((d.get("usage") or {}).get("input_tokens") or 0)
                except Exception as e:  # noqa: BLE001
                    rec.update(pick=None, error=f"{type(e).__name__}: {str(e)[:200]}")
        with lock, open(op, "a") as f:
            f.write(json.dumps(rec) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for n, _ in enumerate(ex.map(one, todo), 1):
            if n % 100 == 0:
                print(f"  {n}/{len(todo)} in={tok[0]:,} {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"JEV done in={tok[0]:,} -> {op}", flush=True)


def score(a):
    b1, b2 = load(W / "heldout_union_by_r1.jsonl"), load(W / "heldout_union_by_r2.jsonl")
    recs = {r["id"]: r for r in map(json.loads, open(out_path(a.model, a.k)))}
    errs = sum(1 for r in recs.values() if r.get("error"))
    served = {r.get("served") for r in recs.values() if r.get("served")}

    def hits(thr):
        base, out, orc = {}, {}, {}
        for i in IDS:
            if i not in b2:
                base[i] = out[i] = orc[i] = False
                continue
            c = b2[i]["cands"]
            top = shortlist(b1[i], b2[i], a.k)
            p0 = top[0] if top else None
            base[i] = p0 is not None and bool(c[p0].get("ok"))
            r = recs.get(i, {})
            p = r.get("pick")
            conf = (r.get("prob") or {}).get(str(p), (r.get("prob") or {}).get(p, 0.0)) if p is not None else 0.0
            use = p if (p is not None and conf >= thr) else p0
            out[i] = use is not None and bool(c[use].get("ok"))
            orc[i] = any(x.get("ok") for x in c)
        return base, out, orc

    print(f"jev {a.model} (served {sorted(served)}) k={a.k}: {len(recs)} records, {errs} errors")
    base, out, orc = hits(a.thr)
    print(f"PRE-REGISTERED: judge pick when p >= {a.thr}, else sum_z")
    c = compare("sum_z pick (wide pool)", base, f"jev override p>={a.thr}", out, orc, orc)
    print(f"  -> {'PASS' if c['delta'] >= 1.0 and c['p'] < 0.10 else 'NO PASS'} (gate: >= +1.0, p < 0.10)")
    print("information only (not chosen from):")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.9, 0.95):
        _, o, _ = hits(thr)
        print(f"  thr {thr:4.2f}: {100 * sum(o.values()) / len(IDS):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--url", default="https://api.typesafe.ai/v1/systemone")
    ap.add_argument("--no-auth", dest="auth", action="store_false")
    ap.add_argument("--key-name", default="jev_api_key", help="line in .env.txt, e.g. openrouter_api_key")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.8)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)
