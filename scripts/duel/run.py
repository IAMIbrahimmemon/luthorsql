"""Zero-shot gated duel (memo 2026-10-10 §3 Step 2a), pre-registered in runs/duel/PREREG.md.

  uv run python scripts/duel/run.py run     # 2 orders x gated pairs -> runs/duel/calls.jsonl (resumable)
  uv run python scripts/duel/run.py score   # pairwise accuracy + end-to-end DECISION -> stdout

Model: Perplexity Decider v1.1 27B via OpenRouter's SystemOne route, request format and key handling as
scripts/analysis/jev_judge.py (key via jev_judge.key('openrouter_api_key'), never printed; leak gate before every
request). One prompt, no variants. Spend cap $2 (provider-reported cost), max 3 concurrent requests.
DECISION only; never reads RESERVED, sealed sets or BIRD dev.
"""
import json
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
sys.path.insert(0, str(ROOT / "scripts/duel"))
from build import render  # noqa: E402
from phase1_eval import IDS, SPLIT, compare, load, sum_z_hits  # noqa: E402

assert SPLIT == "decision"
MODEL, URL, KEYNAME = "perplexity/pplx-decider-v1.1-27b", "https://openrouter.ai/api/v1/systemone", "openrouter_api_key"
CAP_USD, WORKERS = 2.0, 3
PAIRS, INPUTS = ROOT / "data/duel/decision_pairs.jsonl", ROOT / "data/duel/decision_inputs.jsonl"
CALLS = ROOT / "runs/duel/calls.jsonl"
W = ROOT / "runs/phase1/p1w/p1w"
INSTR = ("Two candidate SQL queries, A and B, were written for this question on this SQLite database; they return "
         "different results and exactly one is intended to be correct. Pick the candidate whose result answers the "
         "question as the evidence defines it. Use the contrast: the clause-level SQL differences, the rows each "
         "returns that the other does not, column names, row counts, NULL counts, and which evidence clauses each "
         "implements. Check: only the columns asked for, in the order asked; the filters and joins the evidence "
         "implies; aggregation and grouping; COUNT vs COUNT(DISTINCT) and duplicate rows; ordering/LIMIT for "
         "highest/top/most questions; percentages, divisions and rounding exactly as the evidence states.")


def body(inp, first):
    return {"model": MODEL, "state": render(inp, first),
            "questions": {"correct": {"type": "choice", "instructions": INSTR,
                                      "criteria": {"A": "Candidate A is the correct query",
                                                   "B": "Candidate B is the correct query"}}}}


def post(b, k_):
    hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + k_}
    err = None
    for t in range(5):
        req = urllib.request.Request(URL, data=json.dumps(b).encode(), headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(5 * 2 ** t)
                err = f"HTTP {e.code}"
                continue
            raise RuntimeError(f"HTTP {e.code} {msg}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(5 * 2 ** t)
            err = e
    raise RuntimeError(f"gave up: {err}")


def run():
    from jev_judge import key
    from luthor.teacher.leak import LeakGate
    gate, k_ = LeakGate(), key(KEYNAME)
    inps = {r["id"]: r for r in map(json.loads, open(INPUTS))}
    pairs = [json.loads(l) for l in open(PAIRS)]
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["first"]) for r in map(json.loads, open(CALLS))} if CALLS.exists() else set()
    spent = [sum(float((r.get("usage") or {}).get("cost") or 0) for r in map(json.loads, open(CALLS)))
             if CALLS.exists() else 0.0]
    lock, tok = threading.Lock(), [0, 0]
    todo = [(p["id"], f) for p in pairs for f in ("pick", "runner") if (p["id"], f) not in done]
    print(f"{len(todo)} calls to make ({len(done)} done, ${spent[0]:.4f} spent)", flush=True)

    def one(job):
        i, first = job
        with lock:
            if spent[0] >= CAP_USD:
                return
        b = body(inps[i], first)
        rec = {"id": i, "first": first}
        if gate.hit(json.dumps(b)) is not None:
            rec.update(choice=None, error="leak gate: not sent")
        else:
            try:
                d = post(b, k_)
                ans = d["answers"]["correct"]
                L = ans.get("choice")
                second = "runner" if first == "pick" else "pick"
                rec.update(letter=L, choice={"A": first, "B": second}.get(L), prob=ans.get("probabilities"),
                           confidence=ans.get("confidence"), served=d.get("model"), usage=d.get("usage"))
            except Exception as e:  # noqa: BLE001
                rec.update(choice=None, error=f"{type(e).__name__}: {str(e)[:200]}")
        u = rec.get("usage") or {}
        with lock:
            spent[0] += float(u.get("cost") or 0)
            tok[0] += int(u.get("input_tokens") or 0)
            tok[1] += int(u.get("output_tokens") or 0)
            with open(CALLS, "a") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))
    print(f"done: in={tok[0]:,} out={tok[1]:,} tokens this session; total spent ${spent[0]:.4f} "
          f"(cap ${CAP_USD})", flush=True)


def score():
    pairs = {p["id"]: p for p in map(json.loads, open(PAIRS))}
    calls = {}
    for r in map(json.loads, open(CALLS)):
        calls[(r["id"], r["first"])] = r
    verdict = {}
    for i in pairs:
        a, b = calls.get((i, "pick"), {}).get("choice"), calls.get((i, "runner"), {}).get("choice")
        verdict[i] = a if (a is not None and a == b) else None
    errs = sum(1 for r in calls.values() if r.get("error"))
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in calls.values())
    tin = sum(int((r.get("usage") or {}).get("input_tokens") or 0) for r in calls.values())
    served = sorted({r.get("served") for r in calls.values() if r.get("served")})
    print(f"calls {len(calls)}, errors {errs}, input tokens {tin:,}, cost ${cost:.4f}, served {served}")

    one = [i for i, p in pairs.items() if p["label"] in ("pick", "runner")]
    agree = [i for i in one if verdict[i] is not None]
    right = [i for i in agree if verdict[i] == pairs[i]["label"]]
    acc = len(right) / len(agree) if agree else float("nan")
    eff = sum((verdict[i] or "pick") == pairs[i]["label"] for i in one) / len(one)
    print(f"86-pair set: n={len(one)} (pick right {sum(pairs[i]['label'] == 'pick' for i in one)}, "
          f"runner right {sum(pairs[i]['label'] == 'runner' for i in one)})")
    print(f"  order agreement {len(agree)}/{len(one)} = {len(agree) / len(one):.3f}")
    print(f"  PRIMARY pairwise accuracy (order-agreeing) {len(right)}/{len(agree)} = {acc:.3f}")
    for lab in ("pick", "runner"):
        ag = [i for i in agree if pairs[i]["label"] == lab]
        print(f"    when {lab} right: {sum(verdict[i] == lab for i in ag)}/{len(ag)} agreeing verdicts correct")
    print(f"  effective accuracy (disagreement -> validator pick) {eff:.3f}")
    # per-order accuracy (information)
    for f in ("pick", "runner"):
        cs = [calls.get((i, f), {}).get("choice") for i in one]
        print(f"  single order A={f}: {sum(c == pairs[i]['label'] for c, i in zip(cs, one))}/{len(one)} "
              f"(chose A {sum(c == f for c in cs)}x)")
    bw = [i for i, p in pairs.items() if p["label"] == "both_wrong"]
    print(f"both-wrong gated questions: {len(bw)}; agreeing verdicts {sum(verdict[i] is not None for i in bw)} "
          f"(no effect on EX)")

    b1, b2 = load(W / "heldout_union_by_r1.jsonl"), load(W / "heldout_union_by_r2.jsonl")
    base, orc = sum_z_hits(b1, b2)
    duel = dict(base)
    over = 0
    for i, p in pairs.items():
        if verdict[i] == "runner":
            over += 1
            duel[i] = p["ok_runner"]
        elif verdict[i] == "pick":
            duel[i] = p["ok_pick"]
        assert base[i] == p["ok_pick"], i
    print(f"overrides (agreeing verdict for runner-up): {over} of {len(pairs)} gated questions")
    c = compare("sum_z pick (wide pool)", base, "gated duel (margin<0.5)", duel, orc, orc)
    pv = "GO (zero-shot)" if acc >= 0.75 else ("FINE-TUNE candidate" if acc >= 0.62 else "STOP")
    kill = c["delta"] < 1.0
    print(f"VERDICT pairwise: {pv}; end-to-end gain {c['delta']:+.2f} -> "
          f"{'KILL (gain < +1.0)' if kill else 'gain bar met'}")
    return verdict


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else score()
