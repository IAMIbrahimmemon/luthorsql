"""LLM judge over the wide (k=32) pool: how much of the validator gap a stronger picker recovers.
Held-out v2 DECISION only (8 held-out BIRD-train DBs); eval-only, never training data. Never reads BIRD dev.

  python scripts/analysis/llm_judge.py run   --model deepseek-v4.1-flash:cloud --k 4 --workers 3
  python scripts/analysis/llm_judge.py score --model deepseek-v4.1-flash:cloud --k 4

Per covered question: candidates of the k=32 union pool (runs/phase1/p1w/p1w) that run and return non-null rows,
one per distinct result signature, ordered by sum_z (both wide validators). The top K are shown to the judge in a
per-question shuffled order (seeded by the id, so the validator's order leaks nothing), each as SQL + the first
result rows. With fewer than 2 distinct results there is no call and the sum_z pick stands.
Prompts carry public BIRD-train schemas/questions only, go through the leak gate (providers.Provider.chat) and
are counted in data/teacher/usage.jsonl. Output: <out>/judge_<model>_k<K>.jsonl (resumable).
"""
import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from phase1_eval import IDS, compare, load, rr_arr, z  # noqa: E402

W = ROOT / "runs/phase1/p1w/p1w"
OUT = ROOT / "runs/judge"
CARDS = ROOT / "data/sys1/bird_train/cards"
DBS = ROOT / "data/raw/bird_train/train_databases"
LETTERS = "ABCDEFGH"

PROMPT = """You are checking candidate SQL queries for a text-to-SQL question on a SQLite database.
Exactly one candidate is intended to be correct; pick the one whose result best answers the question as the
evidence defines it. Check: the right columns in the right order (only what is asked), the filters and joins the
evidence implies, aggregation/grouping, DISTINCT vs duplicates, ordering/LIMIT for "highest/top/most" questions.

Database schema:
{schema}

Evidence: {evidence}
Question: {question}

{cands}

Think briefly, then end with one line exactly: Answer: <letter>"""


def preview(db, sql, n=5, seconds=10.0):
    con = sqlite3.connect(f"file:{DBS / db / (db + '.sqlite')}?mode=ro", uri=True)
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > seconds else 0, 10000)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(n + 1)
        more = len(rows) > n
        body = "\n".join(" | ".join(str(v)[:60] for v in r) for r in rows[:n])
        return f"columns: {' | '.join(cols)}\n{body}" + ("\n..." if more else "")
    except Exception as e:  # noqa: BLE001
        return f"(error: {str(e)[:100]})"
    finally:
        con.close()


def shortlist(r1, r2, k):
    c = r2["cands"]
    s = z(rr_arr(r1["cands"])) + z(rr_arr(c))
    live = [j for j, x in enumerate(c) if x.get("runs") and x.get("nonnull")] or list(range(len(c)))
    best = {}
    for j in live:
        key = c[j].get("sig") or c[j]["sql"]
        if key not in best or s[j] > s[best[key]]:
            best[key] = j
    return sorted(best.values(), key=lambda j: -s[j])[:k]


def build(row, order, card):
    from luthor.sqlw.prompt6 import ddl6
    ev = (row.get("context") or "").strip() or "(none)"
    blocks = []
    for L, j in zip(LETTERS, order):
        c = row["cands"][j]
        res = c.get("res") or ""
        blocks.append(f"Candidate {L}:\nSQL: {c['sql']}\nResult ({res.split(';')[0]}):\n{preview(row['db_id'], c['sql'])}")
    return PROMPT.format(schema=ddl6(card, row["question"], row.get("context") or ""), evidence=ev,
                         question=row["question"], cands="\n\n".join(blocks))


def out_path(model, k):
    return OUT / f"judge_{re.sub(r'[^A-Za-z0-9.]+', '_', model)}_k{k}.jsonl"


def run(a):
    from luthor.sq.schema_card import SchemaCard
    from luthor.teacher.leak import LeakGate
    from luthor.teacher.ledger import Ledger
    from luthor.teacher.providers import make
    if a.model.lower().removeprefix("ollama:").startswith("glm"):
        raise SystemExit("GLM is off by owner decision")
    b1, b2 = load(W / "heldout_union_by_r1.jsonl"), load(W / "heldout_union_by_r2.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    op = out_path(a.model, a.k)
    done = {json.loads(l)["id"] for l in open(op)} if op.exists() else set()
    prov = make(a.model, LeakGate(), ledger=Ledger(ROOT / "data/teacher/usage.jsonl"), job="judge")
    lock, cards = threading.Lock(), {}
    todo = [i for i in IDS if i in b2 and i not in done]
    print(f"{len(todo)} to judge ({len(done)} done)", flush=True)
    tok = [0, 0]

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
            msg = build(b2[i], order, cards[db])
            try:
                rep = prov.chat([{"role": "user", "content": msg}], temperature=0.0, max_tokens=a.max_tokens,
                                think=a.think)
                m = re.findall(r"Answer:\s*\**\s*([A-H])\b", rep.text)
                pick = order[LETTERS.index(m[-1])] if m and LETTERS.index(m[-1]) < len(order) else None
                rec.update(order=order, pick=pick, raw=rep.text[-400:], n_in=rep.n_in, n_out=rep.n_out)
                tok[0] += rep.n_in
                tok[1] += rep.n_out
            except Exception as e:  # noqa: BLE001
                rec.update(order=order, pick=None, error=f"{type(e).__name__}: {str(e)[:200]}")
        with lock, open(op, "a") as f:
            f.write(json.dumps(rec) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for n, _ in enumerate(ex.map(one, todo), 1):
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} in={tok[0]:,} out={tok[1]:,} {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"JUDGE done in={tok[0]:,} out={tok[1]:,} -> {op}", flush=True)


def score(a):
    b2 = load(W / "heldout_union_by_r2.jsonl")
    b1 = load(W / "heldout_union_by_r1.jsonl")
    recs = {r["id"]: r for r in map(json.loads, open(out_path(a.model, a.k)))}
    base, judge, orc, topk = {}, {}, {}, {}
    called = parsed = 0
    for i in IDS:
        if i not in b2:
            base[i] = judge[i] = orc[i] = topk[i] = False
            continue
        c = b2[i]["cands"]
        top = shortlist(b1[i], b2[i], a.k)
        base[i] = bool(c[top[0]].get("ok")) if top else False
        r = recs.get(i, {})
        called += "order" in r
        parsed += r.get("pick") is not None
        p = r.get("pick") if r.get("pick") is not None else (top[0] if top else None)
        judge[i] = p is not None and bool(c[p].get("ok"))
        orc[i] = any(x.get("ok") for x in c)
        topk[i] = any(c[j].get("ok") for j in top)
    print(f"judge {a.model} k={a.k}: {called} calls, {parsed} parsed picks; right answer in top-{a.k}: "
          f"{100 * sum(topk.values()) / len(IDS):.2f}")
    compare("sum_z pick (wide pool)", base, f"LLM judge top-{a.k}", judge, orc, orc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--model", default="deepseek-v4.1-flash:cloud")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--think", type=lambda s: s.lower() == "true", default=False)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)
