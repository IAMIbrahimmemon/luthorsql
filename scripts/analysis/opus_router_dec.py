"""System 1 + Opus router on DECISION: System 1 (R1 generator k=32 + R1 validator) answers; the least-confident
questions (no grammar candidate first, then smallest top-2 distinct-result validator margin) go to Opus 5.5 (agent
with read-only SQL access). Strict EX vs the original BIRD-train gold.

  python3 scripts/analysis/opus_router_dec.py <scratchpad>/opus_dec
"""
import glob, json, sqlite3, sys, threading
from concurrent.futures import ProcessPoolExecutor
D = sys.argv[1]
DB = "data/raw/bird_train/train_databases"

def run(db, sql, t=60):
    con = sqlite3.connect(f"file:{DB}/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    tm = threading.Timer(t, con.interrupt); tm.start()
    try:
        return set(con.execute(sql).fetchall())
    except Exception:
        return None
    finally:
        tm.cancel()

def job(a):
    qid, db, pred, gold = a
    p, g = run(db, pred), run(db, gold)
    return qid, p is not None and g is not None and p == g

if __name__ == "__main__":
    ids = json.load(open("data/sys1/heldout_v2/decision.json"))["ids"]; n = len(ids)
    tr = json.load(open("data/raw/bird_train/train.json"))
    order = [r["id"] for r in json.load(open(f"{D}/order.json"))]
    s1 = {}
    for ln in open("runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"):
        r = json.loads(ln)
        c = [x for x in r["cands"] if x.get("src") in ("r1", "both")]
        lv = [x for x in c if x.get("runs") and x.get("nonnull")] or c
        if lv:
            s1[r["id"]] = bool(max(lv, key=lambda x: x["rr"])["ok"])
    ans = {}
    for f in glob.glob(f"{D}/answers/*.jsonl"):
        for ln in open(f):
            try:
                r = json.loads(ln); ans[str(r["question_id"])] = r["sql"]
            except Exception:
                pass
    jobs = [(q, tr[int(q.rsplit("_", 1)[1])]["db_id"], s, tr[int(q.rsplit("_", 1)[1])]["SQL"]) for q, s in ans.items()]
    with ProcessPoolExecutor(6) as ex:
        op = dict(ex.map(job, jobs, chunksize=4))
    base = sum(s1.get(i, False) for i in ids)
    print(f"System 1 alone: {100*base/n:.2f}  | Opus answered {len(op)} of {n}")
    done = 0
    while done < len(order) and order[done] in op:
        done += 1
    pref = order[:done]
    if pref:
        print(f"escalated prefix (least confident first) fully answered: {done}")
        print(f"  on those {done}: System 1 {100*sum(s1.get(i,False) for i in pref)/done:.1f}%  Opus {100*sum(op[i] for i in pref)/done:.1f}%")
    for k in sorted({x for x in (25, 50, 75, 100, 126, 150, 200, 250, 300, 400, 500, 600, 700, 796) if x <= done} | {done}):
        esc = set(order[:k])
        sc = sum(op[i] if i in esc else s1.get(i, False) for i in ids)
        print(f"  escalate {k:4d} ({100*k/n:5.1f}%): {100*sc/n:.2f}")
    if len(op) == n:
        print(f"Opus alone: {100*sum(op.values())/n:.2f}")
    json.dump(op, open(f"{D}/scored.json", "w"))
