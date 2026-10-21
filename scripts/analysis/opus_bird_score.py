"""Score Opus-agent BIRD dev answers (strict EX: result set equals the gold result set).

  python3 scripts/analysis/opus_bird_score.py <answers_dir>
Gold: data/bench/dev_20240627/dev.json (original dev, as our 59.06). Also reports the 2025-11-06 revised gold.
Unanswered questions are reported separately (not counted wrong) since the run fills chunks in a shuffled order.
"""
import collections, glob, json, sqlite3, sys, threading
from concurrent.futures import ProcessPoolExecutor
DB = "data/bench/dev_20240627/dev_databases"

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
    qid, db, pred, g1, g2 = a
    p = run(db, pred)
    out = []
    for g in (g1, g2):
        gs = run(db, g)
        out.append(p is not None and gs is not None and p == gs)
    return qid, db, out

if __name__ == "__main__":
    old = {r["question_id"]: r for r in json.load(open("data/bench/dev_20240627/dev.json"))}
    new = {r["question_id"]: r for r in json.load(open("data/raw/bird_dev_20251106/data/dev_20251106-00000-of-00001.json"))}
    ans = {}
    for f in glob.glob(sys.argv[1] + "/*.jsonl"):
        for ln in open(f):
            try:
                r = json.loads(ln); ans[int(r["question_id"])] = r["sql"]
            except Exception:
                pass
    jobs = [(q, old[q]["db_id"], s, old[q]["SQL"], new[q]["SQL"]) for q, s in ans.items() if q in old]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(job, jobs, chunksize=4))
    n = len(res); ok_old = sum(r[2][0] for r in res); ok_new = sum(r[2][1] for r in res)
    diff = collections.Counter(old[q]["difficulty"] for q, _, _ in res)
    dok = collections.Counter(old[q]["difficulty"] for q, _, o in res if o[0])
    print(json.dumps({"answered": n, "EX_original_gold": round(100 * ok_old / n, 2) if n else None,
                      "EX_2025_revised_gold": round(100 * ok_new / n, 2) if n else None,
                      "by_difficulty_original": {k: f"{dok[k]}/{v} = {100*dok[k]/v:.1f}" for k, v in diff.items()}}, indent=1))
    json.dump({str(q): o for q, _, o in res}, open(sys.argv[1] + "/../scored.json", "w"))
