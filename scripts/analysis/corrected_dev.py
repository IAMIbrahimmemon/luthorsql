import json, sys, sqlite3, time
from multiprocessing import Pool
import os; R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
DEV = R + "data/bench/dev_20240627/dev_databases"
orig = {f"bird_dev_{d['question_id']}": d["SQL"] for d in json.load(open(R + "data/bench/dev_20240627/dev.json"))}
corr = {json.loads(l)["id"]: json.loads(l) for l in open(R + "data/eval/dev6c_bird.jsonl")}
def run(t):
    db, golds, sqls = t
    con = sqlite3.connect(f"file:{DEV}/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    def rows(s):
        if not s: return None
        dl = time.time() + 30
        con.set_progress_handler(lambda: 1 if time.time() > dl else 0, 20_000)
        try: return frozenset(map(tuple, con.execute(s).fetchall()))
        except Exception: return None
    G = [rows(g) for g in golds]; S = [rows(s) for s in sqls]
    return [[g is not None and s is not None and s == g for s in S] for g in G]
def main(scored, stack, name):
    B = {json.loads(l)["id"]: json.loads(l) for l in open(scored)}
    T = {json.loads(l)["id"]: json.loads(l) for l in open(stack)}
    ids = sorted(B); tasks = []
    for i in ids:
        cs = B[i]["cands"]
        sq = [cs[0]["sql"] if cs else None, T[i]["res"]["V:-"][1], T[i]["res"]["V:VAL+DROP"][1]] + [c["sql"] for c in cs]
        tasks.append((B[i]["db_id"], [orig[i], corr[i]["gold_sql"]], sq))
    with Pool(8) as p: res = p.map(run, tasks, chunksize=4)
    ch = lambda i, k: (corr[i]["changed"].get(k) if isinstance(corr[i]["changed"], dict) else f"'{k}': True" in str(corr[i]["changed"]))
    chq = {i for i in ids if ch(i, "question") or ch(i, "evidence")}
    for gi, gname in ((0, "original"), (1, "corrected")):
        f = sum(r[gi][0] for r in res); v = sum(r[gi][1] for r in res); d = sum(r[gi][2] for r in res); o = sum(any(r[gi][3:]) for r in res)
        print(f"{name:3s} {gname:9s} first {100*f/1534:5.2f}  validator {100*v/1534:5.2f}  +repair {100*d/1534:5.2f}  oracle@8 {100*o/1534:5.2f}")
    print(f"    questions/evidence rewritten in corrected dev: {len(chq)}")
if __name__ == "__main__":
    main(*sys.argv[1:4])
