"""System 1 proposes, Opus selects or writes its own (DECISION). System 1 = R1 generator k=32 (live candidates grouped
by result); Opus sees the groups (SQL, result preview, n_of_32) and picks one or writes its own SQL ("NONE").
Questions with no live System-1 candidate use Opus's standalone answer (opus_dec). Strict EX vs original gold.

  python3 scripts/analysis/opus_select_dec.py <scratchpad>
"""
import glob, json, sys
sys.path.insert(0, "scripts/analysis")
from opus_router_dec import job  # noqa: E402
from concurrent.futures import ProcessPoolExecutor
SP = sys.argv[1]
L = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
if __name__ == "__main__":
    ids = json.load(open("data/sys1/heldout_v2/decision.json"))["ids"]; n = len(ids)
    tr = json.load(open("data/raw/bird_train/train.json"))
    alone = json.load(open(f"{SP}/opus_dec/scored.json"))
    lab, orc, s1 = {}, {}, {}
    for ln in open("runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"):
        r = json.loads(ln)
        c = [x for x in r["cands"] if x.get("src") in ("r1", "both") and x.get("runs") and x.get("nonnull")]
        g = {}
        for x in c:
            k = x.get("sig") or x["sql"]
            if k not in g or x["rr"] > g[k]["rr"]:
                g[k] = x
        grp = sorted(g.values(), key=lambda x: -x["rr"])
        if grp:
            lab[r["id"]] = {L[k]: bool(x["ok"]) for k, x in enumerate(grp)}
            orc[r["id"]] = any(lab[r["id"]].values()); s1[r["id"]] = bool(grp[0]["ok"])
    ans = {}
    for f in glob.glob(f"{SP}/opus_sel/answers/*.jsonl"):
        for ln in open(f):
            try:
                r = json.loads(ln); ans[str(r["question_id"])] = r
            except Exception:
                pass
    own = [(q, tr[int(q.rsplit("_", 1)[1])]["db_id"], a["sql"], tr[int(q.rsplit("_", 1)[1])]["SQL"])
           for q, a in ans.items() if str(a.get("pick")).upper() == "NONE" and a.get("sql")]
    with ProcessPoolExecutor(6) as ex:
        ownok = dict(ex.map(job, own, chunksize=4))
    res, st = {}, {"picked": 0, "own": 0, "pick_right": 0, "own_right": 0, "pick_when_right_exists": 0,
                   "right_exists": 0, "own_when_right_exists": 0, "missing": 0}
    for q in ids:
        if q not in lab:
            res[q] = alone[q]; continue
        a = ans.get(q)
        if a is None:
            st["missing"] += 1; res[q] = s1[q]; continue
        st["right_exists"] += orc[q]
        if str(a.get("pick")).upper() == "NONE":
            st["own"] += 1; res[q] = ownok.get(q, False); st["own_right"] += res[q]; st["own_when_right_exists"] += orc[q]
        else:
            st["picked"] += 1; res[q] = lab[q].get(a["pick"], False); st["pick_right"] += res[q]
            st["pick_when_right_exists"] += orc[q] and res[q]
    tot = sum(res.values())
    print(json.dumps(st))
    print(f"System 1 alone {100*sum(s1.get(q,False) for q in ids)/n:.2f} | Opus alone {100*sum(alone[q] for q in ids)/n:.2f}"
          f" | S1 proposes + Opus selects/writes {100*tot/n:.2f}  (answered {len(ans)} of {len(lab)} covered)")
    rx = [q for q in ids if q in lab and q in ans and orc[q]]
    print(f"when a right candidate exists ({len(rx)}): combo right {100*sum(res[q] for q in rx)/len(rx):.1f}%, Opus alone right {100*sum(alone[q] for q in rx)/len(rx):.1f}%")
    nx = [q for q in ids if q in lab and q in ans and not orc[q]]
    print(f"when no right candidate ({len(nx)}): combo right {100*sum(res[q] for q in nx)/len(nx):.1f}%, Opus alone right {100*sum(alone[q] for q in nx)/len(nx):.1f}%")
    json.dump(res, open(f"{SP}/opus_sel/scored.json", "w"))
