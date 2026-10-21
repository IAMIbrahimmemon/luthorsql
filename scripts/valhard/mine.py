"""VALHARD step 2 (Mac, CPU): hard pairs for validator training from the generator's OWN BIRD-train candidate lists.

  .venv/bin/python scripts/valhard/mine.py --base data/valhard/r1/base --pool r1=runs/phase1/p1w/p1w/r1/cands_train.jsonl \
      --out data/valhard/r1 [--workers 6] [--seed 0]

Input: the val2-style base lists built from the same pool (scripts/val2/build_lists.py --pool ... --out <base>:
relabelled with BIRD-Verified, one live representative per result class, train/val split by the 4 pinned DBs) and
the raw pool (every candidate, for the members of each result class). Only BIRD-train DBs: held-out (DECISION /
RESERVED: data/sys1/dev_dbs_bird.json) and contaminated DBs are asserted absent; gold-noise ids
(data/sys1/gold_noise_drop.json) get no hard pairs at all.

(a) mined: for every live wrong result class (in the list, or cut by the list's 12-class cap), the closest
    (right member, wrong member) pair by clause diff (sqldiff.classify; fewest differing components, hard types first,
    then the best generator ranks). Labels come from the base list (relabelled); a class cut by the cap gets the raw
    gen-time label unless the question's gold was corrected (then it is skipped).
(b) synthetic: edits.py's minimal edits of the right representative (or the best-ranked right member that round-trips
    through sqlglot), executed against the list's gold with the gen-time labelling (scripts/val2/execq.py); kept when
    it runs, is non-null, is not right and its result is new for the list (not a list class / mutant, not an earlier
    edit). Edits that come out right are counted, never used.
Writes <out>/hard.jsonl (one record per list: mined pairs + kept edits), <out>/mine_stats.json (counts per type),
<out>/pairs_sample.jsonl (a few examples per type, for reading).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts/val2"))
import build_lists as bl  # noqa: E402
import edits as ed  # noqa: E402
import execq  # noqa: E402
import sqldiff  # noqa: E402

DBROOT = ROOT / "data/raw/bird_train/train_databases"
MEMBERS = 8          # members per class considered for the closest pair (best generator ranks)
GEN_SECONDS = bl.GEN_SECONDS
GOLD_SECONDS = bl.GOLD_SECONDS


def class_members(union: list[dict]) -> dict[str, list[dict]]:
    """Live candidates of the raw pool by result signature, best rank first."""
    return {s: v[:MEMBERS] for s, v in bl.classes(union).items()}


def closest_pair(rights: list[dict], wrongs: list[dict], cache: dict) -> dict | None:
    """The (right, wrong) member pair with the smallest clause diff; hard types first, then fewer components, then
    the best summed generator rank. cache: sql -> sqldiff.feats."""
    best = None
    for r in rights:
        fr = cache.setdefault(r["sql"], sqldiff.feats(r["sql"]))
        for w in wrongs:
            fw = cache.setdefault(w["sql"], sqldiff.feats(w["sql"]))
            if fr is None or fw is None:
                kind, bucket, n = "unparsed", "other", 99
            else:
                kind, bucket, n = sqldiff.classify(r["sql"], w["sql"], fr, fw)
            key = (not sqldiff.is_hard(kind), n, r["rank"] + w["rank"], r["sql"], w["sql"])
            if best is None or key < best[0]:
                best = (key, {"type": kind, "bucket": bucket, "ncomp": n, "right_sql": r["sql"], "right_sig": r.get("sig"), "wrong_sql": w["sql"],
                              "right_rank": r["rank"], "wrong_rank": w["rank"]})
    return best[1] if best else None


def mine_list(rec: dict, union: list[dict], cache: dict) -> list[dict]:
    """(a) one closest pair per live wrong result class of this question."""
    lab = {c["sig"]: bool(c["ok"]) for c in rec["cands"] if not c.get("mutant") and c.get("sig")}
    members = class_members(union)
    out = []
    right_sigs = [s for s, ok in lab.items() if ok]
    rights = [m for s in right_sigs for m in members.get(s, [])]
    if not rights:  # the representative only (e.g. the class was rebuilt by relabelling)
        rights = [c for c in rec["cands"] if c.get("ok") and not c.get("mutant")]
    for sig, ms in sorted(members.items(), key=lambda kv: kv[1][0]["rank"]):
        if sig in lab:
            if lab[sig]:
                continue
            in_list = True
        else:
            if rec.get("gold_corrected"):
                continue  # gen-time label is against the old gold
            if ms[0].get("ok"):
                continue
            in_list = False
        p = closest_pair(rights, ms, cache)
        if p is None:
            continue
        p.update(wrong_sig=sig, in_list=in_list, wrong_res=ms[0].get("res"))
        out.append(p)
    return out


def edit_source(rec: dict, union: list[dict]) -> str | None:
    """The right representative if it round-trips through sqlglot, else the best-ranked right member that does."""
    right = [c for c in rec["cands"] if c.get("ok") and not c.get("mutant")]
    if not right:
        return None
    sigs = {c["sig"] for c in right}
    cands = [right[0]["sql"]] + [c["sql"] for c in sorted(union, key=lambda c: c["rank"])
                                 if c.get("sig") in sigs and bl.live(c) and c["sql"] != right[0]["sql"]]
    return next((s for s in cands if ed.parse_identity(s) is not None), None)


def keep_edits(trials: list[tuple[str, str, tuple]], taken: set) -> tuple[list[dict], collections.Counter]:
    """trials [(kind, sql, (runs, nonnull, ok, sig, res))] -> kept hard negatives + per-outcome counts."""
    out, st, taken = [], collections.Counter(), set(taken)
    for kind, sql, (runs, nonnull, ok, sig, res) in trials:
        if not runs:
            st[f"{kind}:error"] += 1
        elif not nonnull:
            st[f"{kind}:null"] += 1
        elif ok:
            st[f"{kind}:right"] += 1
        elif not sig or sig in taken:
            st[f"{kind}:same_result"] += 1
        else:
            taken.add(sig)
            st[f"{kind}:kept"] += 1
            out.append({"kind": kind, "bucket": ed.BUCKET_OF[kind], "sql": sql, "runs": True, "nonnull": True,
                        "ok": False, "sig": sig, "res": res})
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="dir with train_lists.jsonl + val_lists.jsonl (val2 build)")
    ap.add_argument("--pool", action="append", required=True, metavar="NAME=PATH", help="the raw pool(s) of the base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    held = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
    cont = set(json.load(open(ROOT / "data/sys1/contaminated_dbs.json"))["dbs"])
    noise = set(json.load(open(ROOT / "data/sys1/gold_noise_drop.json"))["ids"])
    pools = bl.load_pools(a.pool)
    recs = []
    for split in ("train", "val"):
        for line in open(a.base / f"{split}_lists.jsonl"):
            r = json.loads(line)
            assert r["db_id"] not in held and r["db_id"] not in cont, (r["id"], r["db_id"])
            assert r["id"].startswith("bird_train_"), r["id"]
            recs.append((split, r))
    paths = {p.stem: str(p) for p in DBROOT.glob("*/*.sqlite")}
    st = collections.Counter()
    mined_ct, mined_in, out_recs, jobs = collections.Counter(), collections.Counter(), {}, collections.defaultdict(list)
    schemas = {}
    for split, r in recs:
        i = r["id"]
        st[f"lists_{split}"] += 1
        if i in noise:
            st["skip_gold_noise_id"] += 1
            continue
        lists = [(n, [dict(c) for c in p[i]["cands"]]) for n, p in pools.items() if i in p]
        union = bl.union_cands(lists)
        mined = mine_list(r, union, {})  # feats cache per question (pairs never cross questions)
        for p in mined:
            mined_ct[p["type"]] += 1
            mined_in[(p["type"], p["in_list"])] += 1
        src = edit_source(r, union)
        rec = {"id": i, "db_id": r["db_id"], "split": split, "mined": mined, "edits": [], "edit_source": src}
        out_recs[i] = rec
        if src is None:
            st["no_editable_right_sql"] += 1
            continue
        db = r["db_id"]
        if db not in schemas:
            schemas[db] = ed.schema_of(paths[db])
        es = ed.edits(src, schemas[db], seed=f"{a.seed}:{i}")
        st["edits_generated"] += len(es)
        if es:
            rec["_trials"] = es
            jobs[db].append((i, r["gold_sql"], [s for _, s in es]))
    print(f"mined {sum(mined_ct.values())} pairs; executing {sum(len(s) for v in jobs.values() for _, _, s in v)} "
          f"edits on {len(jobs)} DBs", flush=True)
    res = execq.run_tasks(jobs, paths, GOLD_SECONDS, GEN_SECONDS, workers=a.workers)
    edit_ct = collections.Counter()
    by_id = {r["id"]: r for _, r in recs}
    for i, rec in out_recs.items():
        trials = rec.pop("_trials", None)
        if not trials:
            continue
        status, _, rr = res[i]
        if status != "ok":
            st[f"edit_gold_{status}"] += 1
            continue
        taken = {c["sig"] for c in by_id[i]["cands"] if c.get("sig")}
        kept, s2 = keep_edits([(k, s, x) for (k, s), x in zip(trials, rr)], taken)
        for e in kept:
            kind, bucket, n = sqldiff.classify(rec["edit_source"], e["sql"])
            e["diff_type"] = kind
            if sqldiff.is_hard(kind):  # e.g. add_join on a non-aggregate query acts as a filter (sem), not grain
                e["bucket"] = sqldiff.BUCKET[kind]
        rec["edits"] = kept
        edit_ct.update(s2)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "hard.jsonl", "w") as f:
        for i in sorted(out_recs, key=bl.qindex):
            f.write(json.dumps(out_recs[i]) + "\n")
    # samples for reading: up to 3 per mined type and per edit kind
    samp, seen = [], collections.Counter()
    rnd = random.Random(a.seed)
    allp = [(i, p) for i, r in out_recs.items() for p in r["mined"] if sqldiff.is_hard(p["type"])]
    rnd.shuffle(allp)
    for i, p in allp:
        if seen[p["type"]] < 3:
            seen[p["type"]] += 1
            samp.append({"src": "mined", "id": i, "q": by_id[i]["question"], **p})
    alle = [(i, e) for i, r in out_recs.items() for e in r["edits"]]
    rnd.shuffle(alle)
    for i, e in alle:
        if seen["edit:" + e["kind"]] < 3:
            seen["edit:" + e["kind"]] += 1
            samp.append({"src": "edit", "id": i, "q": by_id[i]["question"], "right_sql": out_recs[i]["edit_source"],
                         **e})
    with open(a.out / "pairs_sample.jsonl", "w") as f:
        for s in samp:
            f.write(json.dumps(s) + "\n")
    kinds = sorted({k.split(":")[0] for k in edit_ct})
    stats = {
        "seed": a.seed, "pools": a.pool, "base": str(a.base), "build": dict(st),
        "mined_pairs_by_type": dict(mined_ct.most_common()),
        "mined_pairs_by_bucket": dict(collections.Counter(sqldiff.BUCKET.get(k, "other") for k in mined_ct.elements())),
        "mined_hard_in_list": sum(v for (k, il), v in mined_in.items() if il and sqldiff.is_hard(k)),
        "mined_hard_cut_by_cap": sum(v for (k, il), v in mined_in.items() if not il and sqldiff.is_hard(k)),
        "lists_with_mined_hard": sum(any(sqldiff.is_hard(p["type"]) for p in r["mined"]) for r in out_recs.values()),
        "edits_by_kind": {k: {o: edit_ct[f"{k}:{o}"] for o in ("kept", "right", "same_result", "null", "error")}
                          for k in kinds},
        "edits_kept_by_bucket": dict(collections.Counter(e["bucket"] for r in out_recs.values() for e in r["edits"])),
        "lists_with_edit": sum(bool(r["edits"]) for r in out_recs.values()),
        "minutes": round((time.time() - t0) / 60, 1)}
    (a.out / "mine_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
