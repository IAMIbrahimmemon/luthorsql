"""VALHARD step 3 (Mac, CPU, no DB): mix the mined / synthetic hard negatives into the val2-style base lists.

  .venv/bin/python scripts/valhard/mix.py --base data/valhard/r1/base --out data/valhard/r1 --hard data/valhard/r1/hard.jsonl

Without --hard (the default) the output is the base lists unchanged except for --drop-mutations; every hard-negative
change is behind --hard. Per list (train and val alike):
  - base near-miss mutants whose mutation is in --drop-mutations (default: limit; LIMIT/ties are a 50/50 gold
    convention on DECISION) are removed;
  - mined pairs are taken in mined_priority order (grain, sem, proj, proj_other; fewer components; best ranks);
  - mined (a), cut by the cap: up to --max-mined-add wrong classes that the 12-class cap removed but whose closest pair
    to a right member is a hard type are added back (their closest wrong member, prov "mined");
  - mined (a), in the list (--pair-members, default on): for up to --max-pair-members hard pairs, the closest wrong
    member and the closest right member are added as extra members of their result classes when their SQL differs from
    the class's representative (prov "pair"), so the list holds the minimal contrast itself; use --loss group in the
    trainer so a class's extra members add no extra mass;
  - synthetic (b): up to --max-edits kept edits, one per bucket in turn (grain, sem, proj), kinds in edits.KINDS order
    (prov "edit", mutant: true, rank 200+, cost 99.0; the validator never sees rank or cost).
Candidates keep the trainer's fields plus "prov" (base | mut | mined | pair | edit) and "hard_type" (analysis only).
Writes <out>/{train_lists,val_lists}.jsonl and <out>/mix_stats.json (entries by prov and type).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ed  # noqa: E402
import sqldiff  # noqa: E402

EDIT_RANK = 200
MINED_RANK = 300


PRIORITY = {"grain": 0, "sem": 1, "proj": 2}


def mined_priority(p: dict) -> tuple:
    """grain before sem before proj (the pick-gap's largest clean bucket first), the vague proj_other last; then fewer
    differing components, then the best generator ranks."""
    return (3 if p["type"] == "proj_other" else PRIORITY[sqldiff.BUCKET[p["type"]]], p["ncomp"],
            p["right_rank"] + p["wrong_rank"], p["wrong_sql"])


def pick_edits(edits: list[dict], max_n: int) -> list[dict]:
    """Round robin over the buckets (grain, sem, proj), kinds in edits.KINDS order inside a bucket."""
    by = collections.defaultdict(list)
    for e in sorted(edits, key=lambda e: ed.KINDS.index(e["kind"]) if e["kind"] in ed.KINDS else 99):
        by[e["bucket"]].append(e)
    out = []
    while len(out) < max_n and any(by.values()):
        for b in ("grain", "sem", "proj"):
            if by[b] and len(out) < max_n:
                out.append(by[b].pop(0))
    return out


def mix_one(rec: dict, hard: dict | None, a) -> tuple[dict, collections.Counter]:
    st = collections.Counter()
    drop = set(a.drop_mutations or [])
    cands = []
    for c in rec["cands"]:
        if c.get("mutant") and c.get("mutation") in drop:
            st[f"dropped_mutant_{c['mutation']}"] += 1
            continue
        cands.append(dict(c, prov="mut" if c.get("mutant") else "base"))
    out = dict(rec)
    if hard is not None and a.use_hard:
        sigs = {c["sig"] for c in cands if c.get("sig")}
        texts = {c["sql"] for c in cands}
        hard_mined = sorted((p for p in hard.get("mined", []) if sqldiff.is_hard(p["type"])), key=mined_priority)
        # (a) cap-cut hard classes back in
        added = 0
        for p in hard_mined:
            if added >= a.max_mined_add:
                break
            if p["in_list"] or p["wrong_sig"] in sigs or p["wrong_sql"] in texts:
                continue
            cands.append({"rank": MINED_RANK + added, "cost": 99.0, "sql": p["wrong_sql"], "runs": True,
                          "nonnull": True, "ok": False, "sig": p["wrong_sig"], "res": p.get("wrong_res"),
                          "prov": "mined", "hard_type": p["type"]})
            sigs.add(p["wrong_sig"])
            texts.add(p["wrong_sql"])
            added += 1
            st[f"mined_added:{p['type']}"] += 1
        # (a) minimal pairs inside the list as extra class members
        if a.pair_members:
            right_sig = next((c["sig"] for c in cands if c.get("ok") and c["prov"] == "base"), None)
            n = 0
            for p in hard_mined:
                if n >= a.max_pair_members:
                    break
                if not p["in_list"]:
                    continue
                got = False
                for sql, sig, ok in ((p["wrong_sql"], p["wrong_sig"], False), (p["right_sql"], p.get("right_sig") or right_sig, True)):
                    if sql in texts or sig is None:
                        continue
                    cands.append({"rank": MINED_RANK + 50 + n, "cost": 99.0, "sql": sql, "runs": True,
                                  "nonnull": True, "ok": ok, "sig": sig, "prov": "pair", "hard_type": p["type"]})
                    texts.add(sql)
                    got = True
                    st[f"pair_member_{'right' if ok else 'wrong'}:{p['type']}"] += 1
                n += got
        # (b) synthetic edits
        for j, e in enumerate(pick_edits([e for e in hard.get("edits", []) if e["sig"] not in sigs], a.max_edits)):
            cands.append({"rank": EDIT_RANK + j, "cost": 99.0, "sql": e["sql"], "runs": True, "nonnull": True,
                          "ok": False, "sig": e["sig"], "res": e.get("res"), "mutant": True, "mutation": e["kind"],
                          "prov": "edit", "hard_type": e["kind"]})
            sigs.add(e["sig"])
            st[f"edit:{e['kind']}"] += 1
    out["cands"] = cands
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hard", type=Path, help="mine.py's hard.jsonl; without it no hard negative is added")
    ap.add_argument("--max-edits", type=int, default=3)
    ap.add_argument("--max-mined-add", type=int, default=2)
    ap.add_argument("--max-pair-members", type=int, default=2, help="hard pairs per list whose members are added")
    ap.add_argument("--no-pair-members", dest="pair_members", action="store_false")
    ap.add_argument("--drop-mutations", nargs="*", default=["limit"])
    a = ap.parse_args()
    a.use_hard = a.hard is not None
    hard = {r["id"]: r for r in map(json.loads, open(a.hard))} if a.hard else {}
    a.out.mkdir(parents=True, exist_ok=True)
    stats = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    for split in ("train", "val"):
        st, rows = collections.Counter(), []
        for line in open(a.base / f"{split}_lists.jsonl"):
            r = json.loads(line)
            m, s2 = mix_one(r, hard.get(r["id"]), a)
            oks = [bool(c.get("ok")) for c in m["cands"]]
            if not any(oks) or all(oks):
                st["drop_not_right_and_wrong"] += 1
                continue
            st.update(s2)
            rows.append(m)
        with open(a.out / f"{split}_lists.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = max(len(rows), 1)
        prov = collections.Counter((c["prov"], bool(c.get("ok"))) for r in rows for c in r["cands"])
        stats[split] = {"lists": len(rows), "dbs": len({r["db_id"] for r in rows}),
                        "mean_entries": round(sum(len(r["cands"]) for r in rows) / n, 2),
                        "max_entries": max((len(r["cands"]) for r in rows), default=0),
                        "lists_with_hard": sum(any(c["prov"] in ("mined", "pair", "edit") for c in r["cands"]) for r in rows),
                        "weight3_share": round(sum(r.get("weight", 1.0) == 3.0 for r in rows) / n, 4),
                        "entries_by_prov": {f"{p}_{'right' if ok else 'wrong'}": v for (p, ok), v in sorted(prov.items())},
                        "changes": dict(sorted(st.items()))}
    (a.out / "mix_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
