"""Mac side, CPU: stage data/ff2 for the phase-2 box chain (scripts/chains/ff2_box.sh; plan
docs/paper_notes/research/freeform_plan_2026-10-10.md §2, §3.3-2/3, S3). BIRD-train training DBs + the phase-1 DECISION
pools only: no RESERVED record and no BIRD-dev file is written to data/ff2 (the held-out union that val2 scored, which
holds RESERVED rows, is pushed from data/val2 as before and only scored blind on the box).

  uv run python scripts/freeform/ff2_stage.py [--n-train-list 2250] [--n-train-uncovered 300]

1. base   data/ff2_base/{questions,grammar_lists_all}.jsonl from `build_lists.py --stage-ff --out data/ff2_base`
          (run here when missing; it reproduces data/val2/{train,val}_lists.jsonl exactly).
2. subset the questions that get free-form samples on the box (deterministic: md5("ff2:<seed>:<id>") order):
          every val-DB question (the val2 val split: list, noright and uncovered), every train question whose grammar
          list has no live right class (noright: ff rescue), --n-train-uncovered grammar-uncovered train questions and
          --n-train-list listed train questions. The box trains only on these (mixed) lists, so the subset sets both
          the generation and the training time (estimate printed).
   -> data/ff2/gen_questions.jsonl (pool schema with empty cands, for gen.py --pool; + kind, split, gold_sql_orig)
      data/ff2/grammar_lists.jsonl (grammar_lists_all rows of the subset)
3. arithmetic mutants of grammar SQL (data/pool/train_r{1,2}_arith.jsonl, union by SQL) for the subset, relabelled
   here against the list gold (BIRD-Verified corrections included) with merge.label (the gen-time rule, no
   normalisation) -> data/ff2/arith_lab.jsonl
4. DECISION pools of phase 1 (runs/phase1/ff1/ff1/pools: superset + m_both_mut, m_both, m_omni, m_qwen, m_mut), rr
   removed, DECISION ids only -> data/ff2/dec/
5. data/ff2/dbs.txt: the DBs the box needs for labelling (pushed from data/raw/bird_train/train_databases).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/freeform"))
import merge  # noqa: E402

BASE = ROOT / "data/ff2_base"
OUT = ROOT / "data/ff2"
DBROOT = ROOT / "data/raw/bird_train/train_databases"
HV2 = ROOT / "data/sys1/heldout_v2"
FF1_POOLS = ROOT / "runs/phase1/ff1/ff1/pools"
DEC_POOLS = ("superset", "m_both_mut", "m_both", "m_omni", "m_qwen", "m_mut")
# per-question costs measured on phase 1 / val2 (H100): OmniSQL ~850 samples/min, Qwen ~1,400 (ff1 logs, n=16+greedy);
# val2 training 1.81 h for 3,971 lists x 10.7 entries (2 epochs)
OMNI_PER_MIN, QWEN_PER_MIN, TRAIN_MIN_PER_ENTRY = 850.0, 1400.0, 1.81 * 60 / (3971 * 10.65)


def h(seed: int, i: str) -> str:
    return hashlib.md5(f"ff2:{seed}:{i}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-list", type=int, default=2250)
    ap.add_argument("--n-train-uncovered", type=int, default=300)
    ap.add_argument("--n", type=int, default=8, help="samples per question and generator (box)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if not (BASE / "questions.jsonl").exists() or not (BASE / "grammar_lists_all.jsonl").exists():
        print("building data/ff2_base (build_lists.py --stage-ff) ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/val2/build_lists.py"), "--stage-ff", "--out", str(BASE),
                        "--workers", str(a.workers)], check=True)
    a.out.mkdir(parents=True, exist_ok=True)
    qs = [json.loads(l) for l in open(BASE / "questions.jsonl")]
    gl = {r["id"]: r for r in map(json.loads, open(BASE / "grammar_lists_all.jsonl"))}
    held = set(json.loads((ROOT / "data/sys1/dev_dbs_bird.json").read_text()))
    hv = set(json.loads((HV2 / "decision.json").read_text())["ids"]) | set(json.loads((HV2 / "reserved.json").read_text())["ids"])
    assert not {q["db_id"] for q in qs} & held, "held-out DB in the training questions"
    assert not {q["id"] for q in qs} & hv, "DECISION / RESERVED id in the training questions"

    # ---- 2 subset ---------------------------------------------------------------------------------------------------
    by = collections.defaultdict(list)
    for q in qs:
        by[(q["split"], q["kind"])].append(q)
    for v in by.values():
        v.sort(key=lambda q: h(a.seed, q["id"]))
    pick = [q for (sp, _), v in by.items() if sp == "val" for q in v]
    pick += by[("train", "noright")] + by[("train", "nowrong")]
    pick += by[("train", "uncovered")][:a.n_train_uncovered] + by[("train", "list")][:a.n_train_list]
    pick.sort(key=lambda q: int(q["id"].rsplit("_", 1)[1]))
    ids = {q["id"] for q in pick}
    comp = collections.Counter((q["split"], q["kind"]) for q in pick)
    with open(a.out / "gen_questions.jsonl", "w") as f:
        for q in pick:
            f.write(json.dumps({**q, "cands": []}) + "\n")
    with open(a.out / "grammar_lists.jsonl", "w") as f:
        for i in sorted(ids & set(gl), key=lambda i: int(i.rsplit("_", 1)[1])):
            f.write(json.dumps(gl[i]) + "\n")
    dbs = sorted({q["db_id"] for q in pick})
    (a.out / "dbs.txt").write_text("\n".join(dbs) + "\n")
    mb = sum((DBROOT / d / f"{d}.sqlite").stat().st_size for d in dbs) / 1e6
    print(f"subset: {len(pick)} questions on {len(dbs)} DBs ({mb / 1e3:.2f} GB) "
          f"{json.dumps({f'{s}/{k}': v for (s, k), v in sorted(comp.items())})}", flush=True)

    # ---- 3 arithmetic mutants, relabelled against the list gold ---------------------------------------------------
    qmap = {q["id"]: q for q in pick}
    ar: dict[str, dict] = {}
    for role in ("r1", "r2"):
        p = ROOT / f"data/pool/train_{role}_arith.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            if r["id"] not in qmap or r["id"] not in gl:
                continue
            d = ar.setdefault(r["id"], {"id": r["id"], "db_id": qmap[r["id"]]["db_id"],
                                        "gold_sql": qmap[r["id"]]["gold_sql"], "cands": [], "_seen": set()})
            for c in r["cands"]:
                if c.get("sql") and c["sql"] not in d["_seen"]:
                    d["_seen"].add(c["sql"])
                    d["cands"].append({k: v for k, v in c.items() if k in ("rank", "sql", "mut", "parent")})
    n_ar = 0
    if ar:
        recs = [{k: v for k, v in r.items() if k != "_seen"} for r in ar.values()]
        paths = {p.stem: str(p) for p in DBROOT.glob("*/*.sqlite")}
        lab = merge.label(recs, paths, argparse.Namespace(normalize=False, no_gold=False, gold_seconds=30.0,
                                                          cand_seconds=3.0, workers=a.workers))
        with open(a.out / "arith_lab.jsonl", "w") as f:
            for r in recs:
                st, cl = lab[r["id"]]
                if st != "ok":
                    continue
                f.write(json.dumps({"id": r["id"], "db_id": r["db_id"], "ff": {"gold_status": st}, "cands": cl}) + "\n")
                n_ar += 1
        n_right = sum(c.get("ok") for l in open(a.out / "arith_lab.jsonl") for c in json.loads(l)["cands"])
        print(f"arith: {n_ar} questions, {sum(len(r['cands']) for r in recs)} mutants, {n_right} right after relabel",
              flush=True)
    else:
        (a.out / "arith_lab.jsonl").unlink(missing_ok=True)
        print("arith: no mutant files (skipped)", flush=True)

    # ---- 4 DECISION pools of phase 1 ----------------------------------------------------------------------------
    dec = set(json.loads((HV2 / "decision.json").read_text())["ids"])
    res = set(json.loads((HV2 / "reserved.json").read_text())["ids"])
    (a.out / "dec").mkdir(exist_ok=True)
    for name in DEC_POOLS:
        rows = []
        for r in map(json.loads, open(FF1_POOLS / f"{name}.jsonl")):
            assert r["id"] in dec and r["id"] not in res, f"{name}: non-DECISION id {r['id']}"
            r["cands"] = [{k: v for k, v in c.items() if k != "rr"} for c in r["cands"]]
            rows.append(json.dumps(r) + "\n")
        (a.out / "dec" / f"{name}.jsonl").write_text("".join(rows))
    print(f"dec: {len(DEC_POOLS)} DECISION pools copied ({len(rows)} records each)", flush=True)

    # ---- estimate ---------------------------------------------------------------------------------------------------
    n = len(pick)
    gen_min = n * a.n * (1 / OMNI_PER_MIN + 1 / QWEN_PER_MIN) + 4
    entries = sum(len(gl[i]["cands"]) + 3 if i in gl else 6 for i in ids)  # ~+3 ff / arith entries; ff-only ~6
    train_min = entries * TRAIN_MIN_PER_ENTRY
    est = {"questions": n, "dbs": len(dbs), "db_gb": round(mb / 1e3, 2), "arith_questions": n_ar,
           "composition": {f"{s}/{k}": v for (s, k), v in sorted(comp.items())},
           "est_gen_min": round(gen_min), "est_list_entries": entries, "est_train_min_2ep": round(train_min),
           "est_fixed_min": 45, "est_total_h": round((gen_min + train_min + 45) / 60, 2)}
    (a.out / "stage_stats.json").write_text(json.dumps(est, indent=1))
    print("STAGE " + json.dumps(est), flush=True)


if __name__ == "__main__":
    main()
