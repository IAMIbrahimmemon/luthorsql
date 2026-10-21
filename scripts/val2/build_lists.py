"""val2 (Recipe A+) validator training lists from the BIRD-train wide pools (box W, R1 and R2, k=32).

  uv run python scripts/val2/build_lists.py [--workers 6] [--seed 0]

Steps (docs/paper_notes/research/selection_memo_2026-10-10.md, section 3 Step 1; scripts/val2/README.md):
  1 relabel  BIRD-Verified gold (data/raw/bird_verified, question_id = index into data/raw/bird_train/train.json)
             replaces the gold, question and evidence; the corrected gold and every R1 ∪ R2 candidate of a
             question whose SQL changed are re-executed (scripts/val2/execq.py = gen-time labelling). Every gold
             is executed; a question whose gold errors, times out, returns no rows or only NULLs is dropped.
             Dropped too: held-out DBs (data/sys1/dev_dbs_bird.json), contaminated DBs, gold_noise_drop ids
             without a BIRD-Verified entry, and corrected questions graded other than as a set.
  2 lists    R1 ∪ R2 by SQL text; dead candidates (not runs or not nonnull) dropped; one representative per result
             signature (even question index: best-ranked SQL of the class; odd: a seeded random one); at most 12
             classes (every right class, then the best-ranked wrong classes); up to 2 near-miss mutants of the
             right representative (scripts/val2/mutate.py), kept only when they run, are non-null and their sig
             differs from every class in the list; weight 3.0 when a run's greedy (rank 0) is wrong, else 1.0;
             only lists with >= 1 right and >= 1 wrong class.
  3 split    questions on the 4 pinned validation DBs -> data/val2/val_lists.jsonl, the rest -> train_lists.jsonl.
SQL text is never normalised. Writes data/val2/{train_lists,val_lists}.jsonl, relabel_stats.json, build_stats.json.

Phase 2 (free-form, docs/paper_notes/research/freeform_plan_2026-10-10.md §3.3-2). Both modes are off by default, so
the call above is unchanged:
  --stage-ff --out DIR   (Mac, needs the DBs) the normal build into DIR, plus DIR/grammar_lists_all.jsonl (every built
             list BEFORE the right/wrong filters, near-miss mutants included, so lists with no live right class are
             kept for free-form rescue) and DIR/questions.jsonl (every relabelled question: the listed ones and the
             grammar-uncovered BIRD-train questions on the same DBs, gold executed; kind = list | noright | nowrong |
             uncovered; split = train | val by the pinned val DBs; gold_sql_orig = the original BIRD-train gold).
  --ff-extend --grammar-lists G --questions Q --ff-qwen LQ --ff-omni LO [--arith LA] --out DIR
             (CPU, no DB: labels come from merge.py --samples) mixed lists for the questions in Q: up to FF_MAX free-form
             result classes per grammar list (right and wrong; new sigs are added, sigs the grammar list already has
             get the ff text instead of the grammar text with probability SWAP_P, so ff style is not a source tell),
             ff-only lists (up to FF_ONLY_MAX classes) for questions without a grammar list, up to ARITH_MAX arithmetic
             mutant classes (right first); real classes <= CAP (the worst-ranked grammar wrong classes go first) plus
             the <= MAX_MUT near-miss mutants: <= 14 per list. Hygiene: ff samples whose canonical text equals the
             original or corrected gold are dropped; right classes found only by OmniSQL are dropped (seeded) until
             its per-source right rate matches Qwen's (the clean proxy). Candidates keep only the trainer's fields
             plus "prov" (g | ff | arith | mut, analysis only; rerank_sq2 never reads it, nor src / ff_n / ff_sig_n,
             which are removed). Writes DIR/{train_lists,val_lists}.jsonl and DIR/build_stats.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import execq  # noqa: E402
import mutate  # noqa: E402

POOLS = ROOT / "runs/phase1/p1w/p1w"
OUT = ROOT / "data/val2"
DBROOT = ROOT / "data/raw/bird_train/train_databases"
VAL_DBS = ("citeseer", "disney", "professional_basketball", "trains")
CAP = 12
MAX_MUT = 2
MUT_RANK = 100     # mutants are listed after every generated candidate (k <= 32)
MUT_COST = 99.0
GEN_SECONDS = 3.0  # rerank_sq2.py gen default (box W did not override it)
GOLD_SECONDS = 10.0


# ---- pure list logic (tests/test_val2_build.py) ----------------------------------------------------------------

def qindex(qid: str) -> int:
    return int(qid.rsplit("_", 1)[1])


def union_cands(lists: list[tuple[str, list[dict]]]) -> list[dict]:
    """[(run name, cands)] -> one candidate per SQL text; rank/cost = its best rank over the runs, src = the run
    name or 'both'. Sorted by (rank, both before single runs, run order, sql). Candidates without SQL are dropped."""
    seen: dict[str, dict] = {}
    order = {name: k for k, (name, _) in enumerate(lists)}
    for name, cands in lists:
        for c in cands:
            s = c.get("sql")
            if not s:
                continue
            if s not in seen:
                seen[s] = dict(c, src=name)
            else:
                d = seen[s]
                if d["src"] != name:
                    d["src"] = "both"
                if c["rank"] < d["rank"]:
                    d["rank"], d["cost"] = c["rank"], c["cost"]
    return sorted(seen.values(), key=lambda c: (c["rank"], order.get(c["src"], -1), c["sql"]))


def live(c: dict) -> bool:
    return bool(c.get("runs")) and bool(c.get("nonnull"))


def classes(cands: list[dict]) -> dict[str, list[dict]]:
    """Live candidates grouped by result signature, each class sorted by (rank, sql)."""
    by = collections.defaultdict(list)
    for c in cands:
        if live(c) and c.get("sig"):
            by[c["sig"]].append(c)
    return {s: sorted(v, key=lambda c: (c["rank"], c["sql"])) for s, v in by.items()}


def representative(cls: list[dict], qid: str, sig: str, seed: int) -> dict:
    """Even question index: the best-ranked SQL of the class; odd: a seeded random member."""
    if qindex(qid) % 2 == 0 or len(cls) == 1:
        return cls[0]
    return random.Random(f"val2:{seed}:{qid}:{sig}").choice(cls)


def cap_classes(reps: list[dict], cap: int = CAP) -> list[dict]:
    """reps: one candidate per class, each tagged with _best (the class's best rank). All right classes, then the
    best-ranked wrong classes up to cap; returned in (best rank) order."""
    right = [r for r in reps if r["ok"]]
    wrong = sorted((r for r in reps if not r["ok"]), key=lambda r: (r["_best"], r["sql"]))
    keep = right + wrong[:max(cap - len(right), 0)]
    return sorted(keep, key=lambda r: (r["_best"], r["sql"]))


def greedy_wrong(lists: list[tuple[str, list[dict]]]) -> bool:
    """True when the rank-0 candidate of any run's list is not right (dead or missing counts as wrong)."""
    for _, cands in lists:
        g = next((c for c in cands if c.get("rank") == 0), None)
        if g is None or not (g.get("ok") and live(g)):
            return True
    return False


def pick_mutants(trials: list[tuple[str, str, tuple]], taken_sigs: set, max_n: int = MAX_MUT) -> list[dict]:
    """trials: [(kind, sql, (runs, nonnull, ok, sig, res))] in preference order -> up to max_n mutant candidates
    that run, are non-null, are not right, and whose sig is new (not a list class, not an earlier mutant)."""
    out, taken = [], set(taken_sigs)
    for kind, sql, (runs, nonnull, ok, sig, res) in trials:
        if len(out) >= max_n:
            break
        if not (runs and nonnull) or ok or not sig or sig in taken:
            continue
        taken.add(sig)
        out.append({"rank": MUT_RANK + len(out), "cost": MUT_COST, "sql": sql, "runs": True, "nonnull": True,
                    "ok": False, "sig": sig, "res": res, "mutant": True, "mutation": kind})
    return out


def build_list(rec: dict, lists: list[tuple[str, list[dict]]], seed: int) -> tuple[dict | None, dict]:
    """Union -> live classes -> representatives -> cap. Returns (record without mutants, info). The record keeps
    cands_train.jsonl's fields; candidates carry their usual keys plus src."""
    u = union_cands(lists)
    cl = classes(u)
    reps = []
    for sig, members in cl.items():
        r = dict(representative(members, rec["id"], sig, seed))
        r["_best"] = members[0]["rank"]
        reps.append(r)
    kept = cap_classes(reps)
    info = {"union": len(u), "live": sum(map(live, u)), "classes": len(cl), "kept": len(kept),
            "right_members": [m for s, ms in cl.items() for m in ms if m["ok"]]}
    for r in kept:
        r.pop("_best", None)
    out = {k: rec[k] for k in ("id", "db_id", "question", "context", "gold_sql", "in_slots", "covered") if k in rec}
    out["cands"] = kept
    out["weight"] = 3.0 if greedy_wrong(lists) else 1.0
    return out, info


# ---- phase 2: mixed grammar + free-form lists (--ff-extend; tests/test_ff2_lists.py) ---------------------------------

FF_MAX = 4          # ff-represented classes (new sigs or swapped texts) per grammar list
FF_ONLY_MAX = 8     # classes in an ff-only list (no grammar list: grammar-uncovered question)
ARITH_MAX = 1       # arithmetic-mutant classes per grammar list
SWAP_P = 0.5        # a sig both the grammar list and ff have: show the ff text with this probability
KEEP = ("rank", "cost", "sql", "runs", "nonnull", "ok", "sig", "res", "mutant", "mutation")


def unit(*parts) -> float:
    """Stable uniform [0, 1) from the parts (no process-dependent hashing)."""
    return int(hashlib.md5(":".join(map(str, parts)).encode()).hexdigest()[:12], 16) / float(16 ** 12)


def clean_cand(c: dict, prov: str, **kw) -> dict:
    """Only the trainer's fields (+ prov, never read by rerank_sq2): no src / ff_n / ff_sig_n / gen / sql_raw."""
    d = {k: c[k] for k in KEEP if k in c}
    d.update(kw)
    d["prov"] = prov
    return d


def ff_classes(samples: dict[str, list[dict]], gold_canons: set[str], canon) -> tuple[dict, collections.Counter]:
    """samples: {source: labelled ff samples} in preference order (the clean proxy first). Drops samples without SQL,
    dead ones and gold copies (canon(raw text) in gold_canons). -> ({sig: {ok, support, srcs, src, rep}}, stats);
    src = the first source that produced the sig, rep = its most frequent text there (ties -> lowest rank)."""
    st = collections.Counter()
    by: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, cl in samples.items():
        for c in cl:
            if not c.get("sql"):
                st["no_sql"] += 1
            elif not (live(c) and c.get("sig")):
                st["dead"] += 1
            elif gold_canons and canon(c.get("sql_raw") or c["sql"]) in gold_canons:
                st[f"gold_copy_{s}"] += 1
            else:
                by[c["sig"]][s].append(c)
                st[f"kept_{s}"] += 1
    out = {}
    for sig, per in by.items():
        src = next(s for s in samples if per.get(s))
        mem = per[src]
        texts = collections.Counter(m["sql"] for m in mem)
        top = max(texts.values())
        rep = min((m for m in mem if texts[m["sql"]] == top), key=lambda m: m["rank"])
        out[sig] = {"ok": bool(rep.get("ok")), "support": sum(len(v) for v in per.values()),
                    "srcs": sorted(k for k, v in per.items() if v), "src": src, "rep": rep}
    return out, st


def cap_source_rate(classes_by_q: dict[str, dict], clean: str, capped: str, seed: int) -> dict:
    """Right rate per source = right classes / classes that source produced. When `capped` (OmniSQL) is above `clean`
    (Qwen), drop the d right classes produced ONLY by `capped` with the smallest seeded hash, d = the smallest count
    that brings its rate to <= clean's. Mutates classes_by_q; returns before/after stats."""
    def rate(s):
        n = r = 0
        for cl in classes_by_q.values():
            for k in cl.values():
                if s in k["srcs"]:
                    n += 1
                    r += k["ok"]
        return r, n
    rq, nq = rate(clean)
    ro, no = rate(capped)
    target = rq / nq if nq else 0.0
    only = sorted(((q, s) for q, cl in classes_by_q.items() for s, k in cl.items() if k["ok"] and k["srcs"] == [capped]),
                  key=lambda x: unit(seed, "cap", *x))
    d = 0
    if no and ro / no > target and target < 1.0:
        d = min(len(only), math.ceil((ro - target * no) / (1.0 - target) - 1e-9))
    for q, s in only[:d]:
        del classes_by_q[q][s]
    ro2, no2 = rate(capped)
    return {"clean": clean, "capped": capped, "clean_right": rq, "clean_classes": nq,
            "clean_rate": round(target, 4), "capped_right_before": ro, "capped_classes_before": no,
            "capped_rate_before": round(ro / no, 4) if no else None, "capped_only_right": len(only), "dropped": d,
            "capped_rate_after": round(ro2 / no2, 4) if no2 else None}


def ensure_mix(ranked: list, picked: list) -> list:
    """picked = a prefix of ranked [(sig, class)]: make it hold a right and a wrong class when ranked has them
    (swapping out the last picked entries that are not the only one of their kind)."""
    picked = list(picked)
    for want in (True, False):
        if picked and not any(k["ok"] == want for _, k in picked):
            alt = next(((s, k) for s, k in ranked if k["ok"] == want and (s, k) not in picked), None)
            if alt is None:
                continue
            j = max(j for j, (_, k) in enumerate(picked)
                    if sum(x["ok"] == k["ok"] for _, x in picked) > 1 or len(picked) == 1)
            picked[j] = alt
    return picked


def mix_list(q: dict, glist: dict | None, ffc: dict, arith: list[dict], seed: int) -> tuple[dict, collections.Counter]:
    """One mixed list. q: the question record; glist: its grammar list (grammar_lists_all) or None; ffc: ff_classes();
    arith: labelled arithmetic mutants. The caller drops lists without both a right and a wrong entry."""
    st = collections.Counter()
    fields = ("id", "db_id", "question", "context", "gold_sql", "in_slots", "covered", "gold_corrected")
    if glist is None:
        ranked = sorted(ffc.items(), key=lambda kv: (-kv[1]["support"], kv[0]))
        pick = ensure_mix(ranked, ranked[:FF_ONLY_MAX])
        out = {k: q.get(k) for k in fields}
        out.update(in_slots=False, covered=False)
        out["cands"] = [clean_cand(k["rep"], "ff", rank=1000 + j) for j, (_, k) in enumerate(pick)]
        out["weight"] = 1.0
        st["ff_new"] += len(pick)
        return out, st
    out = {k: glist.get(k, q.get(k)) for k in fields}
    out["weight"] = glist.get("weight", 1.0)
    cands = [clean_cand(c, "mut" if c.get("mutant") else "g") for c in glist["cands"]]
    real_sigs = {c["sig"] for c in cands if not c.get("mutant")}
    all_sigs = {c["sig"] for c in cands}
    rank_sig = lambda kv: (-kv[1]["support"], kv[0])  # noqa: E731
    new = sorted(((s, k) for s, k in ffc.items() if s not in all_sigs), key=rank_sig)
    shared = sorted(((s, k) for s, k in ffc.items() if s in real_sigs and unit(seed, "swap", q["id"], s) < SWAP_P),
                    key=rank_sig)
    order = ([("new", s, k) for s, k in new if k["ok"]] + [("swap", s, k) for s, k in shared if k["ok"]]
             + [("new", s, k) for s, k in new if not k["ok"]] + [("swap", s, k) for s, k in shared if not k["ok"]])
    for kind, s, k in order[:FF_MAX]:
        if kind == "swap":
            j = next(j for j, c in enumerate(cands) if c["sig"] == s and not c.get("mutant"))
            cands[j] = clean_cand(k["rep"], "ff", rank=cands[j]["rank"])
        else:
            cands.append(clean_cand(k["rep"], "ff", rank=1000 + k["rep"]["rank"]))
            all_sigs.add(s)
        st[f"ff_{kind}_{'right' if k['ok'] else 'wrong'}"] += 1
    ar = collections.defaultdict(list)
    for c in arith:
        if c.get("sql") and live(c) and c.get("sig") and c["sig"] not in all_sigs:
            ar[c["sig"]].append(c)
    ar_reps = sorted((min(v, key=lambda c: (c.get("rank", 0), c["sql"])) for v in ar.values()),
                     key=lambda c: (not c.get("ok"), c.get("rank", 0), c["sql"]))
    for c in ar_reps[:ARITH_MAX]:
        cands.append(clean_cand(c, "arith", rank=2000 + int(c.get("rank", 0))))
        st[f"arith_{'right' if c.get('ok') else 'wrong'}"] += 1
    while sum(not c.get("mutant") for c in cands) > CAP:
        drop = [j for j, c in enumerate(cands) if c["prov"] == "g" and not c.get("ok")]
        drop = drop or [j for j, c in enumerate(cands) if c["prov"] in ("arith", "ff") and not c.get("ok")]
        if not drop:
            break
        j = max(drop, key=lambda j: (cands[j]["rank"], cands[j]["sql"]))
        st[f"trimmed_{cands[j]['prov']}"] += 1
        del cands[j]
    out["cands"] = cands
    return out, st


def has_both(rec: dict) -> bool:
    oks = [bool(c.get("ok")) for c in rec["cands"]]
    return any(oks) and not all(oks)


def ff_extend(a) -> None:
    sys.path.insert(0, str(ROOT / "scripts/freeform"))
    import probe  # noqa: E402
    cache: dict = {}

    def canon(s):
        if s not in cache:
            cache[s] = probe.canon(s)
        return cache[s]

    def jl(p):
        return {r["id"]: r for r in map(json.loads, open(p))} if p else {}
    qs = [json.loads(l) for l in open(a.questions)]
    glists, lq, lo, la = jl(a.grammar_lists), jl(a.ff_qwen), jl(a.ff_omni), jl(a.arith)
    st = collections.Counter()
    classes_by_q, gold_ok = {}, {}
    for q in qs:
        i = q["id"]
        stat = [(r.get("ff") or {}).get("gold_status", "ok") for r in (lq.get(i), lo.get(i)) if r is not None]
        if not stat:
            st["no_ff_samples"] += 1
            continue
        if any(s != "ok" for s in stat):
            st["gold_not_ok_at_label_time"] += 1
            continue
        gold_ok[i] = True
        golds = {canon(g) for g in (q.get("gold_sql"), q.get("gold_sql_orig")) if g}
        golds.discard("")
        smp = {"qwen": (lq.get(i) or {}).get("cands", []), "omni": (lo.get(i) or {}).get("cands", [])}
        classes_by_q[i], s2 = ff_classes(smp, golds, canon)
        st.update(s2)
    cap = cap_source_rate(classes_by_q, "qwen", "omni", a.seed)
    print("CAP " + json.dumps(cap), flush=True)
    train, val = [], []
    kinds = collections.Counter()
    for q in qs:
        i = q["id"]
        if i not in gold_ok:
            continue
        rec, s2 = mix_list(q, glists.get(i), classes_by_q.get(i, {}), (la.get(i) or {}).get("cands", []), a.seed)
        st.update(s2)
        kind = q.get("kind") or ("list" if i in glists else "uncovered")
        if not has_both(rec):
            st[f"drop_not_right_and_wrong_{kind}"] += 1
            continue
        kinds[kind] += 1
        (val if rec["db_id"] in VAL_DBS else train).append(rec)

    def summ(rows):
        prov = collections.Counter((c["prov"], bool(c.get("ok"))) for r in rows for c in r["cands"])
        n = max(len(rows), 1)
        return {"lists": len(rows), "dbs": len({r["db_id"] for r in rows}),
                "mean_entries": round(sum(len(r["cands"]) for r in rows) / n, 2),
                "max_entries": max((len(r["cands"]) for r in rows), default=0),
                "ff_only_lists": sum(not r.get("covered") for r in rows),
                "lists_with_ff": sum(any(c["prov"] == "ff" for c in r["cands"]) for r in rows),
                "weight3_share": round(sum(r["weight"] == 3.0 for r in rows) / n, 4),
                "entries_by_prov": {f"{p}_{'right' if ok else 'wrong'}": v for (p, ok), v in sorted(prov.items())}}

    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train_lists", train), ("val_lists", val)):
        with open(a.out / f"{name}.jsonl", "w") as f:
            for r in sorted(rows, key=lambda r: qindex(r["id"])):
                f.write(json.dumps(r) + "\n")
    stats = {"mode": "ff-extend", "seed": a.seed, "ff_max": FF_MAX, "ff_only_max": FF_ONLY_MAX, "arith_max": ARITH_MAX,
             "swap_p": SWAP_P, "cap": CAP, "max_mutants": MAX_MUT, "val_dbs": list(VAL_DBS), "questions": len(qs),
             "kinds_kept": dict(kinds), "omni_cap": cap, "train": summ(train), "val": summ(val), "build": dict(st)}
    (a.out / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print("FF2LISTS " + json.dumps({k: stats[k] for k in ("kinds_kept", "train", "val")}), flush=True)


# ---- data -------------------------------------------------------------------------------------------------------

def load_pools(specs: list[str] | None = None) -> dict[str, dict[str, dict]]:
    """Default: box W's R1 and R2 BIRD-train pools. specs ["name=path", ...] (--pool, e.g. VALHARD's one-generator
    lists "r1=runs/phase1/p1w/p1w/r1/cands_train.jsonl") replace them; the first pool supplies the question record."""
    if not specs:
        specs = [f"{role}={POOLS / role / 'cands_train.jsonl'}" for role in ("r1", "r2")]
    out = {}
    for sp in specs:
        name, path = sp.split("=", 1)
        out[name] = {r["id"]: r for r in map(json.loads, open(path))}
    return out


def verified() -> dict[int, dict]:
    fx = {}
    for f in ("bird-verified-train.json", "bird-verified-val.json"):
        for r in json.load(open(ROOT / "data/raw/bird_verified" / f)):
            fx[int(r["question_id"])] = r
    return fx


def db_columns(path: str) -> dict[str, list[str]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')] for t in tabs}
    finally:
        con.close()


def stage_ff(a, out_dir: Path, tr: list, held: set, cont: set, noise: set, fx: dict, paths: dict, recs: dict,
             built: dict, norights: dict, train: list, val: list) -> None:
    """--stage-ff outputs (see the module doc): grammar_lists_all.jsonl + questions.jsonl."""
    listed = {r["id"] for r in train + val}
    split = lambda db: "val" if db in VAL_DBS else "train"  # noqa: E731
    rows = [(i, out, "list" if i in listed else "nowrong") for i, (out, _) in built.items()]
    rows += [(i, out, "noright") for i, out in norights.items()]
    kind_of = {}
    with open(out_dir / "grammar_lists_all.jsonl", "w") as f:
        for i, out, kind in sorted(rows, key=lambda x: qindex(x[0])):
            kind_of[i] = kind
            f.write(json.dumps({**out, "kind": kind, "split": split(out["db_id"])}) + "\n")
    # grammar-uncovered BIRD-train questions on the same DBs, relabelled and gold-checked like the pool questions
    dbs = {r["db_id"] for r in recs.values()}
    st = collections.Counter()
    unc, by_db = {}, collections.defaultdict(list)
    for n, r in enumerate(tr):
        i = f"bird_train_train_{n}"
        if i in recs or r["db_id"] not in dbs or r["db_id"] in held or r["db_id"] in cont:
            continue
        if i in kind_of:
            continue
        v = fx.get(n)
        if i in noise and v is None:
            st["uncovered_drop_gold_noise"] += 1
            continue
        changed = v is not None and " ".join((v["SQL"] or "").split()) != " ".join(r["SQL"].split())
        if changed and v.get("grading_method", "set").strip() != "set":
            st["uncovered_drop_verified_not_set_graded"] += 1
            continue
        q = {"id": i, "db_id": r["db_id"], "question": (v or r)["question"],
             "context": (v.get("evidence") or "") if v is not None else r.get("evidence", ""),
             "gold_sql": v["SQL"] if changed else r["SQL"], "gold_corrected": changed, "in_slots": False,
             "covered": False}
        unc[i] = q
        by_db[r["db_id"]].append((i, q["gold_sql"], []))
    res = execq.run_tasks(by_db, paths, GOLD_SECONDS, GEN_SECONDS, workers=a.workers)
    qrows = []
    for i, rec in recs.items():
        qrows.append({k: rec.get(k) for k in ("id", "db_id", "question", "context", "gold_sql", "gold_corrected",
                                              "in_slots", "covered")} | {"kind": kind_of[i]})
    for i, q in unc.items():
        status = res[i][0]
        if status != "ok":
            st[f"uncovered_drop_gold_{status}"] += 1
            continue
        qrows.append(q | {"kind": "uncovered"})
    with open(out_dir / "questions.jsonl", "w") as f:
        for q in sorted(qrows, key=lambda q: qindex(q["id"])):
            q["gold_sql_orig"] = tr[qindex(q["id"])]["SQL"]
            q["split"] = split(q["db_id"])
            f.write(json.dumps(q) + "\n")
    st.update(f"questions_{q['kind']}" for q in qrows)
    (out_dir / "stage_ff_stats.json").write_text(json.dumps(dict(st), indent=1))
    print("STAGE_FF " + json.dumps(dict(st)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (default data/val2)")
    ap.add_argument("--pool", action="append", default=None, metavar="NAME=PATH",
                    help="candidate pool(s) to build from (cands_train.jsonl format), repeatable; default box W's r1 "
                    "and r2 (VALHARD: one generator, e.g. r1=runs/phase1/p1w/p1w/r1/cands_train.jsonl)")
    ap.add_argument("--stage-ff", action="store_true", help="phase 2: also write grammar_lists_all.jsonl and "
                    "questions.jsonl (listed + grammar-uncovered questions, relabelled) to --out")
    ap.add_argument("--ff-extend", action="store_true", help="phase 2: mixed lists from --grammar-lists, --questions, "
                    "--ff-qwen, --ff-omni [--arith] (no DB needed); nothing else runs")
    ap.add_argument("--grammar-lists", type=Path)
    ap.add_argument("--questions", type=Path)
    ap.add_argument("--ff-qwen", type=Path, help="merge.py --samples output of the clean-proxy generator")
    ap.add_argument("--ff-omni", type=Path, help="merge.py --samples output of OmniSQL (right rate capped)")
    ap.add_argument("--arith", type=Path, help="labelled arithmetic mutants (pool schema, optional)")
    a = ap.parse_args()
    if a.ff_extend:
        if not (a.grammar_lists and a.questions and a.ff_qwen and a.ff_omni):
            sys.exit("--ff-extend needs --grammar-lists, --questions, --ff-qwen and --ff-omni")
        ff_extend(a)
        return
    t0 = time.time()
    out_dir = a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    held = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
    cont = set(json.load(open(ROOT / "data/sys1/contaminated_dbs.json"))["dbs"])
    noise = set(json.load(open(ROOT / "data/sys1/gold_noise_drop.json"))["ids"])
    fx = verified()
    pools = load_pools(a.pool)
    roles = list(pools)
    ids = sorted(set().union(*(set(p) for p in pools.values())), key=qindex)
    paths = {p.stem: str(p) for p in DBROOT.glob("*/*.sqlite")}
    st = collections.Counter()

    # ---- 1 relabel -------------------------------------------------------------------------------------------
    recs, lists_of = {}, {}
    by_db = collections.defaultdict(list)
    for i in ids:
        base = next(pools[r][i] for r in roles if i in pools[r])
        n, db = qindex(i), base["db_id"]
        st["pool_questions"] += 1
        if tr[n]["db_id"] != db:
            st["drop_db_mismatch"] += 1
            continue
        if db in held or db in cont:
            st["drop_heldout_or_contaminated"] += 1
            continue
        v = fx.get(n)
        if i in noise and v is None:
            st["drop_gold_noise"] += 1
            continue
        changed = v is not None and " ".join((v["SQL"] or "").split()) != " ".join(tr[n]["SQL"].split())
        if changed and v.get("grading_method", "set").strip() != "set":
            st["drop_verified_not_set_graded"] += 1
            continue
        rec = dict(base)
        if v is not None:
            st["verified_entry"] += 1
            if v["question"] != rec["question"]:
                st["verified_question_text_changed"] += 1
            rec["question"], rec["context"] = v["question"], v.get("evidence") or ""
        lists = [(role, [dict(c) for c in pools[role][i]["cands"]]) for role in roles if i in pools[role]]
        rec["gold_corrected"] = changed
        if changed:
            rec["gold_sql"] = v["SQL"]
            u = union_cands(lists)
            by_db[db].append((i, v["SQL"], [c["sql"] for c in u]))
        else:
            by_db[db].append((i, rec["gold_sql"], []))
        recs[i], lists_of[i] = rec, lists
    print(f"executing {sum(len(v) for v in by_db.values())} golds + "
          f"{sum(len(s) for v in by_db.values() for _, _, s in v)} candidates on {len(by_db)} DBs", flush=True)
    res = execq.run_tasks(by_db, paths, GOLD_SECONDS, GEN_SECONDS, workers=a.workers)
    flips = collections.defaultdict(collections.Counter)
    gold_sig = {}
    for i in list(recs):
        status, gsig, cres = res[i]
        rec = recs[i]
        if status != "ok":
            st[f"drop_gold_{status}"] += 1
            del recs[i]
            continue
        gold_sig[i] = gsig
        if not rec["gold_corrected"]:
            continue
        st["corrected_questions"] += 1
        new = {c["sql"]: r for c, r in zip(union_cands(lists_of[i]), cres)}
        db = rec["db_id"]
        q_flip = False
        for role, cands in lists_of[i]:
            for c in cands:
                if not c.get("sql"):
                    continue
                runs, nonnull, ok, sig, prev = new[c["sql"]]
                if bool(ok) != bool(c.get("ok")):
                    flips[db]["cand_" + ("to_right" if ok else "to_wrong")] += 1
                    q_flip = True
                if bool(runs) != bool(c.get("runs")):
                    st["cand_runs_changed_on_reexec"] += 1
                c.update(runs=runs, nonnull=nonnull, ok=ok, sig=sig, res=prev)
        old_any = any(c.get("ok") and live(c) for _, cs in [(r, pools[r][i]["cands"]) for r in roles if i in pools[r]] for c in cs)
        new_any = any(c.get("ok") and live(c) for _, cs in lists_of[i] for c in cs)
        flips[db]["questions_corrected"] += 1
        flips[db]["questions_with_label_flip"] += q_flip
        flips[db]["oracle_right_to_wrong"] += old_any and not new_any
        flips[db]["oracle_wrong_to_right"] += new_any and not old_any
    tot = collections.Counter()
    for c in flips.values():
        tot.update(c)
    rel = {"stats": dict(st), "totals": dict(tot), "per_db": {d: dict(c) for d, c in sorted(flips.items())}}
    (out_dir / "relabel_stats.json").write_text(json.dumps(rel, indent=1))
    print("relabel", json.dumps(rel["stats"]), json.dumps(rel["totals"]), flush=True)

    # ---- 2 lists ---------------------------------------------------------------------------------------------
    built, mut_jobs, norights = {}, collections.defaultdict(list), {}
    cols_cache = {}
    bs = collections.Counter()
    for i, rec in recs.items():
        out, info = build_list(rec, lists_of[i], a.seed)
        out["gold_corrected"] = rec["gold_corrected"]
        right = [c for c in out["cands"] if c["ok"]]
        if not right:
            bs["no_live_right"] += 1
            if a.stage_ff:
                norights[i] = out
            continue
        built[i] = (out, info)
        # the right SQL to mutate: the representative if it round-trips through sqlglot, else the best-ranked
        # member of the right class that does
        db = rec["db_id"]
        if db not in cols_cache:
            cols_cache[db] = db_columns(paths[db])
        cand_sqls = [right[0]["sql"]] + [m["sql"] for m in info["right_members"] if m["sql"] != right[0]["sql"]]
        muts = []
        for s in cand_sqls:
            muts = mutate.mutants(s, cols_cache[db], seed=f"{a.seed}:{i}")
            if muts:
                bs["mutation_source_is_rep" if s == right[0]["sql"] else "mutation_source_other_member"] += 1
                break
        if not muts:
            bs["no_mutable_right_sql"] += 1
            continue
        out["_muts"] = muts
        mut_jobs[db].append((i, rec["gold_sql"], [m for _, m in muts]))
    print(f"mutants: executing {sum(len(s) for v in mut_jobs.values() for _, _, s in v)} on {len(mut_jobs)} DBs",
          flush=True)
    mres = execq.run_tasks(mut_jobs, paths, GOLD_SECONDS, GEN_SECONDS, workers=a.workers)
    kind_tried, kind_kept = collections.Counter(), collections.Counter()
    for i, (out, info) in built.items():
        muts = out.pop("_muts", None)
        if not muts:
            continue
        status, gsig, r = mres[i]
        if status != "ok":
            bs["mutant_gold_not_ok"] += 1
            continue
        kind_tried.update(k for k, _ in muts)
        trials = [(k, s, x) for (k, s), x in zip(muts, r)]
        got = pick_mutants(trials, {c["sig"] for c in out["cands"]})
        kind_kept.update(m["mutation"] for m in got)
        out["cands"] += got
        bs[f"lists_with_{len(got)}_mutants"] += 1

    train, val = [], []
    for i, (out, info) in built.items():
        n_right = sum(c["ok"] for c in out["cands"])
        n_wrong_real = sum(not c["ok"] and not c.get("mutant") for c in out["cands"])
        n_wrong = sum(not c["ok"] for c in out["cands"])
        if n_wrong == 0:
            bs["drop_no_wrong_class"] += 1
            continue
        if n_wrong_real == 0:
            bs["kept_only_mutant_negatives"] += 1
        (val if out["db_id"] in VAL_DBS else train).append(out)
        bs["union_cands"] += info["union"]
        bs["live_cands"] += info["live"]
        bs["live_classes"] += info["classes"]
        bs["classes_capped"] += info["classes"] > CAP

    def summ(rows):
        m = [sum(bool(c.get("mutant")) for c in r["cands"]) for r in rows]
        return {"lists": len(rows), "dbs": len({r["db_id"] for r in rows}),
                "mean_entries": round(sum(len(r["cands"]) for r in rows) / max(len(rows), 1), 2),
                "mean_real_classes": round(sum(len(r["cands"]) - k for r, k in zip(rows, m)) / max(len(rows), 1), 2),
                "lists_with_mutant": sum(k > 0 for k in m), "mutants": sum(m),
                "weight3_share": round(sum(r["weight"] == 3.0 for r in rows) / max(len(rows), 1), 4),
                "gold_corrected": sum(r["gold_corrected"] for r in rows)}

    for name, rows in (("train_lists", train), ("val_lists", val)):
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in sorted(rows, key=lambda r: qindex(r["id"])):
                f.write(json.dumps(r) + "\n")
    stats = {"seed": a.seed, "cap": CAP, "max_mutants": MAX_MUT, "val_dbs": list(VAL_DBS),
             "train": summ(train), "val": summ(val), "build": dict(bs),
             "mutation_kinds_tried": dict(kind_tried), "mutation_kinds_kept": dict(kind_kept),
             "minutes": round((time.time() - t0) / 60, 1)}
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    if a.stage_ff:
        stage_ff(a, out_dir, tr, held, cont, noise, fx, paths, recs, built, norights, train, val)


if __name__ == "__main__":
    main()
