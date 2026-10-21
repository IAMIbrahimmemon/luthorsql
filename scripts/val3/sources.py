"""VAL3 data v3, step 1 (Mac, CPU): the NEW question sets the box turns into R1 k=32 lists (scripts/chains/val3_lists_box.sh).

  .venv/bin/python scripts/val3/sources.py [--synsql-dbs 3300] [--synsql-val-dbs 200] [--workers 4]

Design: docs/paper_notes/research/validator/04_training_data.md idea #1 (data v3), 00_synthesis.md.
Generator: R1 = runs/r1r2/r1 (grammar v3). What R1 trained on is re-derived from its args.json, not taken from notes:
  plans = sys1/{bird_train, spider_train, synsql_train} + rare1; train_sq2.load_sets(strict=True): rows with
  outcome "plan", not empty_result, inside the v2 slots (make_example), twins included; dev_dbs_bird.json DBs removed.
  On the Mac the bird_train dir reproduces R1's log exactly (9,728 examples + 167 skipped); trained ids are recomputed
  here (--trained-ids caches them).

Sources written (one plan dir each: plans.jsonl + cards/ + dbs.json, read by `rerank_sq2.py gen --split dev
--all-questions --dev-dbs <dir>/dbs.json`, so uncovered rows are generated too):
  bird_rr   BIRD-train questions on the 47 BIRD-train DBs the box already has (data/sys1/rrdbs):
            a = R1 trained on them (base row trained) but R1 has no k=32 list (not in p1w cands_train)  -> in-sample
            b = R1 never trained on them: no plan row, outcome unsupported, or a plan row outside the v2 slots;
                not a twin-trained question, no rare1 mutant, question text not equal to a trained one -> out-of-sample
  bird_big  the same kinds on 10 of the 13 large BIRD-train DBs that are NOT in rrdbs (~17 GB; the box downloads the
            public BIRD train archive and extracts only these; skipped if the download fails). BIG_SLOW (3 DBs whose gold
            SQL times out at 3 s on 35-59% of questions) is left out.
  synsql    SynSQL-2.5M questions R1 never saw (id not in sys1/synsql_train/plans.jsonl, text not equal to a planned
            question on the same DB), on DBs R1 trained on (cards in sys1/synsql_train/cards), converted to verified
            plans here (scripts/prepare_plans.process_db, grammar v3 vocabulary of R1) so every row is grammar-covered
            like DECISION; non-empty gold only; per DB at most --synsql-keep questions, complexity mix
            --synsql-mix (toward Moderate / Complex).
Dropped everywhere: held-out DBs (dev_dbs_bird.json = DECISION + RESERVED), contaminated DBs, every sealed or
eval DB name (data/sys1/sealed/**, KaggleDBQA, Spider dev + test DBs, BIRD dev DBs), BIRD-train gold-noise ids
without a BIRD-Verified entry, Verified corrections graded other than as a set, plan rows whose gold is empty.
NOT used (sealed: data/eval/sealed_* holds Spider dev, Spider test, Spider-Syn, Spider-DK, Spider-Realistic, KaggleDBQA,
BIRD dev): no Spider dev/test question, no question on any of their DBs. BIRD dev is never read.
Split: train | val. val = DBs no train list is on: the 4 pinned BIRD val DBs (only b questions there) + --synsql-val-dbs
SynSQL DBs (seeded hash). Written:
  data/val3/q/{bird_rr,bird_big,synsql}/{plans.jsonl,cards/,dbs.json}
  data/val3/q/meta.jsonl      one row per question: id, db_id, src (bird | synsql), kind (a | b), sub, split, verified
  data/val3/dbs/synsql/<db>/<db>.sqlite   hard links to data/raw/synsql (the box's SynSQL db root; no disk cost)
  data/val3/big_dbs.json      {db: bytes} of the large DBs used (the box checks its extracted files against it)
  data/val3/q/sources_stats.json
Prints counts only (no label, no score).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

S = ROOT / "data/sys1"
OUT = ROOT / "data/val3"
VAL_DBS = ("citeseer", "disney", "professional_basketball", "trains")
SYN_ROOT = ROOT / "data/raw/synsql"
COMPLEXITIES = ("Simple", "Moderate", "Complex", "Highly Complex")
# Large BIRD-train DBs left out: their gold SQL alone timed out at the box's 3 s limit on 35-59% of these questions
# (Mac, 2026-10-11: donor 88/150, world_development_indicators 65/151, language_corpus 52/129) and one DB is one
# serial execution task in rerank_sq2 gen (k=32 candidates each), so they would cost hours of box time for few lists.
BIG_SLOW = ("donor", "language_corpus", "world_development_indicators")


# ---- pure helpers (tests/test_val3.py) -------------------------------------------------------------------------------

def unit(*parts) -> float:
    """Stable uniform [0, 1) from the parts."""
    return int(hashlib.md5(":".join(map(str, parts)).encode()).hexdigest()[:12], 16) / float(16 ** 12)


def qnorm(q: str) -> str:
    return " ".join((q or "").lower().split())


def bird_kind(i: str, row: dict | None, trained: set, twin_trained: set, rare_bases: set, trained_text: set,
              db: str, question: str) -> tuple[str, str]:
    """(kind, sub) of one BIRD-train question for R1. kind: a (trained, in-sample) | b (never trained, out-of-sample) |
    x (excluded). sub says why."""
    if i in trained:
        return "a", "trained"
    if i in twin_trained:
        return "x", "twin_trained"       # a value-swapped twin of it was trained: near in-sample
    if i in rare_bases:
        return "x", "rare1_mutant"       # a rare1 mutant of it was trained
    if (db, qnorm(question)) in trained_text:
        return "x", "dup_trained_text"
    if row is None:
        return "b", "norow"
    if row.get("outcome") != "plan":
        return "b", "unsupported"
    if row.get("empty_result"):
        return "x", "empty_gold"
    return "b", "outside_slots"


def gold_ok(n: int, i: str, noise: set, fx: dict, orig_sql: str) -> str | None:
    """None when usable, else the drop reason (build_lists' relabel rules)."""
    v = fx.get(n)
    if i in noise and v is None:
        return "gold_noise"
    changed = v is not None and " ".join((v["SQL"] or "").split()) != " ".join(orig_sql.split())
    if changed and v.get("grading_method", "set").strip() != "set":
        return "verified_not_set_graded"
    return None


def pick_synsql(rows: list[dict], keep: int, mix: dict[str, float], seed: str) -> list[dict]:
    """Up to `keep` planned rows of one DB: quotas by complexity (largest remainder of keep * mix), leftovers filled
    in mix order; inside a complexity a seeded order."""
    by = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: unit(seed, r["id"])):
        by[r.get("sql_complexity") or "Simple"].append(r)
    order = sorted(mix, key=lambda c: -mix[c])
    quota = {c: int(keep * mix[c]) for c in order}
    rem = keep - sum(quota.values())
    for c in sorted(order, key=lambda c: -(keep * mix[c] - int(keep * mix[c])))[:rem]:
        quota[c] += 1
    out = []
    for c in order:
        out += by[c][:quota[c]]
        by[c] = by[c][quota[c]:]
    for c in order + [c for c in by if c not in mix]:
        while len(out) < keep and by[c]:
            out.append(by[c].pop(0))
    return out


# ---- data -------------------------------------------------------------------------------------------------------------

def excluded_dbs() -> dict[str, str]:
    """db name (lower case) -> reason. Held-out, contaminated, every sealed / eval DB."""
    ex = {}
    for d in json.load(open(S / "dev_dbs_bird.json")):
        ex[d.lower()] = "heldout"
    for d in json.load(open(S / "contaminated_dbs.json"))["dbs"]:
        ex.setdefault(d.lower(), "contaminated")
    for d in json.load(open(S / "bird_dev_dbs.json")):
        ex.setdefault(d.lower(), "bird_dev")
    for p in (S / "sealed").glob("*/cards/*.json"):   # spider_dk, kaggledbqa, ... schema cards
        ex.setdefault(p.stem.lower(), "sealed")
    for p in (S / "sealed/dbs").glob("*"):              # the sealed sets' database folders
        ex.setdefault(p.name.lower(), "sealed")
    for p in (S / "sealed/dbs").rglob("*.sqlite"):
        ex.setdefault(p.stem.lower(), "sealed")
    for d in json.load(open(S / "sealed/kaggledbqa.dbs.json")):
        ex.setdefault(d.lower(), "sealed")
    for f in ("dev.json", "test.json"):
        for q in json.load(open(ROOT / "data/raw/spider_data" / f)):
            ex.setdefault(q["db_id"].lower(), "sealed_spider_" + f.split(".")[0])
    return ex


def trained_bird_ids(cache: Path) -> set[str]:
    """R1's bird_train training rows (train_sq2.load_sets strict, grammar v3, R1's vocab and cand order)."""
    if cache.exists():
        return set(json.loads(cache.read_text()))
    os.environ.setdefault("LUTHOR_GRAMMAR", "v3")
    from luthor.grammar_cli import preset
    preset()
    import train_sq2
    ra = json.load(open(ROOT / "runs/r1r2/r1/args.json"))
    v = json.load(open(ROOT / "runs/r1r2/r1/vocab.json"))
    train_sq2.VOCAB_ARGS = (v["templates"], v["nums"])
    train_sq2.CAND_ORDER = ra.get("cand_order", "v1")
    train_sq2._init_worker(train_sq2.VOCAB_ARGS, train_sq2.CAND_ORDER)
    _, exs, _ = train_sq2.load_sets([S / "bird_train"], workers=4, strict=True)
    ids = sorted(e.id for e in exs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ids))
    return set(ids)


def verified() -> dict[int, dict]:
    fx = {}
    for f in ("bird-verified-train.json", "bird-verified-val.json"):
        for r in json.load(open(ROOT / "data/raw/bird_verified" / f)):
            fx[int(r["question_id"])] = r
    return fx


def write_dir(d: Path, rows: list[dict], card_src: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    (d / "cards").mkdir(parents=True)
    dbs = sorted({r["db_id"] for r in rows})
    for db in dbs:
        shutil.copy2(card_src / f"{db}.json", d / "cards" / f"{db}.json")
    with open(d / "plans.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    (d / "dbs.json").write_text(json.dumps(dbs))


def bird_sources(a, ex: dict, meta: list, st: collections.Counter) -> None:
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    rows = [json.loads(l) for l in open(S / "bird_train/plans.jsonl")]
    base = {r["id"]: r for r in rows if not r.get("twin_of")}
    trained = trained_bird_ids(a.trained_ids)
    twin_trained = {r["twin_of"] for r in rows if r.get("twin_of") and r["id"] in trained}
    trained = {i for i in trained if i in base}  # base rows (questions); twins are handled above
    rare_bases = {json.loads(l)["id"].split("rare1_", 1)[1].split("_mut_")[0] for l in open(ROOT / "data/rare1/plans.jsonl")}
    trained_text = {(base[i]["db_id"], qnorm(base[i]["question"])) for i in trained}
    listed = {json.loads(l)["id"] for l in open(ROOT / "runs/phase1/p1w/p1w/r1/cands_train.jsonl")}
    noise = set(json.load(open(S / "gold_noise_drop.json"))["ids"])
    fx = verified()
    rr = {p.stem for p in (S / "rrdbs").rglob("*.sqlite")}
    sizes = {p.stem: p.stat().st_size for p in (ROOT / "data/raw/bird_train/train_databases").glob("*/*.sqlite")}
    out = {"bird_rr": [], "bird_big": []}
    for n, q in enumerate(tr):
        i, db = f"bird_train_train_{n}", q["db_id"]
        if db.lower() in ex:
            st[f"bird_drop_db_{ex[db.lower()]}"] += 1
            continue
        if i in listed:
            st["bird_skip_already_listed"] += 1
            continue
        kind, sub = bird_kind(i, base.get(i), trained, twin_trained, rare_bases, trained_text, db, q["question"])
        if kind == "x":
            st[f"bird_drop_{sub}"] += 1
            continue
        why = gold_ok(n, i, noise, fx, q["SQL"])
        if why:
            st[f"bird_drop_{why}"] += 1
            continue
        r = base.get(i)
        if r is None:  # the uncovered row rerank_sq2 gen builds for --all-questions (train_sq2.raw_rows' format)
            r = {"id": i, "source": "bird", "split": "train", "db_id": db, "question": q["question"].strip(),
                 "context": (q.get("evidence") or "").strip(), "gold_sql": q["SQL"], "outcome": "missing"}
        split = "val" if db in VAL_DBS else "train"
        if kind == "a" and split == "val":
            st["bird_drop_insample_on_val_db"] += 1  # val is out-of-sample only
            continue
        where = "bird_rr" if db in rr else "bird_big"
        if db in BIG_SLOW:
            st["bird_drop_slow_big_db"] += 1
            continue
        out[where].append(r)
        meta.append({"id": i, "db_id": db, "src": "bird", "kind": kind, "sub": sub, "split": split,
                     "verified": n in fx, "dir": where})
        st[f"{where}_{kind}_{sub}_{split}"] += 1
    for where, rs in out.items():
        write_dir(a.out / "q" / where, rs, S / "bird_train/cards")
    big = {db: sizes[db] for db in sorted({r["db_id"] for r in out["bird_big"]})}
    (a.out / "big_dbs.json").write_text(json.dumps(big, indent=1))
    st["big_dbs"] = len(big)
    st["big_db_bytes"] = sum(big.values())


# ---- SynSQL -----------------------------------------------------------------------------------------------------------

_SYN = {}


def _syn_init(allowed, nums):
    import prepare_plans as pp
    pp._set_unsup(0.0, None, allowed, nums)  # unsupported rows are not written


def _syn_plan(job):
    import prepare_plans as pp
    db, entry, items = job
    rows, stats, _ = pp.process_db(("synsql", SYN_ROOT, "train", db, entry, items))
    return db, [r for r in rows if not r.get("twin_of")], stats


def synsql_source(a, ex: dict, meta: list, st: collections.Counter) -> None:
    t0 = time.time()
    cards = {p.stem for p in (S / "synsql_train/cards").glob("*.json")}
    seen_ids, seen_text = set(), set()
    for l in open(S / "synsql_train/plans.jsonl"):
        r = json.loads(l)
        seen_ids.add(r["id"])
        seen_text.add((r["db_id"], qnorm(r["question"])))
    ok_dbs = sorted(d for d in cards if d.lower() not in ex
                    and (SYN_ROOT / "databases" / d / f"{d}.sqlite").exists())
    st["synsql_dbs_with_card"] = len(cards)
    st["synsql_dbs_excluded_name"] = sum(d.lower() in ex for d in cards)
    rnk = sorted(ok_dbs, key=lambda d: unit(a.seed, "syndb", d))
    want = rnk[:a.synsql_dbs + a.synsql_val_dbs]
    val_dbs = set(sorted(want, key=lambda d: unit(a.seed, "synval", d))[:a.synsql_val_dbs])
    want = set(want)
    # stream data.json (one record per line): unseen questions on the chosen DBs, a seeded few per complexity
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(SYN_ROOT / "data.json") as f:
        n = -1
        for line in f:
            s = line.strip().lstrip("[").lstrip(",").rstrip("]").strip()
            if not s:
                continue
            n += 1
            if '"db_id": "' not in s:
                continue
            db = s.split('"db_id": "', 1)[1].split('"', 1)[0]
            if db not in want:
                continue
            q = json.loads(s)
            i = f"synsql_train_data_{n}"
            if i in seen_ids or (db, qnorm(q["question"])) in seen_text:
                st["synsql_skip_seen"] += 1
                continue
            per[db][q.get("sql_complexity") or "Simple"].append((unit(a.seed, "synq", i), n, q))
    st["synsql_records"] = n + 1
    tables = {}
    with open(SYN_ROOT / "tables.json") as f:
        for e in json.load(f):
            if e["db_id"] in want:
                tables[e["db_id"]] = e
    jobs = []
    for db in sorted(per):
        items = []
        for c, lst in per[db].items():
            for _, n, q in sorted(lst)[:a.synsql_try_per_complexity]:
                items.append((f"data_{n}", {"db_id": db, "question": q["question"], "sql": q["sql"],
                                            "context": q.get("external_knowledge") or "", "question_id": f"data_{n}",
                                            "sql_complexity": c}))
        if db in tables:
            jobs.append((db, tables[db], items))
    v = json.load(open(ROOT / "runs/r1r2/r1/vocab.json"))
    print(f"synsql: planning {sum(len(j[2]) for j in jobs)} questions on {len(jobs)} DBs", flush=True)
    cplx = {f"synsql_train_data_{it[1]['question_id'].split('_')[1]}": it[1]["sql_complexity"] for j in jobs for it in j[2]}
    rows_by_db, pst = {}, collections.Counter()
    os.environ.setdefault("LUTHOR_GRAMMAR", "v3")
    with Pool(a.workers, initializer=_syn_init, initargs=(set(v["templates"]), set(v["nums"]))) as pool:
        for k, (db, rows, s2) in enumerate(pool.imap_unordered(_syn_plan, jobs), 1):
            rows_by_db[db] = rows
            pst.update(s2)
            if k % 250 == 0:
                print(f"  {k}/{len(jobs)} DBs, {time.time() - t0:.0f}s", flush=True)
    st.update({f"synsql_plan_{k}": v for k, v in pst.items() if not k.startswith("reason:")})
    mix = dict(zip(COMPLEXITIES, a.synsql_mix))
    rows = []
    for db in sorted(rows_by_db):
        ok = []
        for r in rows_by_db[db]:
            if r.get("outcome") != "plan" or r.get("empty_result"):
                continue
            r["sql_complexity"] = cplx.get(r["id"], "Simple")
            ok.append(r)
        keep = pick_synsql(ok, a.synsql_keep, mix, f"{a.seed}:{db}")
        for r in keep:
            split = "val" if db in val_dbs else "train"
            rows.append(r)
            meta.append({"id": r["id"], "db_id": db, "src": "synsql", "kind": "b", "sub": r["sql_complexity"],
                         "split": split, "verified": False, "dir": "synsql"})
            st[f"synsql_{split}_{r['sql_complexity']}"] += 1
    write_dir(a.out / "q/synsql", rows, S / "synsql_train/cards")
    dbroot = a.out / "dbs/synsql"
    if dbroot.exists():
        shutil.rmtree(dbroot)
    for db in sorted({r["db_id"] for r in rows}):
        (dbroot / db).mkdir(parents=True)
        src, dst = SYN_ROOT / "databases" / db / f"{db}.sqlite", dbroot / db / f"{db}.sqlite"
        try:
            os.link(src, dst)
        except OSError:  # another file system (e.g. --out in a scratch dir)
            shutil.copy2(src, dst)
    st["synsql_questions"] = len(rows)
    st["synsql_dbs"] = len({r["db_id"] for r in rows})
    st["synsql_val_dbs"] = len({r["db_id"] for r in rows if r["db_id"] in val_dbs})
    st["synsql_db_bytes"] = sum((dbroot / d / f"{d}.sqlite").stat().st_size for d in os.listdir(dbroot))
    st["synsql_minutes"] = round((time.time() - t0) / 60, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trained-ids", type=Path, default=OUT / "r1_bird_trained_ids.json",
                    help="cache of R1's bird_train training row ids (recomputed when missing)")
    ap.add_argument("--synsql-dbs", type=int, default=3300, help="SynSQL train DBs sampled")
    ap.add_argument("--synsql-val-dbs", type=int, default=200, help="extra SynSQL DBs whose questions go to val only")
    ap.add_argument("--synsql-try-per-complexity", type=int, default=8, help="unseen questions planned per DB and complexity")
    ap.add_argument("--synsql-keep", type=int, default=4, help="planned questions kept per DB")
    ap.add_argument("--synsql-mix", type=float, nargs=4, default=[0.2, 0.4, 0.3, 0.1],
                    help="share of kept questions per complexity: Simple Moderate Complex 'Highly Complex'")
    ap.add_argument("--skip-synsql", action="store_true")
    a = ap.parse_args()
    os.environ["LUTHOR_GRAMMAR"] = "v3"  # R1's grammar; before any luthor.sq import (workers inherit it)
    t0 = time.time()
    (a.out / "q").mkdir(parents=True, exist_ok=True)
    ex = excluded_dbs()
    st, meta = collections.Counter(), []
    st["excluded_db_names"] = len(ex)
    bird_sources(a, ex, meta, st)
    if not a.skip_synsql:
        synsql_source(a, ex, meta, st)
    elif (a.out / "q/meta.jsonl").exists():  # keep the SynSQL rows of an earlier run
        meta += [m for m in map(json.loads, open(a.out / "q/meta.jsonl")) if m["src"] == "synsql"]
    with open(a.out / "q/meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    st["questions_total"] = len(meta)
    st["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / "q/sources_stats.json").write_text(json.dumps(dict(sorted(st.items())), indent=1))
    print(json.dumps(dict(sorted(st.items())), indent=1))


if __name__ == "__main__":
    main()
