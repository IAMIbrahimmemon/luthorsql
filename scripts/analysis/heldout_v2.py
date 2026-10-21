"""Held-out yardstick v2 (research problem 06-11): every BIRD-train question on the held-out training databases whose
ORIGINAL gold SQL executes, split into a DECISION set (used for every choice) and a RESERVED confirmation set (never
used for a decision; scored once per claim, with --confirm).

Why v2: the v1 yardstick (runs/r1r2/r2/rr/dev_scored.jsonl, n=1,000) holds only grammar-v3-covered questions, so a
grammar arm cannot show a coverage gain on it, and it scores plan-rendered gold rather than the ORIGINAL BIRD-train SQL.
v2 scores ORIGINAL gold, keeps uncovered questions, and drops retail_world (its SQLite file does not match the
questions: 330 of 373 golds fail with e.g. "no such table: Order Details").

  python scripts/analysis/heldout_v2.py build  [--seconds 30] [--workers 4] [--reserved-frac 0.30]   (writes data/sys1/heldout_v2/)
  python scripts/analysis/heldout_v2.py score CANDS.jsonl [--split decision|reserved|all --confirm] [--seconds 30] [--out per_id.jsonl]

CANDS.jsonl needs one row per question with fields `id` (bird_train_train_<train.json index>) and `sql`. A missing id
or a NULL/failing/timed-out SQL counts as wrong. Matching is BIRD execution accuracy: the candidate's row set equals
the gold's row set (luthor.sqlw.execute semantics, read-only SQLite, first 100,000 rows, a time limit per query).

Split rule (no label- or result-based choice): strata are (database, v3-covered, gold pathology); in each stratum the
reserved set takes round(frac * size) questions, preferring questions no file in USAGE_SOURCES has ever scored, then
SHA-256(seed, id) order. The build output is heldout_v2.out next to this file.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from luthor.sqlw.execute import connect, signature  # noqa: E402

TRAIN_JSON = REPO / "data/raw/bird_train/train.json"
DB_ROOT = REPO / "data/raw/bird_train/train_databases"
HELDOUT_DBS = REPO / "data/sys1/dev_dbs_bird.json"
COVERAGE_V3 = REPO / "reports/fanout_r3/grammar_coverage/v3_train.jsonl"
OUT_DIR = REPO / "data/sys1/heldout_v2"
SEED = "heldout_v2/2026-10-08"
EXCLUDED_DBS = {"retail_world": "SQLite file does not match the annotated schema; most ORIGINAL golds fail "
                                "(e.g. 'no such table: Order Details')"}
# Every repo file known to hold held-out question ids that something scored, generated for, or trained on.
# label -> (path, meaning). Ids are matched with the regex bird_train_train_<n>.
USAGE_SOURCES = {
    "v1_yardstick_1000": ("runs/r1r2/r2/rr/dev_scored.jsonl", "v1 held-out yardstick (R1/R2/r6c scored; all decisions)"),
    "ho200": ("reports/fanout_r3/rl/ho200.jsonl", "RL/k-best pool study; 01-01 pilot is its 154 executable items"),
    "01-01_pilot154": ("reports/research/01-01/heldout_kbest_candidates.jsonl", "01-01 pilot (subset of ho200)"),
    "01-01_confirmation200": ("reports/research/01-01/confirmation_candidates.jsonl", "01-01 confirmation cohort"),
    "sq3_pilot": ("runs/sq3_pilot/cands_train.jsonl", "older sq3 pilot candidate file"),
    "sqlw_rl_hard": ("runs/sqlw_sft/rl_hard.jsonl", "older sqlw RL prompt file (trained on by that lineage)"),
    "proof_protocol_fit": ("reports/fanout_r2/proof_protocol/inputs/fit_train_dev.jsonl", "round-2 proof protocol fit"),
    "docs_mentions": ("reports/fanout_r1/columns/r2/char_train_dev.out", "ids printed in a column study"),
    "p0102": ("reports/research/p0102/stackgap.json", "p0102 stack-gap examples"),
    "p0102_blockskip": ("reports/research/p0102/blockskip.json", "p0102 block-skip examples"),
}
ID_RE = re.compile(r"bird_train_train_\d+(?!\d)")
BIG = 100_000


def qid(i: int) -> str:
    return f"bird_train_train_{i}"


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pathology(rows) -> str | None:
    """Gold result shapes that BIRD-Verified marks wrong at 80-100% on train (04-07) and that ORIGINAL dev gold never has."""
    if not rows:
        return "empty"
    if len(rows) == 1 and len(rows[0]) == 1:
        v = rows[0][0]
        if v is None:
            return "single_null"
        if not isinstance(v, (bool, str)) and v == 0:
            return "single_zero"
    if all(v is None for r in rows for v in r):
        return "all_null"
    return None


def run_sql(run, sql):
    """-> ("ok", rows) | ("timeout", msg) | ("error", msg). `run` is luthor.sqlw.execute.connect(...)."""
    if not sql or not str(sql).strip():
        return "error", "empty SQL"
    try:
        return "ok", run(sql)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            return "timeout", "interrupted"
        return "error", str(e)[:200]
    except Exception as e:  # noqa: BLE001  (sqlite3.Warning for multiple statements, ValueError, ...)
        return "error", f"{type(e).__name__}: {str(e)[:200]}"


def db_path(db: str) -> str:
    return str(DB_ROOT / db / f"{db}.sqlite")


def _gold_chunk(task):
    db, items, seconds = task
    run = connect(db_path(db), seconds)
    out = []
    for i, sql in items:
        t0 = time.time()
        st, res = run_sql(run, sql)
        r = {"id": qid(i), "status": st, "secs": round(time.time() - t0, 3)}
        if st == "ok":
            r.update(nrows=len(res), ncols=len(res[0]) if res else 0, sig=signature(res), pathology=pathology(res),
                     capped=len(res) >= BIG)
        else:
            r["error"] = res
        out.append(r)
    return out


def exec_golds(pairs_by_db: dict, seconds: float, workers: int, chunk: int = 40) -> dict:
    """{db: [(train_index, sql)]} -> {id: gold record}. Chunks are spread over a process pool."""
    tasks = [(db, items[j:j + chunk], seconds) for db, items in pairs_by_db.items() for j in range(0, len(items), chunk)]
    tasks.sort(key=lambda t: -len(t[1]))
    out = {}
    with Pool(workers) as p:
        for res in p.imap_unordered(_gold_chunk, tasks):
            for r in res:
                out[r["id"]] = r
    return out


def usage_map(universe_ids) -> dict:
    """id -> sorted labels of USAGE_SOURCES files that mention it."""
    u = set(universe_ids)
    used = collections.defaultdict(set)
    for label, (rel, _) in USAGE_SOURCES.items():
        p = REPO / rel
        if not p.exists():
            continue
        for x in set(ID_RE.findall(p.read_text(errors="replace"))) & u:
            used[x].add(label)
    return {k: sorted(v) for k, v in used.items()}


def reserve_priority(labels) -> int:
    """0 never scored; 1 only in the 01-01 confirmation cohort; 2 scored but never in the v1 yardstick; 3 v1 yardstick."""
    s = set(labels)
    if not s:
        return 0
    if s <= {"01-01_confirmation200"}:
        return 1
    if "v1_yardstick_1000" not in s and "proof_protocol_fit" not in s:
        return 2
    return 3


def split(rows: list[dict], frac: float, seed: str = SEED):
    """rows: dicts with id, db_id, covered, pathology, used -> (decision ids, reserved ids). Deterministic."""
    strata = collections.defaultdict(list)
    for r in rows:
        strata[(r["db_id"], bool(r["covered"]), r["pathology"] is not None)].append(r)
    dec, res = [], []
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda r: (reserve_priority(r["used"]),
                                                    hashlib.sha256(f"{seed}|{r['id']}".encode()).hexdigest()))
        k = int(math.floor(frac * len(items) + 0.5))
        res += [r["id"] for r in items[:k]]
        dec += [r["id"] for r in items[k:]]
    return sorted(dec, key=_idnum), sorted(res, key=_idnum)


def _idnum(x: str) -> int:
    return int(x.rsplit("_", 1)[1])


def load_coverage() -> dict:
    cov = {}
    if COVERAGE_V3.exists():
        for line in open(COVERAGE_V3):
            r = json.loads(line)
            cov[qid(r["i"])] = r["o"]
    return cov


def build(seconds: float, workers: int, frac: float):
    T = json.load(open(TRAIN_JSON))
    held = json.load(open(HELDOUT_DBS))
    cov = load_coverage()
    all_held = {qid(i): t for i, t in enumerate(T) if t["db_id"] in held}
    by_db = collections.defaultdict(list)
    for i, t in enumerate(T):
        if t["db_id"] in held:
            by_db[t["db_id"]].append((i, t["SQL"]))
    t0 = time.time()
    gold = exec_golds(by_db, seconds, workers)
    print(f"executed {len(gold)} ORIGINAL held-out golds in {time.time() - t0:.0f}s (limit {seconds}s, {workers} workers)")
    used = usage_map(all_held)

    print("\n== gold status per held-out DB (ORIGINAL BIRD-train SQL, data/raw/bird_train/train_databases) ==")
    print(f"{'db':14s} {'n':>5s} {'ok':>5s} {'error':>6s} {'timeout':>8s} {'patho':>6s}")
    for db in held:
        rs = [gold[x] for x in all_held if all_held[x]["db_id"] == db]
        c = collections.Counter(r["status"] for r in rs)
        npat = sum(1 for r in rs if r.get("pathology"))
        print(f"{db:14s} {len(rs):5d} {c['ok']:5d} {c['error']:6d} {c['timeout']:8d} {npat:6d}"
              + ("   EXCLUDED: " + EXCLUDED_DBS[db] if db in EXCLUDED_DBS else ""))
    errs = collections.Counter(gold[x]["error"][:50] for x in all_held
                               if gold[x]["status"] != "ok" and all_held[x]["db_id"] not in EXCLUDED_DBS)
    if errs:
        print("non-excluded gold failures:", dict(errs.most_common(10)))

    rows = []
    for x, t in all_held.items():
        g = gold[x]
        if t["db_id"] in EXCLUDED_DBS or g["status"] != "ok":
            continue
        rows.append({"id": x, "db_id": t["db_id"], "covered": cov.get(x) == "plan", "v3_outcome": cov.get(x),
                     "pathology": g["pathology"], "gold_sig": g["sig"], "gold_nrows": g["nrows"],
                     "gold_secs": g["secs"], "used": used.get(x, [])})
    rows.sort(key=lambda r: _idnum(r["id"]))
    dec, res = split(rows, frac)
    R = {r["id"]: r for r in rows}

    def summary(ids, name):
        c = collections.Counter(R[x]["db_id"] for x in ids)
        cv = sum(R[x]["covered"] for x in ids)
        pa = sum(R[x]["pathology"] is not None for x in ids)
        nv = sum(not R[x]["used"] for x in ids)
        print(f"{name:10s} n={len(ids):5d}  v3-covered {cv} ({100 * cv / max(1, len(ids)):.1f}%)  "
              f"gold pathology {pa}  never scored {nv}")
        return c

    print(f"\n== v2 universe: executable ORIGINAL gold, {len(held) - len(EXCLUDED_DBS)} DBs (retail_world excluded) ==")
    cu = summary([r["id"] for r in rows], "universe")
    cd = summary(dec, "decision")
    cr = summary(res, "reserved")
    print(f"\n{'db':14s} {'univ':>5s} {'dec':>5s} {'res':>5s} {'v3cov%':>7s} {'never':>6s} {'v1':>5s}")
    for db in sorted(cu, key=lambda d: -cu[d]):
        ids = [r["id"] for r in rows if r["db_id"] == db]
        cvp = 100 * sum(R[x]["covered"] for x in ids) / len(ids)
        nv = sum(not R[x]["used"] for x in ids)
        v1 = sum("v1_yardstick_1000" in R[x]["used"] for x in ids)
        print(f"{db:14s} {cu[db]:5d} {cd[db]:5d} {cr[db]:5d} {cvp:7.1f} {nv:6d} {v1:5d}")

    print("\n== prior use of universe questions (a question can be in several files) ==")
    lab = collections.Counter(l for r in rows for l in r["used"])
    for l, (rel, what) in USAGE_SOURCES.items():
        print(f"  {l:24s} {lab[l]:5d}  {rel}  ({what})")
    never = [r["id"] for r in rows if not r["used"]]
    print(f"  never scored by anything: {len(never)} of {len(rows)} "
          f"(v3 outcomes: {dict(collections.Counter(R[x]['v3_outcome'] for x in never))})")
    pr = collections.Counter(reserve_priority(R[x]["used"]) for x in res)
    print(f"  reserved by priority (0 never scored, 1 01-01 confirmation only, 2 scored outside v1, 3 in v1): {dict(sorted(pr.items()))}")
    v1_all = {x for x in all_held if "v1_yardstick_1000" in used.get(x, [])}
    print(f"  v1 yardstick held-out ids in train.json: {len(v1_all)} incl. retail_world "
          f"{sum(all_held[x]['db_id'] == 'retail_world' for x in v1_all)}; in v2 universe {len(v1_all & set(R))}")
    print("  v3 outcome of universe:", dict(collections.Counter(r["v3_outcome"] for r in rows)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {"train_json": (str(TRAIN_JSON.relative_to(REPO)), file_sha256(TRAIN_JSON)),
               "heldout_dbs": (str(HELDOUT_DBS.relative_to(REPO)), file_sha256(HELDOUT_DBS)),
               "coverage_v3": (str(COVERAGE_V3.relative_to(REPO)), file_sha256(COVERAGE_V3))}
    common = {"version": 2, "built": time.strftime("%Y-%m-%d"), "seed": SEED, "reserved_frac": frac,
              "gold": "ORIGINAL BIRD-train SQL (train.json), executed read-only", "gold_seconds": seconds,
              "excluded_dbs": EXCLUDED_DBS, "sources": {k: {"path": p, "sha256": h} for k, (p, h) in sources.items()},
              "match": "set equality of result rows (BIRD EX), first 100,000 rows"}
    for name, ids, rule in (("decision", dec, "use for every choice"),
                            ("reserved", res, "NEVER use for a decision; score once per final claim with --confirm")):
        man = dict(common, split=name, rule=rule, n=len(ids), ids_sha256=ids_sha256(ids),
                   database_counts=dict(collections.Counter(R[x]["db_id"] for x in ids)), ids=ids)
        (OUT_DIR / f"{name}.json").write_text(json.dumps(man, indent=1) + "\n")
    uni = dict(common, split="universe", n=len(rows), ids_sha256=ids_sha256(R),
               decision_sha256=ids_sha256(dec), reserved_sha256=ids_sha256(res),
               usage_sources={k: v[0] for k, v in USAGE_SOURCES.items()},
               dropped_gold={x: {"db_id": all_held[x]["db_id"], "status": gold[x]["status"], "error": gold[x].get("error")}
                             for x in all_held if all_held[x]["db_id"] not in EXCLUDED_DBS and gold[x]["status"] != "ok"},
               rows=rows)
    (OUT_DIR / "universe.json").write_text(json.dumps(uni, indent=1) + "\n")
    print(f"\nwrote {OUT_DIR.relative_to(REPO)}/{{decision,reserved,universe}}.json  "
          f"decision sha256 {ids_sha256(dec)[:16]}  reserved sha256 {ids_sha256(res)[:16]}")


# ---------------------------------------------------------------- scoring
def load_manifest(name: str) -> dict:
    m = json.load(open(OUT_DIR / f"{name}.json"))
    assert ids_sha256(m["ids"]) == m["ids_sha256"], f"{name}.json id list does not match its sha256"
    return m


def load_candidates(path) -> dict:
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in out:
            raise ValueError(f"duplicate id {r['id']} in {path}")
        out[r["id"]] = r.get("sql")
    return out


def _score_db(task):
    db, items, seconds = task  # items: [(id, gold_sql, cand_sql)]
    run = connect(db_path(db), seconds)
    out = []
    for x, gsql, csql in items:
        gs, gres = run_sql(run, gsql)
        if gs != "ok":
            out.append({"id": x, "ok": False, "gold": gs, "cand": None})
            continue
        if csql is None:
            out.append({"id": x, "ok": False, "gold": "ok", "cand": "missing"})
            continue
        cs, cres = run_sql(run, csql)
        ok = cs == "ok" and set(map(tuple, cres)) == set(map(tuple, gres))
        out.append({"id": x, "ok": bool(ok), "gold": "ok", "cand": cs})
    return out


def score_candidates(cands: dict, ids, seconds: float = 30.0, workers: int = 4) -> dict:
    """{id: sql | None} scored on `ids` (held-out train ids) -> {id: record with ok, gold, cand}. Library entry point."""
    T = json.load(open(TRAIN_JSON))
    by = collections.defaultdict(list)
    for x in ids:
        t = T[_idnum(x)]
        by[t["db_id"]].append((x, t["SQL"], cands.get(x)))
    tasks = [(db, it[j:j + 40], seconds) for db, it in by.items() for j in range(0, len(it), 40)]
    out = {}
    if workers <= 1:
        for t in tasks:
            for r in _score_db(t):
                out[r["id"]] = r
    else:
        with Pool(workers) as p:
            for res in p.imap_unordered(_score_db, tasks):
                for r in res:
                    out[r["id"]] = r
    return out


def report(res: dict, uni_rows: dict, title: str):
    ids = sorted(res, key=_idnum)
    n = len(ids)
    acc = lambda xs: (100 * sum(res[x]["ok"] for x in xs) / len(xs)) if xs else float("nan")
    print(f"{title}: EX {acc(ids):.2f} on n={n}  (missing {sum(res[x]['cand'] == 'missing' for x in ids)}, "
          f"cand errors {sum(res[x]['cand'] == 'error' for x in ids)}, timeouts {sum(res[x]['cand'] == 'timeout' for x in ids)}, "
          f"gold failures {sum(res[x]['gold'] != 'ok' for x in ids)})")
    cov = [x for x in ids if uni_rows[x]["covered"]]
    unc = [x for x in ids if not uni_rows[x]["covered"]]
    clean = [x for x in ids if uni_rows[x]["pathology"] is None]
    print(f"  v3-covered {acc(cov):.2f} (n={len(cov)})   uncovered {acc(unc):.2f} (n={len(unc)})   "
          f"gold-pathology-free {acc(clean):.2f} (n={len(clean)})")
    dbs = sorted({uni_rows[x]["db_id"] for x in ids})
    print("  per DB: " + "  ".join(f"{d} {acc([x for x in ids if uni_rows[x]['db_id'] == d]):.1f}"
                                    f"/{sum(uni_rows[x]['db_id'] == d for x in ids)}" for d in dbs))


def score_main(a):
    uni = json.load(open(OUT_DIR / "universe.json"))
    rows = {r["id"]: r for r in uni["rows"]}
    if a.split == "decision":
        ids = load_manifest("decision")["ids"]
    else:
        if not a.confirm:
            sys.exit("refusing to score the reserved set without --confirm (it is for one final confirmation per claim)")
        ids = load_manifest("reserved")["ids"] + (load_manifest("decision")["ids"] if a.split == "all" else [])
    cands = load_candidates(a.cands)
    extra = [x for x in cands if x not in rows]
    if extra:
        print(f"note: {len(extra)} candidate ids are outside the v2 universe and are ignored")
    res = score_candidates(cands, ids, a.seconds, a.workers)
    report(res, rows, f"{a.cands} [{a.split}]")
    if a.out:
        with open(a.out, "w") as fh:
            for x in sorted(res, key=_idnum):
                fh.write(json.dumps(dict(res[x], db_id=rows[x]["db_id"])) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seconds", type=float, default=30.0)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--reserved-frac", type=float, default=0.30)
    s = sub.add_parser("score")
    s.add_argument("cands")
    s.add_argument("--split", choices=["decision", "reserved", "all"], default="decision")
    s.add_argument("--confirm", action="store_true")
    s.add_argument("--seconds", type=float, default=30.0)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.seconds, a.workers, a.reserved_frac)
    else:
        score_main(a)


if __name__ == "__main__":
    main()
