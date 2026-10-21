"""Shared harness for the Phase-1 test 1c decode-time fixes 03-03 (anchor/Steiner coherence) and 06-04 (evidence
slot constraints) on the R1/R2 held-out head dumps (docs/BUDGET500_PLAN.md, frozen rules).

Only runs/phase1/p1s/dumps/{r1,r2}/heldout is read (never bird_dev). Evaluation set: held-out v2 DECISION ids
(data/sys1/heldout_v2/decision.json); a DECISION question not in the dump (grammar-uncovered) counts wrong in every arm.
Gold: execution match against the ORIGINAL BIRD-train gold SQL (train.json), read-only SQLite with a time limit, row
sets compared as sets (heldout_v2.score_candidates semantics). Results are cached per (db, sql) in the scratch dir.

BASELINE = Mac replay of the dump through the unchanged decoders (redecode(recorded=True, strict=False)).
ARM      = the same replay through a Scores wrapper that adds the arm's bonuses (paired, same machine).
Metrics: first pick = the greedy plan; oracle@8 = any of the lazy k-best-8 list right (primary) and any of the
flip pool (secondary, the production gen --k 8 list).
Parameter choice: leave-one-database-out over the 8 DECISION DBs, objective = greedy accuracy on the other 7 DBs
(of the same model), ties to the smallest total weight (the grid is ordered by it; the all-zero point is first).
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("LUTHOR_GRAMMAR", "v3")  # R1/R2 are grammar v3 (before the first luthor.sq import)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luthor.sq2 import targets as T  # noqa: E402

_V = json.loads((ROOT / "runs/r1r2/r2/vocab.json").read_text())  # R1 and R2 vocab.json are identical (same sha1)
T.set_vocab(_V["templates"], _V["nums"])

from luthor.sq2 import headdump as H  # noqa: E402
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

DUMPS = {"r2": ROOT / "runs/phase1/p1s/dumps/r2/heldout", "r1": ROOT / "runs/phase1/p1s/dumps/r1/heldout"}
DEC = json.loads((ROOT / "data/sys1/heldout_v2/decision.json").read_text())
IDS = list(DEC["ids"])
TRAIN = json.loads((ROOT / "data/raw/bird_train/train.json").read_text())
DB_ROOT = ROOT / "data/raw/bird_train/train_databases"
DB = {i: TRAIN[int(i.rsplit("_", 1)[1])]["db_id"] for i in IDS}
GOLD = {i: TRAIN[int(i.rsplit("_", 1)[1])]["SQL"] for i in IDS}
SCRATCH = Path(os.environ.get("P1C_CACHE", Path(tempfile.gettempdir()) / "p1c_cache"))
SECONDS = 30.0
WORKERS = int(os.environ.get("P1C_WORKERS", "6"))


def assert_heldout(path: Path) -> None:
    assert "bird_dev" not in str(path), "BIRD dev dumps are off-limits"


# ---- Scores wrapper ----------------------------------------------------------------------------------------------

class BiasScores:
    """targets.Scores over a DumpScores, with optional additive hooks. Hooks return a new list (never mutate)."""

    def __init__(self, base, ptr_fn=None, vals_fn=None, choice_fn=None):
        self.base, self.ptr_fn, self.vals_fn, self.choice_fn = base, ptr_fn, vals_fn, choice_fn
        self.option_ids, self.cands = base.option_ids, base.cands

    def choice(self, k):
        x = self.base.choice(k)
        return self.choice_fn(k, x) if self.choice_fn else x

    def ptr(self, s):
        x = self.base.ptr(s)
        return self.ptr_fn(s, x) if self.ptr_fn else x

    def pair(self, s, o, h):
        return self.base.pair(s, o, h)

    def vals(self, s, o):
        x = self.base.vals(s, o)
        return self.vals_fn(s, o, x) if self.vals_fn else x

    def limit(self, s):
        return self.base.limit(s)


def greedy_sql(sc, compile_fn) -> str | None:
    try:
        return compile_fn(T.decode(sc))
    except Exception:
        return None


def list_sqls(sc, compile_fn) -> dict:
    """greedy, flips (gen --k 8) and kbest8 SQL lists for Scores sc (same code as the box)."""
    out = H.run_lists(sc, compile_fn, 8, (8,))
    g = out.get("greedy")
    return {"greedy": g["sql"] if g else None,
            "flips": [x["sql"] for x in out.get("flips") or []],
            "kbest8": [x["sql"] for x in out.get("kbest8") or []]}


def load_dump(model: str) -> H.HeadDump:
    assert_heldout(DUMPS[model])
    return H.HeadDump(DUMPS[model])


def dec_questions(d: H.HeadDump):
    dec = set(IDS)
    for qd in d:
        if qd.id in dec:
            yield qd


# ---- execution with a persistent cache ---------------------------------------------------------------------------

def _norm(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _rows_key(rows) -> str:
    s = sorted({tuple(_norm(v) for v in r) for r in rows}, key=repr)
    return hashlib.md5(repr(s).encode()).hexdigest()


def _exec_db(task):
    db, sqls = task
    con = sqlite3.connect(f"file:{DB_ROOT / db / (db + '.sqlite')}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)
    out = []
    for sql in sqls:
        box["t"] = time.time() + SECONDS
        try:
            rows = con.execute(sql).fetchmany(100_000)
            out.append((db, sql, _rows_key(rows)))
        except Exception as ex:  # noqa: BLE001
            out.append((db, sql, "ERR:" + type(ex).__name__))
    con.close()
    return out


class Executor:
    """(db, sql) -> result key (md5 of the normalised row set, or ERR:...). Cached on disk (jsonl, append-only)."""

    def __init__(self, name: str = "exec"):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.path = SCRATCH / f"{name}.jsonl"
        self.cache: dict = {}
        if self.path.exists():
            for line in open(self.path):
                try:
                    db, sql, k = json.loads(line)
                    self.cache[(db, sql)] = k
                except Exception:
                    pass

    def run(self, pairs) -> None:
        todo = sorted({(db, s) for db, s in pairs if s and (db, s) not in self.cache})
        if not todo:
            return
        by = collections.defaultdict(list)
        for db, s in todo:
            by[db].append(s)
        tasks = [(db, ss[j:j + 25]) for db, ss in by.items() for j in range(0, len(ss), 25)]
        t0 = time.time()
        with open(self.path, "a") as fh, Pool(WORKERS) as p:
            for res in p.imap_unordered(_exec_db, tasks):
                for db, s, k in res:
                    self.cache[(db, s)] = k
                    fh.write(json.dumps([db, s, k]) + "\n")
        print(f"  executed {len(todo)} SQL in {time.time() - t0:.0f}s", flush=True)

    def gold_keys(self) -> dict:
        self.run([(DB[i], GOLD[i]) for i in IDS])
        return {i: self.cache[(DB[i], GOLD[i])] for i in IDS}

    def ok(self, qid: str, sql: str | None, gold: dict) -> bool:
        if not sql:
            return False
        g = gold[qid]
        if g.startswith("ERR:"):
            return False
        return self.cache.get((DB[qid], sql)) == g


# ---- LODO + stats ------------------------------------------------------------------------------------------------

def lodo_select(grid_hits: dict, grid_order: list) -> dict:
    """grid_hits: {param: {id: bool}} over all IDS. -> {db: param} chosen on the other DBs' greedy accuracy."""
    dbs = sorted(set(DB.values()))
    out = {}
    for d in dbs:
        best, bv = None, -1
        for p in grid_order:
            v = sum(grid_hits[p][i] for i in IDS if DB[i] != d)
            if v > bv:
                best, bv = p, v
        out[d] = best
    return out


def pct(h: dict) -> float:
    return 100.0 * sum(bool(h[i]) for i in IDS) / len(IDS)


def compare(label: str, a: dict, b: dict) -> str:
    c = cluster_signflip(a, b, DB)
    q = question_signflip(a, b)
    return (f"  {label:22s} base {pct(a):6.2f}  arm {pct(b):6.2f}  delta {c['delta']:+.2f}  DB sign-flip p={c['p']:.3f} "
            f"CI [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]  McNemar p={q['p']:.3f} (+{q['wins']}/-{q['losses']})"), c


def per_db(h: dict) -> str:
    out = []
    for d in sorted(set(DB.values())):
        ii = [i for i in IDS if DB[i] == d]
        out.append(f"{d}:{100 * sum(h[i] for i in ii) / len(ii):.1f}")
    return " ".join(out)


# ---- the generic experiment --------------------------------------------------------------------------------------

def run_experiment(model: str, make_arm, grid: list, ex: Executor, log) -> dict:
    """make_arm(qd, param) -> Scores (param None = baseline). Returns per-question hit dicts and selections."""
    t0 = time.time()
    d = load_dump(model)
    gold = ex.gold_keys()
    base_lists, grid_sql, qds = {}, {p: {} for p in grid}, []
    for qd in dec_questions(d):
        cf = qd.compile_fn()
        base_lists[qd.id] = list_sqls(qd.scores(recorded=True, strict=False), cf)
        for p in grid:
            grid_sql[p][qd.id] = greedy_sql(make_arm(qd, p), cf)
        qds.append(qd.id)
    log(f"[{model}] decoded baseline lists + {len(grid)} grid greedy for {len(qds)} DECISION questions in dump "
        f"({time.time() - t0:.0f}s)")
    pairs = [(DB[i], s) for i, L in base_lists.items() for s in [L["greedy"]] + L["flips"] + L["kbest8"]]
    pairs += [(DB[i], s) for p in grid for i, s in grid_sql[p].items()]
    ex.run(pairs)

    def hits_of(sqls: dict) -> dict:
        return {i: (i in sqls and ex.ok(i, sqls[i], gold)) for i in IDS}

    def any_of(lists: dict, name: str) -> dict:
        return {i: (i in lists and any(ex.ok(i, s, gold) for s in lists[i][name])) for i in IDS}

    base = {"greedy": hits_of({i: L["greedy"] for i, L in base_lists.items()}),
            "kbest8": any_of(base_lists, "kbest8"), "flips": any_of(base_lists, "flips")}
    gpu = {}
    for qd in dec_questions(d):  # reference only: the box's own recorded greedy
        g = qd.lists.get("greedy")
        gpu[qd.id] = g["sql"] if g else None
    ex.run([(DB[i], s) for i, s in gpu.items()])
    gpu_hits = hits_of(gpu)
    grid_hits = {p: hits_of(grid_sql[p]) for p in grid}
    sel = lodo_select(grid_hits, grid)
    # full lists for every parameter some fold chose
    chosen = sorted(set(sel.values()), key=grid.index)
    arm_lists = {p: {} for p in chosen}
    need_db = {p: {db for db, q in sel.items() if q == p} for p in chosen}
    for qd in dec_questions(d):
        for p in chosen:
            if DB[qd.id] in need_db[p]:
                arm_lists[p][qd.id] = list_sqls(make_arm(qd, p), qd.compile_fn())
    ex.run([(DB[i], s) for p in chosen for i, L in arm_lists[p].items() for s in [L["greedy"]] + L["flips"] + L["kbest8"]])
    merged = {}
    for p in chosen:
        merged.update(arm_lists[p])
    arm = {"greedy": hits_of({i: L["greedy"] for i, L in merged.items()}),
           "kbest8": any_of(merged, "kbest8"), "flips": any_of(merged, "flips")}
    # replay-vs-arm list identity (how many questions the arm changed at all)
    changed = sum(1 for i in merged if merged[i]["greedy"] != base_lists[i]["greedy"])
    changed_k = sum(1 for i in merged if merged[i]["kbest8"] != base_lists[i]["kbest8"])
    return {"model": model, "n_dump": len(qds), "base": base, "arm": arm, "gpu_greedy": gpu_hits,
            "grid_hits": grid_hits, "sel": sel, "changed_greedy": changed, "changed_kbest8": changed_k,
            "secs": time.time() - t0}


def report(res: dict, fmt_param, log) -> dict:
    m = res["model"]
    log(f"\n== {m.upper()} (DECISION n={len(IDS)}, {res['n_dump']} in dump; the rest count wrong) ==")
    log(f"  GPU recorded greedy (reference) {pct(res['gpu_greedy']):.2f};  Mac replay baseline greedy "
        f"{pct(res['base']['greedy']):.2f}")
    log("  grid (in-sample greedy acc on all 8 DBs; for information only, the arm uses LODO):")
    for p, h in res["grid_hits"].items():
        log(f"    {fmt_param(p):34s} {pct(h):6.2f}")
    log("  LODO choice per held-out DB: " + ", ".join(f"{d}={fmt_param(p)}" for d, p in sorted(res["sel"].items())))
    out = {}
    for name, lab in (("greedy", "first pick (greedy)"), ("kbest8", "oracle@8 k-best-8"), ("flips", "oracle@8 flip pool")):
        line, c = compare(lab, res["base"][name], res["arm"][name])
        log(line)
        out[name] = c
    log(f"  per-DB greedy base: {per_db(res['base']['greedy'])}")
    log(f"  per-DB greedy arm : {per_db(res['arm']['greedy'])}")
    log(f"  questions whose greedy SQL changed {res['changed_greedy']}, k-best-8 list changed {res['changed_kbest8']}; "
        f"runtime {res['secs']:.0f}s")
    return out


def gate(outs: dict, log) -> str:
    """Frozen gate on R2 (primary): first pick or oracle@8 >= +0.5 with DB sign-flip p < 0.10."""
    r2 = outs["r2"]
    ok = any(r2[k]["delta"] >= 0.5 and r2[k]["p"] < 0.10 for k in ("greedy", "kbest8"))
    v = "KEEP" if ok else "DROP"
    log(f"\nGATE (R2 primary; first pick or oracle@8 k-best-8 >= +0.5 with DB sign-flip p < 0.10): {v}")
    return v


class Log:
    def __init__(self, path: Path):
        self.fh = open(path, "w")

    def __call__(self, s: str = "") -> None:
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()
