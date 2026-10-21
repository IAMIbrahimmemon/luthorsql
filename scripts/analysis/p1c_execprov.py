"""Phase 1, test 1c (docs/BUDGET500_PLAN.md): two decode-time fixes on the R1/R2 held-out head dumps, Mac CPU.

Pre-declared design (written before any run; exactly ONE variant per problem, no parameters to fit):

Evaluation: held-out v2 DECISION ids (data/sys1/heldout_v2/decision.json, 796 questions, 8 DBs). Questions not in
the dump count wrong for every arm. Correct = BIRD EX against the ORIGINAL train.json gold SQL (set equality of
the first 100,000 rows, read-only SQLite, 30 s timeout for gold and candidates; same rule as heldout_v2.py).
Dumps: runs/phase1/p1s/dumps/{r2,r1}/heldout only (R2 primary, R1 secondary). bird_dev dumps are never opened.

BASELINE (both problems): Mac replay with the unmodified decoder, redecode(qd, recorded=True, strict=False):
greedy, flip pool (flips, k=8, = production candidate pool) and lazy k-best-8 (kbest8).
  first pick = greedy.  oracle@8 (primary list) = flips;  oracle@8 (secondary list) = kbest8.

PROBLEM 01-07 (MAP under an execution constraint), arm "execmap":
  live(sql) := runs without error/timeout and returns >= 1 row with a non-NULL value (topk.choose's rule).
  A question is ZERO-LIVE if no plan in the baseline flips U kbest8 is live. Only zero-live questions change:
  run the decoder's own lazy k-best on the same replay scores (topk.kbest(sc, 64, compile_fn), its default pop
  limit 8*k), walk it in score order and take the first live plan (= highest-scoring live plan in the k-best-64
  order). If found: first pick = that plan; the flip list and the kbest8 list each have their 8th (last) entry
  replaced by it (size stays 8). If none is live: everything unchanged. Non-zero-live questions are unchanged.
  (The same delta also applies to the first-live fallback pick, since on zero-live questions that fallback
  returns greedy.) No tuned parameter (K = 64 fixed a priori), so no leave-one-DB-out fit is needed.

PROBLEM 03-08 (value provenance mask), arm "provmask" (the doc's formal mask, attack idea 3):
  At decode time, whenever the decoder picks the literal(s) of a filter-like slot (*.filter, m.c0/c1, m.d0/d1;
  NOT m.having) with operator '=' or 'in', on a real column (not ROWS) with no function template, whose SQLite
  declared type has TEXT affinity (contains CHAR, CLOB or TEXT), every literal candidate that is a string, not
  a number and not date-shaped (^\\d{4}-\\d{2}) and whose lower(trim(value)) does not occur among
  lower(trim(column values)) gets its value score set to (min score - 1e4) (a hard mask; case/whitespace
  variants count as present, numbers and dates unconstrained). If every candidate would be masked, the scores
  are left unmasked. The mask sits under the decoders (a Scores wrapper; topk._Steer sees masked vectors), so
  greedy, the flip pool and k-best-8 are all re-decoded with it. No tuned parameter, so no LODO fit is needed.
  Note: the caller's wording "provenance in the question/evidence or the DB value index" is weaker; every
  candidate already comes from a question/evidence span or a DB lookup, so that rule would mask nothing. The
  column-level provenance of 03-08's formal statement is used.

Statistics: luthor.stats.cluster_signflip (exact DB sign-flip over the 8 DECISION DBs, 256 patterns, 95% CI by
inversion) and question_signflip. Gate (frozen): keep if first pick or oracle@8 improves >= +0.5 points with DB
sign-flip p < 0.10.

Run: LUTHOR_GRAMMAR=v3 .venv/bin/python scripts/analysis/p1c_execprov.py
Writes scripts/analysis/p1c_execmap.out, scripts/analysis/p1c_provmask.out and per-question rows
scripts/analysis/p1c_execprov_rows.jsonl.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("LUTHOR_GRAMMAR", "v3")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/analysis"))

from luthor.sqlw.execute import connect  # noqa: E402

DUMPS = {"r2": REPO / "runs/phase1/p1s/dumps/r2/heldout", "r1": REPO / "runs/phase1/p1s/dumps/r1/heldout"}
DECISION = REPO / "data/sys1/heldout_v2/decision.json"
TRAIN_JSON = REPO / "data/raw/bird_train/train.json"
DB_ROOT = REPO / "data/raw/bird_train/train_databases"
OUT = REPO / "scripts/analysis"
SECONDS = 30.0
K_DEEP = 64
assert all("bird_dev" not in str(p) for p in DUMPS.values())


def db_path(db):
    return str(DB_ROOT / db / f"{db}.sqlite")


def idnum(x):
    return int(re.search(r"(\d+)$", x).group(1))


# ---- execution ---------------------------------------------------------------------------------------------------

class Exec:
    def __init__(self, db):
        self.run = connect(db_path(db), SECONDS)
        self.cache = {}
        self.calls = 0

    def __call__(self, sql):
        """-> (status, rows | None)."""
        if not sql or not str(sql).strip():
            return ("error", None)
        if sql not in self.cache:
            self.calls += 1
            try:
                self.cache[sql] = ("ok", self.run(sql))
            except sqlite3.OperationalError as e:
                self.cache[sql] = ("timeout" if "interrupted" in str(e) else "error", None)
            except Exception:  # noqa: BLE001
                self.cache[sql] = ("error", None)
        return self.cache[sql]


def is_live(res):
    st, rows = res
    return st == "ok" and bool(rows) and any(v is not None for r in rows for v in r)


# ---- provenance mask ----------------------------------------------------------------------------------------------

_NUM = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")
_DATE = re.compile(r"^\d{4}-\d{2}")


class ColValues:
    def __init__(self, db):
        self.con = sqlite3.connect(f"file:{db_path(db)}?mode=ro", uri=True)
        self.con.text_factory = lambda b: b.decode("utf-8", "replace")
        self.types, self.vals = {}, {}

    def _q(self, x):
        return '"' + x.replace('"', '""') + '"'

    def is_text(self, table, col):
        key = (table.lower(), col.lower())
        if key not in self.types:
            t = None
            try:
                for r in self.con.execute(f"PRAGMA table_info({self._q(table)})"):
                    if r[1].lower() == col.lower():
                        u = (r[2] or "").upper()
                        t = any(s in u for s in ("CHAR", "CLOB", "TEXT"))
            except Exception:  # noqa: BLE001
                t = None
            self.types[key] = bool(t)
        return self.types[key]

    def values(self, table, col):
        key = (table.lower(), col.lower())
        if key not in self.vals:
            try:
                rows = self.con.execute(f"SELECT DISTINCT lower(trim(CAST({self._q(col)} AS TEXT))) FROM "
                                        f"{self._q(table)}").fetchall()
                self.vals[key] = {r[0] for r in rows if r[0] is not None}
            except Exception:  # noqa: BLE001
                self.vals[key] = None
        return self.vals[key]


def constrained(v):
    return isinstance(v, str) and not _NUM.match(v) and not _DATE.match(v.strip())


class ProvScores:
    """targets.Scores over another Scores; vals(slot + '|pm', o) is vals(slot, o) with the provenance mask."""

    def __init__(self, base, cv: ColValues):
        self.base, self.cv = base, cv
        self.option_ids, self.cands = base.option_ids, base.cands
        self.masked_calls = 0

    def choice(self, k):
        return self.base.choice(k)

    def ptr(self, s):
        return self.base.ptr(s)

    def pair(self, s, o, h):
        return self.base.pair(s, o, h)

    def limit(self, s):
        return self.base.limit(s)

    def vals(self, s, o):
        if not s.endswith("|pm"):
            return self.base.vals(s, o)
        xs = list(self.base.vals(s[:-3], o))
        if not xs:
            return xs
        t, c = self.option_ids[int(o)].split(".", 1)
        dom = self.cv.values(t, c)
        if dom is None:
            return xs
        bad = [i for i, cd in enumerate(self.cands) if constrained(cd.value) and
               cd.value.strip().lower() not in dom]
        if not bad or len(bad) == len(xs):
            return xs
        lo = min(xs) - 1e4
        for i in bad:
            xs[i] = lo
        self.masked_calls += 1
        return xs


def install_mask():
    """Patch targets._pick_values so eligible filter literals read the masked vector (slot + '|pm')."""
    from luthor.sq2 import targets as T
    from luthor.sq.plan import ROWS
    orig = T._pick_values

    def pick(sc, slot, opt, op):
        ids = sc.option_ids
        eligible = (op in ("=", "in") and slot != "m.having" and not ids[opt].endswith(f".{ROWS}")
                    and _prov_of(sc) is not None)
        if eligible and "fn" in T.PAIRS:
            f = sc.pair(slot, opt, "fn")
            if T.VOCAB["templates"][max(range(len(f)), key=lambda i: f[i])]:
                eligible = False
        if eligible:
            t, c = ids[opt].split(".", 1)
            eligible = _prov_of(sc).cv.is_text(t, c)
        return orig(sc, slot + "|pm" if eligible else slot, opt, op)

    T._pick_values = pick
    return orig


def _prov_of(sc):
    """The ProvScores under a topk._Steer (or itself)."""
    x = sc
    for _ in range(3):
        if isinstance(x, ProvScores):
            return x
        x = getattr(x, "base", None)
    return None


# ---- per DB worker ------------------------------------------------------------------------------------------------

def entries(lst):
    lst = [lst] if isinstance(lst, dict) else (lst or [])
    return [x["sql"] for x in lst]


def work(task):
    run_name, db, ids = task
    from luthor.sq2.headdump import HeadDump, redecode, run_lists
    from luthor.sq2.topk import kbest
    from luthor.sq2 import targets as T
    train = json.load(open(TRAIN_JSON))
    hd = HeadDump(DUMPS[run_name])
    ex = Exec(db)
    cv = ColValues(db)
    out = []
    orig_pick = T._pick_values
    for qid in sorted(ids, key=lambda i: hd.where[i]):
        t0 = time.time()
        gold = train[idnum(qid)]["SQL"]
        gst, grows = ex(gold)
        gset = set(map(tuple, grows)) if gst == "ok" else None

        def correct(sql):
            if gset is None or not sql:
                return False
            st, rows = ex(sql)
            return st == "ok" and set(map(tuple, rows)) == gset

        qd = hd.question(qid)
        base, _ = redecode(qd, recorded=True, strict=False)
        b_g = base["greedy"]["sql"] if base.get("greedy") else None
        b_fl, b_kb = entries(base.get("flips")), entries(base.get("kbest8"))
        pool = [s for s in dict.fromkeys([b_g] + b_fl + b_kb) if s]
        live = {s: is_live(ex(s)) for s in pool}
        zero_live = not any(live.values())
        n_exec_before = ex.calls
        # ---- 01-07 arm
        found, found_rank, deep_n = None, None, 0
        if zero_live:
            ds = qd.scores(recorded=True, strict=False)
            try:
                deep = kbest(ds, K_DEEP, qd.compile_fn(), with_cost=False)
            except Exception:  # noqa: BLE001
                deep = []
            deep_n = len(deep)
            cf = qd.compile_fn()
            for r, p in enumerate(deep):
                try:
                    s = cf(p)
                except Exception:  # noqa: BLE001
                    continue
                if s in live:
                    continue
                if is_live(ex(s)):
                    found, found_rank = s, r
                    break
        deep_execs = ex.calls - n_exec_before
        e_g = found if found else b_g
        e_fl = (b_fl[:7] + [found]) if found else b_fl
        e_kb = (b_kb[:7] + [found]) if found else b_kb
        # ---- 03-08 arm
        ps = ProvScores(qd.scores(recorded=True, strict=False), cv)
        install_mask()
        try:
            arm = run_lists(ps, qd.compile_fn(), 8, (8,))
        finally:
            T._pick_values = orig_pick
        p_g = arm["greedy"]["sql"] if arm.get("greedy") else None
        p_fl, p_kb = entries(arm.get("flips")), entries(arm.get("kbest8"))
        rec = {"run": run_name, "id": qid, "db": db, "gold_status": gst, "zero_live": zero_live,
               "greedy_live": bool(b_g and live.get(b_g)),
               "base": {"first": correct(b_g), "flips": any(correct(s) for s in b_fl),
                        "kbest8": any(correct(s) for s in b_kb)},
               "execmap": {"first": correct(e_g), "flips": any(correct(s) for s in e_fl),
                           "kbest8": any(correct(s) for s in e_kb), "found": found is not None,
                           "found_rank": found_rank, "deep_n": deep_n, "deep_execs": deep_execs},
               "provmask": {"first": correct(p_g), "flips": any(correct(s) for s in p_fl),
                            "kbest8": any(correct(s) for s in p_kb), "masked_calls": ps.masked_calls,
                            "greedy_changed": p_g != b_g, "greedy_live": bool(p_g and is_live(ex(p_g))),
                            "flips_changed": p_fl != b_fl, "kbest_changed": p_kb != b_kb},
               "secs": round(time.time() - t0, 3)}
        out.append(rec)
    return out


# ---- report -------------------------------------------------------------------------------------------------------

def report(rows, dec_ids, dbmap, arm, title, runtime):
    from luthor.stats import cluster_signflip, question_signflip
    lines = [title, ""]
    for run_name in ("r2", "r1"):
        R = {r["id"]: r for r in rows if r["run"] == run_name}
        lines.append(f"== {run_name.upper()} ({'primary' if run_name == 'r2' else 'secondary'}): "
                     f"{len(R)} of {len(dec_ids)} DECISION questions in the dump (rest count wrong for both)")
        for metric, name in (("first", "first pick (greedy)"), ("flips", "oracle@8 flip pool"),
                             ("kbest8", "oracle@8 k-best-8")):
            a = {i: bool(R[i]["base"][metric]) if i in R else False for i in dec_ids}
            b = {i: bool(R[i][arm][metric]) if i in R else False for i in dec_ids}
            cs, qs = cluster_signflip(a, b, dbmap), question_signflip(a, b)
            ba, aa = 100 * sum(a.values()) / len(a), 100 * sum(b.values()) / len(b)
            lines.append(f"  {name:22s} base {ba:6.2f}  arm {aa:6.2f}  delta {cs['delta']:+.2f}  "
                         f"DB sign-flip p={cs['p']:.4f} CI95 [{cs['ci'][0]:+.2f}, {cs['ci'][1]:+.2f}]  "
                         f"question sign-flip p={qs['p']:.4f} (wins {qs['wins']}, losses {qs['losses']})")
            lines.append("      per-DB D_g: " + ", ".join(f"{k} {int(v):+d}" for k, v in cs["per_db"].items()))
        n = len(R)
        if arm == "execmap":
            z = [r for r in R.values() if r["zero_live"]]
            f = [r for r in z if r["execmap"]["found"]]
            lines.append(f"  greedy live {sum(r['greedy_live'] for r in R.values())}/{n}; zero-live (flips U kbest8) "
                         f"{len(z)}/{n}; deep k-best-{K_DEEP} found a live plan for {len(f)}; of those correct "
                         f"{sum(r['execmap']['first'] for r in f)}; baseline greedy correct on zero-live "
                         f"{sum(r['base']['first'] for r in z)}")
            if z:
                lines.append(f"  deep list length mean {sum(r['execmap']['deep_n'] for r in z) / len(z):.1f}; "
                             f"SQL executions (calls to g) on zero-live: mean {sum(r['execmap']['deep_execs'] for r in z) / len(z):.1f}, "
                             f"max {max(r['execmap']['deep_execs'] for r in z)}; found ranks "
                             f"{sorted(r['execmap']['found_rank'] for r in f)}")
        else:
            m = [r for r in R.values() if r["provmask"]["masked_calls"]]
            lines.append(f"  questions with >=1 masked value vector {len(m)}/{n}; greedy SQL changed "
                         f"{sum(r['provmask']['greedy_changed'] for r in R.values())}; flips changed "
                         f"{sum(r['provmask']['flips_changed'] for r in R.values())}; kbest8 changed "
                         f"{sum(r['provmask']['kbest_changed'] for r in R.values())}; greedy live base "
                         f"{sum(r['greedy_live'] for r in R.values())} -> arm {sum(r['provmask']['greedy_live'] for r in R.values())}")
        lines.append("")
    lines.append(f"runtime (both arms, both runs, wall): {runtime:.0f} s")
    return "\n".join(lines) + "\n"


def gate_lines(rows, dec_ids, dbmap, arm):
    from luthor.stats import cluster_signflip
    R = {r["id"]: r for r in rows if r["run"] == "r2"}
    out = []
    for metric in ("first", "flips", "kbest8"):
        a = {i: bool(R[i]["base"][metric]) if i in R else False for i in dec_ids}
        b = {i: bool(R[i][arm][metric]) if i in R else False for i in dec_ids}
        cs = cluster_signflip(a, b, dbmap)
        out.append((metric, cs["delta"], cs["p"], cs["delta"] >= 0.5 and cs["p"] < 0.10))
    keep = any(x[3] for x in out)
    s = "GATE (R2, frozen: >= +0.5 and DB p < 0.10 on first pick or oracle@8): " + ("KEEP" if keep else "DROP")
    s += "  [" + "; ".join(f"{m} {d:+.2f} p={p:.4f}" for m, d, p, _ in out) + "]"
    return s + "\n"


def main():
    t0 = time.time()
    dec = json.load(open(DECISION))["ids"]
    train = json.load(open(TRAIN_JSON))
    dbmap = {i: train[idnum(i)]["db_id"] for i in dec}
    from luthor.sq2.headdump import HeadDump
    tasks = []
    for run_name, path in DUMPS.items():
        have = set(HeadDump(path).ids)
        by = {}
        for i in dec:
            if i in have:
                by.setdefault(dbmap[i], []).append(i)
        tasks += [(run_name, db, ids) for db, ids in by.items()]
    tasks.sort(key=lambda t: -len(t[2]))
    rows = []
    with Pool(int(os.environ.get("P1C_WORKERS", "4"))) as p:
        for res in p.imap_unordered(work, tasks):
            rows += res
            print(f"done {res[0]['run'] if res else '?'} {res[0]['db'] if res else '?'} n={len(res)} "
                  f"t={time.time() - t0:.0f}s", flush=True)
    rt = time.time() - t0
    with open(OUT / "p1c_execprov_rows.jsonl", "w") as fh:
        for r in sorted(rows, key=lambda r: (r["run"], idnum(r["id"]))):
            fh.write(json.dumps(r) + "\n")
    for arm, title in (("execmap", "01-07 MAP decoding under an execution constraint (k-best-64 first-live on zero-live questions)"),
                       ("provmask", "03-08 value provenance mask (decode-time, text '='/'in' literals must occur in the column)")):
        txt = report(rows, dec, dbmap, arm, title + "\nheld-out v2 DECISION (796 q, 8 DBs); baseline = Mac replay, "
                     "unmodified decoder; arm = same replay with the change (paired)", rt)
        txt += gate_lines(rows, dec, dbmap, arm)
        (OUT / f"p1c_{arm}.out").write_text(txt)
        print(txt)


if __name__ == "__main__":
    main()
