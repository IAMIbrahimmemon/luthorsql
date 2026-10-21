"""Phase 1, test 1c — research problem 06-04 (evidence text as slot constraints).

PRE-DECLARED VARIANT (the only one; written before any run):
  Parse the BIRD evidence with the existing rule compiler (reports/fanout_r1/evidence/r1/evcomp.py,
  compile_evidence; BIRD-train precision/recall in reports/fanout_r1/evidence/r1/prec_recall.out) against the dump's
  schema card, with one addition: every column name is also registered under its snake_case alias (spaces -> "_"),
  so "character_name" reaches "Character Name". Its output becomes soft constraints, added as score bonuses inside
  decoding (a Scores wrapper; the unchanged greedy / flip-pool / lazy k-best-8 decoders run on the biased scores):

  1. (column, value) filter pairs  ("X refers to col = 'v'", IN / BETWEEN / LIKE / IS NULL forms):
       - pointer bonus +beta_c on every column option named `col` (any table) in the filter-type slots
         (filter, c0/c1, d0/d1, having) of every block;
       - value bonus +beta_v on every literal candidate whose evcomp.norm_val equals one of the pair's values, in the
         value scores of an option named `col` (any slot, including formula value roles |tK).
  2. Columns named in the evidence formulas / right-hand sides that are not filter-pair columns
     ("tallest refers to max(height)"): pointer bonus +beta_c on options named so in the projection slots (p0..p3).
  3. Function: evidence that is a percentage (evcomp pct) -> +beta_f on compose choices 'percent' and every compose
     template with a division and a numeric constant; evidence that divides but is not a percentage (evcomp div) ->
     +beta_f on compose 'ratio' and every compose template with a division and no numeric constant.
  Row pseudo-options are never biased. Parameters: beta_c in {0, 1, 2, 4}, beta_v in {0, 2, 4, 8},
  beta_f in {0, 2, 4, 8} (logit units; median top-1 gaps: pointer ~2.5, value ~14, choice ~10). Chosen by
  leave-one-database-out over the 8 DECISION DBs on greedy accuracy of the same model; ties to the smallest total
  weight. (0, 0, 0) is the baseline itself.
  Gate (frozen): keep if R2 first pick or oracle@8 improves >= +0.5 with DB sign-flip p < 0.10; else drop, and no
  second variant on this split.

  .venv/bin/python scripts/analysis/p1c_evidence_slots.py      -> scripts/analysis/p1c_evidence_slots.out
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p1c_ae_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "reports/fanout_r1/evidence/r1"))
import evcomp  # noqa: E402
from luthor.sq.plan import ROWS  # noqa: E402

BC = (0.0, 1.0, 2.0, 4.0)
BV = (0.0, 2.0, 4.0, 8.0)
BF = (0.0, 2.0, 4.0, 8.0)
GRID = sorted(((c, v, f) for c in BC for v in BV for f in BF), key=lambda p: (sum(p), p))
FILT = re.compile(r"\.(filter|c\d|d\d|having)$")
PROJ = re.compile(r"\.p\d$")

COMPOSE = list(C.T.MAIN_CHOICES["compose"])
PCT_IDX = [i for i, n in enumerate(COMPOSE) if n == "percent" or ("/" in n and "{N" in n)]
DIV_IDX = [i for i, n in enumerate(COMPOSE) if n == "ratio" or ("/" in n and "{N" not in n)]

_SCHEMA: dict = {}


def schema(model: str, db: str) -> dict:
    key = (model, db)
    if key not in _SCHEMA:
        s = evcomp.load_schema(C.DUMPS[model] / "cards" / f"{db}.json")
        alias = {}
        for c in list(s["cols"]):
            a = re.sub(r"\s+", "_", c.strip())
            if a != c and a not in s["cols"]:
                s["cols"][a] = s["cols"][c]
                alias[a] = c
        s["alias"] = alias
        _SCHEMA[key] = s
    return _SCHEMA[key]


class Prep:
    def __init__(self, qd, model):
        s = schema(model, qd.db_id)
        ev = evcomp.compile_evidence(qd.meta.get("context") or "", s)
        canon = lambda c: s["alias"].get(c, c)
        self.fcols, self.vals = set(), {}
        for col, _op, vals, _src in ev["pairs"]:
            c = canon(col)
            self.fcols.add(c)
            if vals:
                self.vals.setdefault(c, set()).update(vals)
        self.pcols = {canon(c) for c in ev["cols"]} - self.fcols
        self.pct, self.div = bool(ev["pct"]), bool(ev["div"])
        ids = qd.option_ids
        self.colname = [None if o.endswith("." + ROWS) else o.split(".", 1)[1].lower() for o in ids]
        self.cnorm = [evcomp.norm_val(c.value) for c in qd.cands]
        self.empty = not (self.fcols or self.pcols or self.pct or self.div)


_PREP: dict = {}
MODEL = {"m": "r2"}


def make_arm(qd, p):
    base = qd.scores(recorded=True, strict=False)
    if p is None or not any(p):
        return base
    bc, bv, bf = p
    pr = _PREP.get(qd.id)
    if pr is None:
        pr = _PREP[qd.id] = Prep(qd, MODEL["m"])
    if pr.empty:
        return base

    def ptr_fn(s, x):
        if not bc:
            return x
        want = pr.fcols if FILT.search(s) else pr.pcols if PROJ.search(s) else None
        if not want:
            return x
        return [v + bc if pr.colname[j] in want else v for j, v in enumerate(x)]

    def vals_fn(s, o, x):
        if not bv:
            return x
        want = pr.vals.get(pr.colname[int(o)])
        if not want:
            return x
        return [v + bv if pr.cnorm[i] in want else v for i, v in enumerate(x)]

    def choice_fn(k, x):
        if not bf or k != "m.compose":
            return x
        idx = PCT_IDX if pr.pct else DIV_IDX if pr.div else None
        if not idx:
            return x
        y = list(x)
        for i in idx:
            if i < len(y):
                y[i] += bf
        return y

    return C.BiasScores(base, ptr_fn=ptr_fn, vals_fn=vals_fn, choice_fn=choice_fn)


def fmt(p):
    return f"beta_c={p[0]:g} beta_v={p[1]:g} beta_f={p[2]:g}"


def main():
    out = HERE / "p1c_evidence_slots.out"
    log = C.Log(out)
    log(__doc__.split("  .venv")[0].rstrip())
    log(f"\nrun {time.strftime('%Y-%m-%d %H:%M:%S')}; Mac CPU; grid {len(GRID)} points; exec time limit {C.SECONDS}s")
    ex = C.Executor("exec_ev")
    outs = {}
    t0 = time.time()
    for m in ("r2", "r1"):
        _PREP.clear()
        MODEL["m"] = m
        d = C.load_dump(m)
        n = nf = nv = nc = 0
        for qd in C.dec_questions(d):
            pr = Prep(qd, m)
            n += 1
            nf += bool(pr.fcols)
            nv += bool(pr.vals)
            nc += pr.pct or pr.div
        log(f"[{m}] parser coverage on the {n} DECISION questions in the dump: filter-pair column {nf}, "
            f"with values {nv}, percentage/division {nc}")
        res = C.run_experiment(m, make_arm, GRID, ex, log)
        outs[m] = C.report(res, fmt, log)
    C.gate(outs, log)
    log(f"total runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
