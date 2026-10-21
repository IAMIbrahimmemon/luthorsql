"""Frozen System-1 pick rules (research note 10_candidate_shaping, PREREG runs/phase1/pickrules/PREREG.md).

Pure functions over one pool row's candidate list (`row["cands"]`, as written by the repair box / VALHARD scoring:
each candidate has sql, rr (validator score), runs, nonnull, res (result summary), cost (generator -logprob gap to its
own rank 0), src (None / r1 / both = generator list, k32 = k32 rescue, repair = constrained repair)). Nothing here is
fitted; the only constant is GAMMA = 0.2, frozen 2026-10-11 from DECISION LODO.

  pick_max(cands)      VALHARD baseline: max rr over live candidates
  pick_r2(cands)       R2: drop degenerate scalar 0 / NULL results first (unless every live candidate is one), then max rr
  pick_r1(cands)       R1: max of rr - GAMMA * generator cost (repair candidates get the question's max non-repair cost)
  pick_r1r2(cands)     R2 then R1 (the pre-registered bundle)

Live = runs and nonnull; if no candidate is live, all candidates are live (as in repair_eval.py / candidate_shaping.py).
Candidates without SQL are ignored; a missing rr counts as -1e9. Ties go to the earlier candidate in list order
(Python max). Inputs are never mutated. Semantics are identical to candidate_shaping.pick_max / pick_prune(drop_degen)
/ pick_cost(gamma, "cap", degen).
"""
from __future__ import annotations

import re

GAMMA = 0.2
MISSING_RR = -1e9
DEGEN = re.compile(r"^1 row x 1 col \((integer|real|null)\); first: \((0|0\.0|None),?\)$")


def _rr(c: dict) -> float:
    r = c.get("rr")
    return MISSING_RR if r is None else float(r)


def candidates(cands: list) -> list:
    return [c for c in cands if c.get("sql")]


def live(cands: list) -> list:
    cs = candidates(cands)
    L = [c for c in cs if c.get("runs") and c.get("nonnull")]
    return L or cs


def degenerate(c: dict) -> bool:
    """A single-row single-column result whose value is 0, 0.0 or NULL."""
    return bool(DEGEN.match(c.get("res") or ""))


def drop_degenerate(L: list) -> list:
    return [c for c in L if not degenerate(c)] or L


def gen_cost(c: dict, cands: list) -> float:
    """Generator cost; repair candidates (own constrained list, other scale) get the max non-repair cost."""
    if c.get("src") == "repair":
        return max((x.get("cost", 0.0) or 0.0 for x in candidates(cands) if x.get("src") != "repair"), default=0.0)
    return c.get("cost", 0.0) or 0.0


def _argmax(L: list, key):
    return max(L, key=key) if L else None


def pick_max(cands: list):
    return _argmax(live(cands), _rr)


def pick_r2(cands: list):
    return _argmax(drop_degenerate(live(cands)), _rr)


def pick_r1(cands: list, gamma: float = GAMMA):
    return _argmax(live(cands), lambda c: _rr(c) - gamma * gen_cost(c, cands))


def pick_r1r2(cands: list, gamma: float = GAMMA):
    return _argmax(drop_degenerate(live(cands)), lambda c: _rr(c) - gamma * gen_cost(c, cands))


RULES = {"max": pick_max, "r2": pick_r2, "r1": pick_r1, "r1r2": pick_r1r2}


def pick(row: dict, rule: str):
    """Apply a named rule to one pool row ({"id", "cands": [...]}); returns the picked candidate or None."""
    return RULES[rule](row.get("cands") or [])
