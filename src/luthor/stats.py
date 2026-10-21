"""Paired tests for "system B beats system A" on a benchmark split into few databases (BIRD dev: 11).

The percentile DB-cluster bootstrap used until round R3 is anti-conservative with 11 clusters: under exact nulls
built on real discordance patterns its nominal-5% interval excludes 0 in 7-12% of simulations
(docs/research_problems/07_selection_repair_eval.md, 07-07). This module replaces it with exact randomization tests.

  d_i = y_i^B - y_i^A in {-1, 0, 1};  D_g = sum of d_i over database g (n_g questions);  delta = sum_i d_i / N.

* cluster_signflip (primary): H0 = the per-database totals D_g are sign-symmetric about 0 (independent across
  databases). Every one of the 2^G sign patterns is enumerated (G = 11: 2,048), so the p-value is exact; it stays
  valid under any within-database dependence. The confidence interval inverts the test for a shift of delta0 per
  question: D_g(delta0) = D_g - n_g * delta0. Reads as "would generalise to new databases like these".
* question_signflip: H0 = each discordant question is equally likely to favour A or B (exact McNemar). Sharper,
  but conditional on these questions: it says nothing about new databases. Report it as such.

Smallest attainable two-sided p with G = 11 is 2/2048 ~ 0.001. Power is the price of honesty: with 11 databases a
+0.5-point change is not detectable on BIRD dev alone (see 07-07); pool more databases for that.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np


def _clusters(a: dict, b: dict, db: dict):
    ids = sorted(a)
    assert set(ids) == set(b), "both systems must cover the same questions"
    D, n = defaultdict(float), defaultdict(int)
    for i in ids:
        D[db[i]] += int(bool(b[i])) - int(bool(a[i]))
        n[db[i]] += 1
    keys = sorted(D)
    return np.array([D[k] for k in keys]), np.array([n[k] for k in keys], float), keys


def _signs(G: int, max_exact: int = 16, draws: int = 200_000, seed: int = 0) -> np.ndarray:
    if G <= max_exact:
        return np.array(list(itertools.product((-1.0, 1.0), repeat=G)))
    rng = np.random.default_rng(seed)  # Monte Carlo beyond 2^16 patterns (+ identity, so p > 0)
    return np.vstack([np.ones(G), rng.choice((-1.0, 1.0), size=(draws, G))])


def cluster_signflip(a: dict, b: dict, db: dict, alpha: float = 0.05, grid_step: float = 0.0001) -> dict:
    """Exact DB-level sign-flip test and its inverted confidence interval (delta in points)."""
    D, n, keys = _clusters(a, b, db)
    N = n.sum()
    S = _signs(len(D))
    obs = D.sum()
    p = float(np.mean(np.abs(S @ D) >= abs(obs) - 1e-9))

    def pval(d0):  # two-sided p for H0: shift d0 per question
        x = D - n * d0
        return np.mean(np.abs(S @ x) >= abs(x.sum()) - 1e-9)

    est = obs / N
    lo_lim = -1.0
    hi_lim = 1.0
    grid = np.arange(lo_lim, hi_lim + grid_step / 2, grid_step)
    # p(d0) is unimodal around est in practice; scan outward from est for the first rejected value on each side
    inside = [g for g in _scan(grid, est, pval, alpha)]
    lo, hi = (min(inside), max(inside)) if inside else (est, est)
    return {"delta": 100 * est, "p": p, "ci": (100 * lo, 100 * hi), "G": len(D), "patterns": len(S),
            "min_p": 2 / len(S) if len(S) == 2 ** len(D) else None, "per_db": dict(zip(keys, D.tolist()))}


def _scan(grid, est, pval, alpha):
    i0 = int(np.searchsorted(grid, est))
    out = []
    for rng_ in (range(i0, len(grid)), range(i0 - 1, -1, -1)):
        for i in rng_:
            if pval(grid[i]) >= alpha:  # alpha level two-sided: keep if not rejected
                out.append(grid[i])
            else:
                break
    return out


def question_signflip(a: dict, b: dict) -> dict:
    """Exact McNemar (question-level sign flip): two-sided binomial test on discordant questions."""
    ids = sorted(a)
    w = sum(1 for i in ids if b[i] and not a[i])
    l_ = sum(1 for i in ids if a[i] and not b[i])
    m = w + l_
    if m == 0:
        return {"delta": 0.0, "p": 1.0, "wins": 0, "losses": 0}
    k = min(w, l_)
    tail = sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m
    return {"delta": 100 * (w - l_) / len(ids), "p": min(1.0, 2 * tail), "wins": w, "losses": l_}


def percentile_bootstrap(a: dict, b: dict, db: dict, B: int = 2000, seed: int = 0) -> tuple:
    """The old DB-cluster percentile interval (kept only for comparison; anti-conservative at G = 11)."""
    D, n, _ = _clusters(a, b, db)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), size=(B, len(D)))
    est = D[idx].sum(1) / n[idx].sum(1) * 100
    return tuple(np.percentile(est, [2.5, 97.5]).tolist())
