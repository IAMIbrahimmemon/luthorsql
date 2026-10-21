"""luthor.stats: the DB-level sign-flip test has exact size under its null (checked by full enumeration on a small
design), its interval inverts the test, and McNemar matches the binomial formula."""
import itertools
import math
import random

import numpy as np
import pytest

from luthor.stats import cluster_signflip, question_signflip


def _design(G=6, per=5, seed=0):
    rnd = random.Random(seed)
    ids = [f"q{g}_{k}" for g in range(G) for k in range(per)]
    db = {i: i.split("_")[0] for i in ids}
    a = {i: rnd.random() < 0.5 for i in ids}
    b = {i: (not a[i]) if rnd.random() < 0.3 else a[i] for i in ids}
    return a, b, db


def test_exact_size_under_db_sign_symmetry():
    """Under H0 (each DB's total is equally likely +D_g or -D_g), P(p <= alpha) <= alpha exactly: enumerate all
    2^G relabelings (swap A/B inside a whole DB)."""
    a, b, db = _design()
    dbs = sorted(set(db.values()))
    for alpha in (0.05, 0.1, 0.25):
        rej = 0
        for flips in itertools.product((False, True), repeat=len(dbs)):
            sw = {d for d, f in zip(dbs, flips) if f}
            a2 = {i: (b[i] if db[i] in sw else a[i]) for i in a}
            b2 = {i: (a[i] if db[i] in sw else b[i]) for i in a}
            rej += cluster_signflip(a2, b2, db, grid_step=0.01)["p"] <= alpha
        assert rej / 2 ** len(dbs) <= alpha + 1e-12


def test_interval_inverts_the_test():
    a, b, db = _design(G=11, per=40, seed=3)
    r = cluster_signflip(a, b, db)
    lo, hi = r["ci"]
    assert lo <= r["delta"] <= hi
    assert (r["p"] < 0.05) == (lo > 0 or hi < 0)


def test_identical_systems():
    a, _, db = _design()
    r = cluster_signflip(a, a, db)
    assert r["delta"] == 0 and r["p"] == pytest.approx(1.0)
    assert question_signflip(a, a)["p"] == 1.0


def test_mcnemar_matches_binomial():
    ids = [str(i) for i in range(40)]
    a = {i: False for i in ids}
    b = {i: int(i) < 9 for i in ids}  # 9 wins, 0 losses
    a.update({"39": True})            # 1 loss
    r = question_signflip(a, b)
    assert (r["wins"], r["losses"]) == (9, 1)
    assert r["p"] == pytest.approx(2 * sum(math.comb(10, j) for j in range(2)) / 2 ** 10)
