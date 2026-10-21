"""luthor.contamination: the prefix-filtered index returns exactly what a brute-force Jaccard scan returns, the
importable signature near_dup(questions, ref, threshold) behaves, and the dev-database checks catch name and schema
copies."""
import random

import pytest

from luthor import contamination as C


def _brute(qs, ref, t, n=5, unit="char"):
    rs = [C.shingles(r, n, unit) for r in ref]
    return [any(C.jaccard(C.shingles(q, n, unit), b) >= t for b in rs) for q in qs]


def _corpus(seed, k):
    rnd = random.Random(seed)
    vocab = "how many what which list the of in average highest players schools county name id total year 2010 user".split()
    return [" ".join(rnd.choice(vocab) for _ in range(rnd.randint(1, 12))) + rnd.choice(["?", "", "."]) for _ in range(k)]


@pytest.mark.parametrize("t", [0.2, 0.4, 0.6, 0.85, 1.0])
@pytest.mark.parametrize("unit,n", [("char", 5), ("char", 3), ("word", 2)])
def test_index_matches_brute_force(t, unit, n):
    qs, ref = _corpus(1, 300), _corpus(2, 80)
    assert C.near_dup(qs, ref, t, n=n, unit=unit) == _brute(qs, ref, t, n, unit)


def test_signature_and_normalisation():
    ref = ["What is the highest eligible free rate for K-12 students in Alameda County?"]
    qs = ["what is the HIGHEST eligible free rate, for K-12 students in Alameda County",
          "List the names of drivers who won in 2009.", ""]
    assert C.near_dup(qs, ref) == [True, False, False]
    assert C.near_dup(qs, ref, threshold=1.0)[0] is True  # punctuation/case only -> identical shingles
    assert C.near_dup([], ref) == []
    assert C.near_dup(qs, []) == [False, False, False]
    with pytest.raises(ValueError):
        C.near_dup(qs, ref, threshold=0.0)


def test_short_texts_and_jaccard_edges():
    assert C.shingles("ab", 5) == frozenset({"ab"})
    assert C.shingles("  ", 5) == frozenset()
    assert C.jaccard(frozenset(), frozenset()) == 0.0
    s, j = C.max_similarity(["How many users?"], ["how many users", "zzz"])[0]
    assert j == 0 and s == 1.0


def test_dev_loader_and_db_checks():
    qe, q = C.load_dev_questions(), C.load_dev_questions(with_evidence=False)
    assert len(qe) == len(q) == 1534
    assert qe[0].startswith(q[0]) and len(qe[0]) > len(q[0])
    assert len(C.dev_db_ids()) == 11
    assert C.dev_db_flags(["formula_1", "Formula-1", "hockey", ""]) == [True, True, False, False]
    assert C.is_dev_db("california_schools") and not C.is_dev_db("retail_world")
    tabs = sorted(C.dev_table_sets()["formula_1"])
    assert C.schema_near_dev([t.upper() for t in tabs]) == "formula_1"
    assert C.schema_near_dev(["orders", "customers", "products"]) is None


def test_flag_rows():
    dev = C.load_dev_questions()
    rows = [{"db_id": "hockey", "question": "How many goalies played in 2010?", "evidence": ""},
            {"db_id": "formula_1", "question": "Totally new question about lap times?", "evidence": ""}]
    out = C.flag_rows(rows, ref=dev[:5] + ["How many goalies played in 2010?"])
    assert out == [{"dev_db": False, "near_dup": True}, {"dev_db": True, "near_dup": False}]
