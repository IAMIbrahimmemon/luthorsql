"""Arithmetic pool mutants (scripts/pool/arith_mutants.py): evidence constants, SQL rewrites, gen-time labelling."""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/pool"))
import arith_mutants as am  # noqa: E402


def test_ev_constants_arith_only():
    assert am.ev_constants("total = MULTIPLY(SUM(num_sales), 100000); x refers to num_sales < 0.2") == {100000.0}
    assert am.ev_constants("avg per month = Divide(Count(ward_no), 12); in 2018 refers to date like '%2018%'") == {12.0}
    assert am.ev_constants("rate = a / 9; on 2018/8/1 refers to FL_DATE = '2018/8/1'") == {9.0}
    assert am.ev_constants("percentage = DIVIDE(COUNT(x), COUNT(y)) * 100; year = 2011") == set()
    assert am.ev_constants(None) == set()


def test_mutants_kinds_and_types():
    ms = dict(am.mutants('SELECT SUM(T1."n") FROM "t" AS T1', "1 row x 1 col (integer); first: (3)", "x = a / 9"))
    assert ms["x100"] == 'SELECT SUM(T1."n") * 100 FROM "t" AS T1'
    assert ms["d100"] == 'SELECT CAST(SUM(T1."n") AS REAL) / 100 FROM "t" AS T1'
    assert ms["d100i"] == 'SELECT SUM(T1."n") / 100 FROM "t" AS T1'
    assert ms["d9"] == 'SELECT CAST(SUM(T1."n") AS REAL) / 9 FROM "t" AS T1'
    assert "pct" not in ms
    # text projections are left alone, numeric ones get an @index; percent evidence adds castreal / pct
    ms = dict(am.mutants("SELECT name, COUNT(a) * 100 / COUNT(*) FROM t", "1 row x 2 cols (text, integer); first: ('x', 3)",
                         "percentage refers to ..."))
    assert all(k.endswith("@1") for k in ms)
    assert ms["castreal@1"] == "SELECT name, CAST((COUNT(a) * 100) AS REAL) / COUNT(*) FROM t"
    assert "pct@1" in ms
    assert am.mutants("SELECT * FROM t", "1 row x 1 col (integer); first: (1)", "") == []
    assert am.mutants("SELECT name FROM t", "1 row x 1 col (text); first: ('a')", "") == []


def test_parents_cap_distinct_live():
    cands = [{"rank": i, "sql": f"S{i}", "runs": True, "nonnull": i != 1, "sig": s}
             for i, s in enumerate(["a", "b", "a", "c", "d"])]
    assert [c["sql"] for c in am.parents(cands, 2)] == ["S0", "S3"]


def test_exec_task_keeps_new_live_and_labels_ok(tmp_path):
    db = tmp_path / "d.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (n INTEGER, p INTEGER)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(200, 50), (300, 10)])
    con.commit()
    con.close()
    parent = "SELECT n * p FROM t"
    muts = [(0, k, s) for k, s in am.mutants(parent, "2 rows x 1 col (integer); first: (10000)", "")]
    gold = "SELECT n * p / 100 FROM t"
    [(qid, gold_ok, kept)] = am.exec_task((str(db), [("q", gold, [parent], muts, ["sigparent"])]))
    assert qid == "q" and gold_ok
    by = {k: (ok, sig) for _, k, _, runs, nn, ok, sig, _ in kept}
    # d100 (100.0, 30.0) is the same row set as d100i (100, 30) under Python ==, so only the first one is kept
    assert by["d100"][0] and "d100i" not in by
    assert not by["x100"][0]
    # a mutant equal to the parent's result (as a set) or to an earlier pool sig is dropped
    [(_, _, kept2)] = am.exec_task((str(db), [("q", gold, [parent], [(0, "same", "SELECT p * n FROM t")], [])]))
    assert kept2 == []
