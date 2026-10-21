"""val2 list build (scripts/val2/build_lists.py, mutate.py): union, dedup by result signature, cap, weight, mutants."""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/val2"))
import build_lists as bl  # noqa: E402
import execq  # noqa: E402
import mutate  # noqa: E402


def c(rank, sql, sig, ok=False, runs=True, nonnull=True):
    return {"rank": rank, "cost": float(rank), "sql": sql, "runs": runs, "nonnull": nonnull, "ok": ok, "sig": sig,
            "res": "1 row"}


REC = {"id": "bird_train_train_10", "db_id": "d", "question": "q", "context": "", "gold_sql": "g",
       "in_slots": True, "covered": True}


def test_union_keeps_best_rank_and_src():
    u = bl.union_cands([("r1", [c(0, "A", "s1", True), c(1, "B", "s2")]),
                        ("r2", [c(0, "B", "s2"), c(3, "C", "s3"), {"rank": 4, "cost": 1.0, "sql": None}])])
    by = {x["sql"]: x for x in u}
    assert set(by) == {"A", "B", "C"}
    assert by["B"]["src"] == "both" and by["B"]["rank"] == 0
    assert by["A"]["src"] == "r1" and by["C"]["src"] == "r2"
    assert [x["sql"] for x in u] == ["B", "A", "C"]  # rank-0 ties: "both" first, then run order; then rank 3


def test_dead_dropped_and_one_rep_per_sig():
    lists = [("r1", [c(0, "A1", "s1", True), c(1, "A2", "s1", True), c(2, "W1", "s2"), c(3, "W2", "s2"),
                     c(4, "D", None, runs=False, nonnull=False), c(5, "E", "s3", nonnull=False)])]
    out, info = bl.build_list(REC, lists, seed=0)  # even index: best-ranked member
    assert [x["sql"] for x in out["cands"]] == ["A1", "W1"]
    assert info["classes"] == 2 and info["live"] == 4
    assert all(bl.live(x) for x in out["cands"])
    assert set(out) >= {"id", "db_id", "question", "context", "gold_sql", "in_slots", "covered", "cands", "weight"}


def test_odd_index_rep_is_seeded_random_member():
    rec = dict(REC, id="bird_train_train_11")
    members = [c(k, f"A{k}", "s1", True) for k in range(6)]
    picks = {bl.build_list(rec, [("r1", members + [c(9, "W", "s2")])], seed=s)[0]["cands"][0]["sql"] for s in range(20)}
    assert len(picks) > 1  # random over members across seeds
    a = bl.build_list(rec, [("r1", members + [c(9, "W", "s2")])], seed=3)[0]
    b = bl.build_list(rec, [("r1", members + [c(9, "W", "s2")])], seed=3)[0]
    assert a == b  # deterministic for a fixed seed


def test_cap_keeps_right_and_best_ranked_wrong():
    cands = [c(20, "R", "sr", True)] + [c(k, f"W{k}", f"s{k}") for k in range(15)]
    out, info = bl.build_list(REC, [("r1", cands)], seed=0)
    sqls = [x["sql"] for x in out["cands"]]
    assert len(sqls) == bl.CAP and "R" in sqls
    assert {s for s in sqls if s != "R"} == {f"W{k}" for k in range(11)}
    assert info["classes"] == 16


def test_weight_from_greedy():
    right_greedy = [("r1", [c(0, "A", "s1", True), c(1, "B", "s2")]), ("r2", [c(0, "A", "s1", True)])]
    wrong_greedy = [("r1", [c(0, "A", "s1", True)]), ("r2", [c(0, "B", "s2"), c(1, "A", "s1", True)])]
    dead_greedy = [("r1", [c(0, "X", None, runs=False, nonnull=False), c(1, "A", "s1", True)])]
    assert bl.build_list(REC, right_greedy, 0)[0]["weight"] == 1.0
    assert bl.build_list(REC, wrong_greedy, 0)[0]["weight"] == 3.0
    assert bl.build_list(REC, dead_greedy, 0)[0]["weight"] == 3.0


def test_pick_mutants_filters():
    ok = (True, True, False, "new1", "r")
    trials = [("limit", "M0", (True, True, False, "s1", "r")),   # same sig as a list class
              ("cmp", "M1", (False, False, False, None, None)),  # does not run
              ("distinct", "M2", (True, False, False, "z", "r")),  # all NULL / empty
              ("x100", "M3", (True, True, True, "g", "r")),      # right (same result as gold)
              ("extra_col", "M4", ok),
              ("limit", "M5", (True, True, False, "new1", "r")),  # duplicate of an earlier mutant's sig
              ("cmp", "M6", (True, True, False, "new2", "r")),
              ("distinct", "M7", (True, True, False, "new3", "r"))]
    got = bl.pick_mutants(trials, {"s1", "s2"})
    assert [m["sql"] for m in got] == ["M4", "M6"]  # at most 2
    assert all(m["mutant"] and m["ok"] is False and m["rank"] >= bl.MUT_RANK for m in got)


def test_mutants_one_edit_and_identity_only():
    cols = {"t": ["id", "name", "score"]}
    sql = 'SELECT T1."name" FROM "t" AS T1 WHERE T1."score" > 5'
    m = dict(mutate.mutants(sql, cols, seed="x"))
    assert m["distinct"] == 'SELECT DISTINCT T1."name" FROM "t" AS T1 WHERE T1."score" > 5'
    assert m["limit"] == sql + " LIMIT 1"
    assert m["cmp"] == 'SELECT T1."name" FROM "t" AS T1 WHERE T1."score" >= 5'
    assert m["extra_col"] in ('SELECT T1."name", T1."id" FROM "t" AS T1 WHERE T1."score" > 5',
                              'SELECT T1."name", T1."score" FROM "t" AS T1 WHERE T1."score" > 5')
    assert "x100" not in m  # text projection: no x100 edit
    agg = 'SELECT CAST(SUM(T1."score") AS REAL) * 100 / COUNT(T1."id") FROM "t" AS T1'
    assert dict(mutate.mutants(agg, cols))["x100"] == 'SELECT CAST(SUM(T1."score") AS REAL) / COUNT(T1."id") FROM "t" AS T1'
    assert dict(mutate.mutants('SELECT COUNT(*) FROM "t"', cols))["x100"] == 'SELECT COUNT(*) * 100 FROM "t"'
    assert dict(mutate.mutants('SELECT "name" FROM "t" LIMIT 1', cols))["limit"] == 'SELECT "name" FROM "t"'
    # no round-trip identity (sqlglot would rewrite != as <>): never mutated, so formatting cannot leak
    assert mutate.mutants('SELECT "name" FROM "t" WHERE "id" != 1', cols) == []


def test_exec_labels_like_gen(tmp_path):
    db = tmp_path / "t.sqlite"
    con = sqlite3.connect(db)
    con.executescript('CREATE TABLE t(id INTEGER, name TEXT, score REAL);'
                      "INSERT INTO t VALUES (1,'a',1.0),(2,'b',7.0),(3,'c',9.0);")
    con.commit()
    con.close()
    q = [("k1", 'SELECT "name" FROM "t" WHERE "score" > 5',
          ['SELECT "name" FROM "t" WHERE "score" >= 7', 'SELECT "name" FROM "t" WHERE "score" > 5 LIMIT 1',
           'SELECT nope FROM t', 'SELECT NULL FROM t']),
         ("k2", 'SELECT "name" FROM "t" WHERE "score" > 100', ['SELECT 1']),
         ("k3", 'SELECT NULL', [])]
    out = {k: (st, sg, r) for k, st, sg, r in execq.exec_questions((str(db), q, 5.0, 5.0))}
    st, gsig, r = out["k1"]
    assert st == "ok"
    assert r[0][:3] == (True, True, True) and r[0][3] == gsig
    assert r[1][:3] == (True, True, False) and r[1][3] != gsig
    assert r[2][:2] == (False, False)
    assert r[3][:3] == (True, False, False)
    assert out["k2"][0] == "empty" and out["k3"][0] == "allnull"
