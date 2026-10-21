"""BIRD dev look #1 tooling: pick rule, one-look lock, blind merge / audit, blind execution parity. No BIRD dev data."""
import importlib.util
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LUTHOR_GRAMMAR", "v3")
for p in ("src", "scripts", "scripts/repair", "scripts/devlook", "scripts/valhard"):
    sys.path.insert(0, str(ROOT / p))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _load("devlook_score", "scripts/analysis/devlook_score.py")
B = _load("blind_gen", "scripts/devlook/blind_gen.py")
P = _load("devlook_prep", "scripts/devlook/prep.py")


def test_pick_matches_valhard_eval():
    V = _load("valhard_eval", "scripts/analysis/valhard_eval.py")
    rnd = random.Random(0)
    for _ in range(500):
        cands = [{"sql": rnd.choice([None, f"q{j}"]), "runs": rnd.random() < .7, "nonnull": rnd.random() < .7,
                  **({"rr": rnd.gauss(0, 1)} if rnd.random() < .9 else {})} for j in range(rnd.randint(0, 6))]
        assert S.pick(cands) is V.pick(cands)


def test_lock_one_look(tmp_path, monkeypatch):
    lock = tmp_path / "LOOK_USED"
    monkeypatch.setattr(S, "LOCK", lock)
    f = tmp_path / "x.jsonl"
    f.write_text("{}\n")
    a = type("A", (), {"variant": "A", "scored": f, "confirm_second_run": False})()
    S.take_lock(a)
    assert lock.exists() and json.loads(lock.read_text())["variant"] == "A"
    with pytest.raises(SystemExit):
        S.take_lock(a)
    a.confirm_second_run = True
    S.take_lock(a)
    assert len(lock.read_text().splitlines()) == 2


def test_assert_blind_and_merge(tmp_path):
    with pytest.raises(SystemExit):
        B.assert_blind({"id": "x", "cands": [{"sql": "s", "ok": True}]})
    with pytest.raises(SystemExit):
        B.assert_blind({"id": "x", "gold_sql": "s", "cands": []})
    pool = tmp_path / "pool.jsonl"
    extra = tmp_path / "extra.jsonl"
    q = {"id": "q1", "db_id": "d", "question": "Q", "context": "C"}
    pool.write_text(json.dumps(q | {"cands": [{"rank": 0, "sql": "a", "runs": True, "src": "k64"}]}) + "\n")
    extra.write_text(json.dumps(q | {"cands": [{"rank": 0, "sql": "a"}, {"rank": 1, "sql": "b", "repair": "F1"}]}) + "\n"
                     + json.dumps(q | {"id": "q2", "cands": [{"sql": "c"}]}) + "\n")
    out = tmp_path / "out.jsonl"
    B.main(["merge", "--pool", str(pool), "--extra", str(extra), "--src", "k32", "--add-missing", "--out", str(out)])
    rows = [json.loads(l) for l in open(out)]
    assert [r["id"] for r in rows] == ["q1", "q2"]
    assert [(c["sql"], c["src"], c["rank"]) for c in rows[0]["cands"]] == [("a", "k64", 0), ("b", "k32", 1)]
    assert rows[0]["cands"][1]["repair"] == "F1"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(q | {"cands": [{"sql": "a", "ok": False}]}) + "\n")
    with pytest.raises(SystemExit):
        B.main(["merge", "--pool", str(bad), "--extra", str(extra), "--src", "k32", "--out", str(out)])


def test_audit_flags_labels(tmp_path):
    good = tmp_path / "g.jsonl"
    good.write_text(json.dumps({"id": "a", "cands": [{"sql": "s", "runs": True, "sig": "x"}]}) + "\n")
    bad = tmp_path / "b.jsonl"
    bad.write_text(json.dumps({"id": "a", "cands": [{"sql": "s", "ok": True}]}) + "\n")
    assert P.audit_file(good) == 0 and P.audit_file(bad) == 1


def test_blind_exec_equals_rerank_exec(tmp_path):
    import rerank_sq2 as R
    db = tmp_path / "t.sqlite"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE t(a INTEGER, b TEXT); INSERT INTO t VALUES (1,'x'),(2,NULL),(NULL,NULL);")
    con.commit()
    con.close()
    sqls = ["SELECT a FROM t", "SELECT b FROM t WHERE a IS NULL", "SELECT nope FROM t", None, "SELECT a FROM t",
            "SELECT COUNT(*) FROM t", "SELECT a FROM t WHERE a > 5"]
    ref = R._exec_db((str(db), [("SELECT 1", sqls)], 3.0))[0]
    got = B._exec_blind((str(db), [sqls], 3.0))[0]
    assert [(r[0], r[1], r[3], r[4]) for r in ref] == got


def test_scorer_exec_semantics(tmp_path):
    db = tmp_path / "t.sqlite"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1),(1),(2);")
    con.commit()
    con.close()
    qs = [("q1", ["SELECT a FROM t"], "SELECT DISTINCT a FROM t", None),        # set equality: hit
          ("q2", ["SELECT a FROM t"], "SELECT a FROM t WHERE a = 1", ["SELECT a FROM t", None]),
          ("q3", ["SELECT broken"], "SELECT a FROM t", None),                  # gold fails: scored 0
          ("q4", ["SELECT a FROM t"], None, None)]                              # no pick: wrong
    out = {q: (h, g, o) for q, h, g, o in S._exec_db((str(db), qs, 3.0))}
    assert out["q1"][0] == [True] and out["q2"][0] == [False] and out["q2"][2] == [True]
    assert out["q3"] == ([False], [True], None) and out["q4"][0] == [False]
