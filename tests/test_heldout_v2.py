"""scripts/analysis/heldout_v2.py: gold pathology tags, a deterministic stratified split that reserves never-scored
questions first, execution-match scoring on a toy SQLite database, and integrity of the written manifests."""
import collections
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("heldout_v2", REPO / "scripts/analysis/heldout_v2.py")
hv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hv)


def test_pathology():
    assert hv.pathology([]) == "empty"
    assert hv.pathology([(0,)]) == "single_zero"
    assert hv.pathology([(0.0,)]) == "single_zero"
    assert hv.pathology([(None,)]) == "single_null"
    assert hv.pathology([(None, None), (None, None)]) == "all_null"
    assert hv.pathology([("0",)]) is None
    assert hv.pathology([(0,), (1,)]) is None
    assert hv.pathology([(3,)]) is None


def _rows(n=200):
    out = []
    for i in range(n):
        out.append({"id": hv.qid(i), "db_id": f"db{i % 3}", "covered": i % 4 != 0,
                    "pathology": "empty" if i % 50 == 0 else None,
                    "used": [] if i % 4 == 0 and i % 8 == 0 else (["v1_yardstick_1000"] if i % 4 else ["ho200"])})
    return out


def test_split_deterministic_disjoint_stratified():
    rows = _rows()
    d1, r1 = hv.split(rows, 0.3)
    d2, r2 = hv.split(list(reversed(rows)), 0.3)
    assert (d1, r1) == (d2, r2)
    assert not set(d1) & set(r1) and set(d1) | set(r1) == {r["id"] for r in rows}
    R = {r["id"]: r for r in rows}
    strata = collections.Counter((r["db_id"], r["covered"], r["pathology"] is not None) for r in rows)
    got = collections.Counter((R[x]["db_id"], R[x]["covered"], R[x]["pathology"] is not None) for x in r1)
    for k, n in strata.items():
        assert got[k] == int(0.3 * n + 0.5)
    # within a stratum, never-scored questions are reserved before scored ones
    for k in strata:
        res = [x for x in r1 if (R[x]["db_id"], R[x]["covered"], R[x]["pathology"] is not None) == k]
        dec = [x for x in d1 if (R[x]["db_id"], R[x]["covered"], R[x]["pathology"] is not None) == k]
        if res and dec:
            assert max(hv.reserve_priority(R[x]["used"]) for x in res) <= min(hv.reserve_priority(R[x]["used"]) for x in dec)


def test_reserve_priority():
    assert hv.reserve_priority([]) == 0
    assert hv.reserve_priority(["01-01_confirmation200"]) == 1
    assert hv.reserve_priority(["ho200", "sq3_pilot"]) == 2
    assert hv.reserve_priority(["v1_yardstick_1000", "ho200"]) == 3


@pytest.fixture
def toy(tmp_path, monkeypatch):
    db = tmp_path / "dbs" / "toy"
    db.mkdir(parents=True)
    con = sqlite3.connect(db / "toy.sqlite")
    con.executescript("CREATE TABLE t(a INTEGER, b TEXT); INSERT INTO t VALUES (1,'x'),(2,'y'),(2,'z');")
    con.commit()
    con.close()
    train = [{"db_id": "toy", "question": "q0", "evidence": "", "SQL": "SELECT a FROM t WHERE a = 2"},
             {"db_id": "toy", "question": "q1", "evidence": "", "SQL": "SELECT b FROM t ORDER BY a"},
             {"db_id": "toy", "question": "q2", "evidence": "", "SQL": "SELECT nope FROM t"}]
    (tmp_path / "train.json").write_text(json.dumps(train))
    monkeypatch.setattr(hv, "DB_ROOT", tmp_path / "dbs")
    monkeypatch.setattr(hv, "TRAIN_JSON", tmp_path / "train.json")
    return tmp_path


def test_score_candidates_execution_match(toy):
    ids = [hv.qid(0), hv.qid(1), hv.qid(2)]
    cands = {hv.qid(0): "SELECT DISTINCT a FROM t WHERE a > 1",  # set semantics: {(2,)} == {(2,)}
             hv.qid(1): "SELECT b FROM t WHERE a = 1",           # wrong rows
             hv.qid(2): "SELECT 1"}                              # gold fails -> wrong
    res = hv.score_candidates(cands, ids, seconds=5, workers=1)
    assert [res[x]["ok"] for x in ids] == [True, False, False]
    assert res[hv.qid(2)]["gold"] == "error"
    res = hv.score_candidates({hv.qid(0): "SELEC broken"}, ids[:2], seconds=5, workers=1)
    assert res[hv.qid(0)] == {"id": hv.qid(0), "ok": False, "gold": "ok", "cand": "error"}
    assert res[hv.qid(1)]["cand"] == "missing" and not res[hv.qid(1)]["ok"]


def test_run_sql_statuses(toy):
    run = hv.connect(hv.db_path("toy"), 5)
    assert hv.run_sql(run, "SELECT a FROM t")[0] == "ok"
    assert hv.run_sql(run, "SELECT x FROM nowhere")[0] == "error"
    assert hv.run_sql(run, "")[0] == "error"
    assert hv.run_sql(run, "SELECT 1; SELECT 2")[0] == "error"
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    assert hv.run_sql(hv.connect(hv.db_path("toy"), 0.2), slow)[0] == "timeout"


def test_duplicate_candidate_ids_rejected(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": "a", "sql": "x"}\n{"id": "a", "sql": "y"}\n')
    with pytest.raises(ValueError):
        hv.load_candidates(p)


@pytest.mark.skipif(not (hv.OUT_DIR / "universe.json").exists(), reason="manifests not built")
def test_written_manifests_are_consistent():
    uni = json.load(open(hv.OUT_DIR / "universe.json"))
    dec, res = hv.load_manifest("decision"), hv.load_manifest("reserved")
    ids = {r["id"] for r in uni["rows"]}
    assert not set(dec["ids"]) & set(res["ids"]) and set(dec["ids"]) | set(res["ids"]) == ids
    assert uni["decision_sha256"] == dec["ids_sha256"] and uni["reserved_sha256"] == res["ids_sha256"]
    held = set(json.load(open(hv.HELDOUT_DBS)))
    assert all(r["db_id"] in held and r["db_id"] not in hv.EXCLUDED_DBS for r in uni["rows"])
    assert len(res["ids"]) >= 300
