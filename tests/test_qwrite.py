"""scripts/qwrite.py (rare-skeleton question writing) without a GPU: dry prompts, sharding, the round-trip
check with a fake generator, and the build stage's rows going through sq2.train.make_example.
Uses the public BIRD-train 'trains' database and card already in data/sys1.   pytest tests/test_qwrite.py"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

DB_ROOT = REPO / "data/sys1/rrdbs"
CARDS = REPO / "data/sys1/bird_train/cards"
PLANS = REPO / "data/sys1/bird_train/plans.jsonl"
VOCAB = REPO / "data/sys1/templates.json"
pytestmark = pytest.mark.skipif(not (DB_ROOT / "trains/trains.sqlite").exists() or not PLANS.exists(),
                                reason="needs data/sys1 (BIRD-train trains db, cards, plans)")

import build_rare_mix as brm  # noqa: E402
import qwrite  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq.plan import compile_plan  # noqa: E402

SEED_ID = "bird_train_train_690"  # How many trains are there that run in the east direction?


def _seed(sid: str = SEED_ID):
    for line in open(PLANS):
        r = json.loads(line)
        if r["id"] == sid:
            return r
    raise AssertionError("seed row missing")


def _mutant(value="west", mid: str = "bird_train_train_690_mut_filt_op_value", sid: str = SEED_ID) -> dict:
    """A selected.jsonl row built the way build_rare_mix.select builds it."""
    seed = _seed(sid)
    card = SchemaCard.load(CARDS / "trains.json")
    plan = copy.deepcopy(seed["plan"])
    plan["filters"][0]["values"] = [value]
    from luthor.sq.plan import Plan
    sql = compile_plan(Plan.from_dict(plan), card.graph())
    [(_, status, n, sig, sample, dt)] = brm.exec_task((str(DB_ROOT / "trains/trains.sqlite"), [(mid, sql)], 5.0))
    assert status == "ok"
    tabs, ddl = brm.schema_snippet(card, sql)
    return {"id": mid, "seed_id": sid, "db_id": "trains", "op": "filt_op_value", "skel": brm.skeleton(plan),
            "bird_support": 0, "plan": plan, "compiled_sql": sql, "sig": sig, "n_rows": n, "sample_rows": sample,
            "exec_s": dt, "tables": tabs, "schema": ddl}


@pytest.fixture()
def work(tmp_path):
    sel = tmp_path / "selected.jsonl"
    rows = [_mutant()]
    sel.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path, sel, rows


def _args(tmp, sel, *extra):
    return ["--selected", str(sel), "--work", str(tmp / "work"), "--out", str(tmp / "out"), "--cards", str(CARDS),
            "--db-roots", str(DB_ROOT), "--vocab", str(VOCAB), "--workers", "1", *extra]


def test_dry_prompts(work, capsys):
    tmp, sel, rows = work
    shots = qwrite.Shots()
    sh = shots.get(rows[0])
    assert len(sh) == 3 and sh[0]["id"] == SEED_ID                  # the seed question is the first style shot
    assert len({s["db_id"] for s in sh[1:]} | {"trains"}) == 3        # the others come from other databases
    assert all(s["source"] == "bird" and s["split"] == "train" for s in sh)
    assert not {s["db_id"] for s in sh} & qwrite.BIRD_DEV_11
    msgs = qwrite.write_messages(rows[0], sh)
    user = msgs[1]["content"]
    assert rows[0]["compiled_sql"] in user and 'CREATE TABLE trains' in user and "'west'" in user
    assert "JSON only" in user and sh[1]["question"].strip() in user
    qwrite.main(["write", *_args(tmp, sel, "--dry")])
    qwrite.main(["check", *_args(tmp, sel, "--dry")])                 # nothing written yet: seed stands in
    out = capsys.readouterr().out
    assert "WRITE dry" in out and "CHECK dry" in out and "Database Schema:" in out
    assert not (tmp / "work").exists()                                # dry writes nothing


def test_shards_partition():
    ids = [f"bird_train_train_{i}_mut_{op}" for i in range(500) for op in ("proj_add", "filt_drop")]
    rows = [{"id": i} for i in ids]
    a = qwrite.in_shard(rows, (0, 2))
    b = qwrite.in_shard(rows, (1, 2))
    assert not {r["id"] for r in a} & {r["id"] for r in b}
    assert len(a) + len(b) == len(rows) and 400 < len(a) < 600
    assert qwrite.in_shard(rows, (0, 1)) == rows
    assert qwrite.in_shard(rows, (1, 2)) == b                         # stable across calls / stages
    with pytest.raises(AssertionError):
        qwrite.parse_shard("2/2")


def _fake_writer(prompts, n=1, **kw):
    return [[('{"question": "How many trains run in the west direction?", "evidence": "west is a direction"}', 20),
             ('Sure! {"question": "Count the trains heading west.", "evidence": ""}', 15)][:n] for _ in prompts]


def test_write_then_check_accepts_gold_rejects_wrong(work):
    tmp, sel, rows = work
    recs = qwrite.main(["write", *_args(tmp, sel)], gen=_fake_writer)
    assert recs[0]["cands"][0] == {"question": "How many trains run in the west direction?", "evidence": "west is a direction"}
    assert len(recs[0]["cands"]) == 2 and (tmp / "work/written.s0of1.jsonl").exists()

    gold = rows[0]["compiled_sql"]
    wrong = gold.replace("= 'west'", "!= 'nowhere'")   # all trains: a different result

    def checker_one_right(prompts, **kw):  # first pair -> wrong SQL, second pair -> the gold SQL
        assert "Database Schema:" in prompts[0][0]["content"] and "west is a direction\nHow many" in prompts[0][0]["content"]
        return [[(f"reasoning...\n```sql\n{wrong}\n```", 30)], [(f"```sql\n{gold}\n```", 10)]]

    out = qwrite.main(["check", *_args(tmp, sel)], gen=checker_one_right)
    assert out[0]["accepted"]["question"] == "Count the trains heading west."
    assert [c["sig"] == rows[0]["sig"] for c in out[0]["cands"]] == [False, True]

    def checker_all_wrong(prompts, **kw):
        return [[(f"```sql\n{wrong}\n```", 10)], [("```sql\nSELECT COUNT(*) FROM nope\n```", 10)]]

    out = qwrite.main(["check", *_args(tmp, sel)], gen=checker_all_wrong)
    assert out[0]["accepted"] is None
    assert qwrite.accept(rows[0], [{"runs": False, "sig": rows[0]["sig"]}]) is None  # must also run


def test_build_rows_load_in_trainer(work):
    pytest.importorskip("torch")
    tmp, sel, rows = work
    qwrite.main(["write", *_args(tmp, sel)], gen=_fake_writer)
    gold = rows[0]["compiled_sql"]
    qwrite.main(["check", *_args(tmp, sel)], gen=lambda p, **kw: [[(f"```sql\n{gold}\n```", 5)] for _ in p])
    built = qwrite.main(["build", *_args(tmp, sel)])
    assert len(built) == 1
    plans = [json.loads(l) for l in open(tmp / "out/plans.jsonl")]
    assert (tmp / "out/cards/trains.json").exists() and plans[0]["id"] == "rare1_" + rows[0]["id"]
    r = plans[0]
    ref = _seed()   # the exact field set of the BIRD-train rows sys1_full trained on (+3 provenance fields)
    assert set(ref) <= set(r) and set(r) - set(ref) == {"seed_id", "mut_op", "skel"}
    assert r["outcome"] == "plan" and r["empty_result"] is False and r["context"] == "west is a direction"
    assert r["plan"]["filters"][0]["values"] == ["west"] and len(r["value_labels"]) == 1

    from luthor.sq2.targets import set_vocab
    from luthor.sq2.train import load_dir, make_example
    set_vocab(json.loads(VOCAB.read_text()), json.loads(VOCAB.with_name("templates_nums.json").read_text()))
    card = SchemaCard.load(tmp / "out/cards/trains.json")
    e = make_example(r, card, random.Random(0))
    assert e is not None and e.t is not None and e.question == r["question"] and "trains" in e.tables
    cards, exs, gold_rows = load_dir(tmp / "out")                    # the directory loads as a --plans dir
    assert len(exs) == 1 and r["id"] in gold_rows


def test_build_drops_ungrounded_value(tmp_path):
    """A question that never states a (numeric, not enumerable) literal can't be a training row: the model
    points at values. (Low-cardinality text values such as 'west' ground on the card's enumerated values,
    exactly as for BIRD-train rows in prepare_plans.py.)"""
    tmp, sel = tmp_path, tmp_path / "selected.jsonl"
    rows = [_mutant(7, "bird_train_train_691_mut_filt_op_value", "bird_train_train_691")]  # cars of train 7
    sel.write_text(json.dumps(rows[0]) + "\n")
    qwrite.main(["write", *_args(tmp, sel)],
                gen=lambda p, n=1, **kw: [[('{"question": "How many cars does that train have?", "evidence": ""}', 9)]
                                          for _ in p])
    gold = rows[0]["compiled_sql"]
    qwrite.main(["check", *_args(tmp, sel)], gen=lambda p, **kw: [[(f"```sql\n{gold}\n```", 5)] for _ in p])
    built = qwrite.main(["build", *_args(tmp, sel)])
    assert built == []
    st = json.loads((tmp / "out/build_stats.json").read_text())
    assert st["counts"].get("ungrounded_value") == 1


def test_selection_guards():
    bad = brm.forbidden_dbs()
    assert qwrite.BIRD_DEV_11 <= bad
    assert brm.support_bucket(0) == "0" and brm.support_bucket(19) == "5-19" and brm.support_bucket(20) == "20-99"
    sup = brm.bird_support()
    seed = _seed()
    assert sup[brm.skeleton(seed["plan"])] >= 100                     # COUNT + one '=' filter: very common
    ok = [{"id": f"m{i}", "db_id": f"db{i % 3}", "skel": f"s{i % 5}", "bird_support": i % 5} for i in range(60)]
    got = brm.balance(ok, cap=20, per_skel=4, per_db=100)
    assert len(got) == 20 and max(__import__("collections").Counter(m["skel"] for m in got).values()) <= 4
