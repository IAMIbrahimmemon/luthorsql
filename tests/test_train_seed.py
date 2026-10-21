"""train_sq2.py --seed and --extra-train: seed 0 keeps every earlier run's database grouping bit for bit, other seeds
give their own (reproducible) order; extra plan rows (e.g. teacher-generated) load with their provenance, never on
held-out databases, never overwriting another row's gold, repeated by weight. CPU, no model."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_sq2 as ts  # noqa: E402
from luthor.sq2.train import db_groups, group_seed  # noqa: E402
from test_sq2_r1_levers import _card, _rows  # noqa: E402


def _exs():
    return [SimpleNamespace(id=f"q{i}", db_id=f"db{i % 7}") for i in range(200)]


def _order(groups):
    return [(db, [e.id for e in es]) for db, es in groups]


def test_seed_zero_is_the_old_order():
    exs = _exs()
    for epoch in range(3):
        assert group_seed(0, epoch) == epoch
        assert _order(db_groups(exs, 8, seed=group_seed(0, epoch))) == _order(db_groups(exs, 8, seed=epoch))


def test_other_seeds_differ_and_repeat():
    exs = _exs()
    s0 = _order(db_groups(exs, 8, seed=group_seed(0, 0)))
    s1 = _order(db_groups(exs, 8, seed=group_seed(1, 0)))
    assert s1 != s0
    assert _order(db_groups(exs, 8, seed=group_seed(1, 0))) == s1  # same seed twice: identical
    assert _order(db_groups(exs, 8, seed=group_seed(1, 1))) != s1  # each epoch its own order
    assert sorted(e for _, es in s1 for e in es) == sorted(e.id for e in exs)  # a reordering, nothing lost
    assert len({group_seed(s, e) for s in range(4) for e in range(50)}) == 200
    with pytest.raises(ValueError):
        group_seed(-1, 0)


def test_flags_default_to_the_old_run():
    a = ts.parser().parse_args(["--plans", "p", "--dev-dbs", "d.json", "--out", "o"])
    assert a.seed == 0 and a.extra_train == [] and not a.auto_resume and a.save_every == 0 and a.job_hours == 0
    assert json.loads(json.dumps({k: str(v) for k, v in vars(a).items()}))["seed"] == "0"  # args.json records it


def test_parse_extra():
    assert ts.parse_extra("data/t/rows.jsonl:2.5") == (Path("data/t/rows.jsonl"), 2.5)
    assert ts.parse_extra("data/t") == (Path("data/t"), 1.0)
    assert ts.parse_extra("x.jsonl:.5") == (Path("x.jsonl"), 0.5)
    assert ts.parse_extra("x.jsonl:0") == (Path("x.jsonl"), 0.0)


def _plan_dir(tmp_path):
    """--plans dir with the shop card; an extra .jsonl (no cards of its own) of teacher rows on shop and on a
    held-out database, one id clashing with an already loaded row."""
    d = tmp_path / "bird_train"
    (d / "cards").mkdir(parents=True)
    _card().save(d / "cards" / "shop.json")
    held = _card()
    held.db_id = "heldout"
    held.save(d / "cards" / "heldout.json")
    rows = []
    for k, r in enumerate(_rows()):
        rows.append({**r, "id": f"t{k}", "source": "teacher_gen", "teacher": "omnisql32b"})
    rows.append({**_rows()[0], "id": "clash", "source": "teacher_gen"})
    rows.append({**_rows()[1], "id": "h0", "db_id": "heldout", "source": "teacher_gen"})
    rows.append({**_rows()[2], "id": "t9", "outcome": "unsupported"})  # not a plan row: ignored
    f = tmp_path / "teacher" / "rows.jsonl"
    f.parent.mkdir()
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d, f


def test_extra_train_rows(tmp_path):
    d, f = _plan_dir(tmp_path)
    cards, gold = {}, {"clash": {"id": "clash", "source": "bird"}}
    out, c = ts.load_extra(f"{f}:2.5", [d], cards, gold, {"heldout"}, k=0, workers=1)
    kept = {e.id for e in out}
    assert kept == {"t0", "t1", "t2", "clash~x0"}  # held-out db dropped, clashing id renamed, non-plan row ignored
    assert gold["clash"] == {"id": "clash", "source": "bird"}  # the earlier row's gold is untouched
    assert gold["t0"]["teacher"] == "omnisql32b" and gold["clash~x0"]["id"] == "clash~x0"
    assert "shop" in cards and "heldout" not in cards
    assert len(out) == 4 * 2 + 2  # weight 2.5: two copies plus a fixed half
    assert c["held_out_dbs"] == 1 and c["renamed_ids"] == 1 and c["kept"] == 4 and c["train_rows"] == 10
    assert c["kept_by_source"] == {"teacher_gen/omnisql32b": 3, "teacher_gen": 1}
    assert c["read_by_source"]["?"] == 1
    out2, _ = ts.load_extra(f"{f}:2.5", [d], {}, {"clash": {}}, {"heldout"}, k=0, workers=1)
    assert [e.id for e in out2] == [e.id for e in out]  # the fractional sample is fixed


def test_extra_train_ids_unique_across_specs(tmp_path):
    d, f = _plan_dir(tmp_path)
    gold = {}
    a, _ = ts.load_extra(str(f), [d], {}, gold, set(), k=0, workers=1)
    b, c = ts.load_extra(str(f), [d], {}, gold, set(), k=1, workers=1)  # the same file twice: no id overwritten
    assert c["renamed_ids"] == len(b) and not ({e.id for e in a} & {e.id for e in b})
