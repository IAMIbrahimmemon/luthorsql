"""VAL3 chain (runs/phase1/val3/PREREG.md): staging, box script dry runs, and the eval statistics. CPU only, no labels
of RESERVED are read (fixtures are synthetic)."""
from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EV = _mod("val3_eval", "scripts/analysis/val3_eval.py")
ST = _mod("val3_stage", "scripts/chains/val3_stage.py")
from luthor.stats import cluster_signflip  # noqa: E402


# ---- eval statistics ------------------------------------------------------------------------------------------------
def test_paired_signflip_equals_mcnemar_on_one_seed():
    for w, l_ in [(0, 0), (5, 1), (9, 2), (11, 6), (3, 7), (20, 8)]:
        d = [1.0] * w + [-1.0] * l_ + [0.0] * 13
        assert EV.paired_signflip_one_sided(d) == pytest.approx(EV.mcnemar_one_sided(w, l_))


def test_paired_signflip_half_steps_brute_force():
    import itertools
    d = [1, 0.5, 0.5, -0.5, 1, -1, 0.5]
    obs = sum(d)
    pats = list(itertools.product((-1, 1), repeat=len(d)))
    brute = sum(sum(s * abs(x) for s, x in zip(p, d)) >= obs - 1e-9 for p in pats) / len(pats)
    assert EV.paired_signflip_one_sided(d) == pytest.approx(brute)


def test_db_signflip_matches_cluster_signflip_on_binary():
    rnd = random.Random(0)
    ids = [f"q{i}" for i in range(160)]
    db = {i: f"db{k % 8}" for k, i in enumerate(ids)}
    a = {i: rnd.random() < 0.55 for i in ids}
    b = {i: (a[i] if rnd.random() < 0.85 else not a[i]) or rnd.random() < 0.05 for i in ids}
    ref = cluster_signflip(a, b, db)
    mine = EV.db_signflip({i: int(b[i]) - int(a[i]) for i in ids}, db)
    assert mine["p"] == pytest.approx(ref["p"])
    assert mine["delta"] == pytest.approx(ref["delta"])
    assert mine["ci"][0] == pytest.approx(ref["ci"][0], abs=0.02) and mine["ci"][1] == pytest.approx(ref["ci"][1], abs=0.02)


def _cell(deltas, q=0.01, dbp=0.1):
    return {"delta": sum(deltas) / len(deltas), "seed_deltas": deltas, "q_p_one_sided": q, "db_p": dbp}


def test_verdict_rules():
    assert EV.verdict(_cell([1.2])) == "PASS"
    assert EV.verdict(_cell([0.9])) == "NO PASS"
    assert EV.verdict(_cell([1.2], q=0.06)) == "NO PASS"
    assert EV.verdict(_cell([1.2], dbp=0.25)) == "NO PASS"
    assert EV.verdict(_cell([0.29])) == "KILL"
    assert EV.verdict(_cell([1.8, 0.4])) == "PASS"          # mean 1.1, each seed >= +0.3
    assert EV.verdict(_cell([2.0, 0.2])) == "NO PASS"       # mean 1.1, one seed below the +0.3 floor
    assert EV.verdict(_cell([0.5, 0.0])) == "KILL"          # mean 0.25


def test_compare_two_seeds_mean():
    ids = [f"q{i}" for i in range(16)]
    db = {i: f"db{k % 8}" for k, i in enumerate(ids)}
    base = {i: False for i in ids}
    h1 = {i: k < 4 for k, i in enumerate(ids)}
    h2 = {i: k < 2 for k, i in enumerate(ids)}
    c = EV.compare(base, [h1, h2], ids, db)
    assert c["seed_deltas"] == [25.0, 12.5] and c["delta"] == pytest.approx(18.75)
    assert c["wins"] == 3.0 and c["losses"] == 0
    assert c["q_p_one_sided"] == pytest.approx(1 / 16)  # 4 nonzero diffs, all positive


def test_join_labels_and_pick():
    lab = {"q": {"id": "q", "cands": [{"sql": "a", "ok": False}, {"sql": "b", "ok": True}]}}
    sc = {"q": {"id": "q", "cands": [{"sql": "a", "rr": 1.0, "runs": True, "nonnull": True},
                                    {"sql": "b", "rr": 2.0, "runs": True, "nonnull": True}]}}
    j = EV.join_labels(sc, lab)
    assert EV.pick(j["q"]["cands"])["ok"] is True
    bad = {"q": {"id": "q", "cands": [{"sql": "b"}, {"sql": "a"}]}}
    with pytest.raises(SystemExit):
        EV.join_labels(bad, lab)


def test_close_pairs_and_hard_lists():
    lab = {"q": {"cands": [{"sql": "a", "sig": "A", "rank": 0, "runs": True, "nonnull": True, "ok": False},
                           {"sql": "b", "sig": "B", "rank": 1, "runs": True, "nonnull": True, "ok": True}]}}
    ref = {"q": {"cands": [{**c, "rr": r} for c, r in zip(lab["q"]["cands"], (1.0, 0.5))]}}
    new = {"q": {"cands": [{**c, "rr": r} for c, r in zip(lab["q"]["cands"], (0.0, 3.0))]}}
    assert EV.close_pairs(ref, ["q"]) == {"own_n": 1, "own_acc": 0.0}
    cp = EV.close_pairs(new, ["q"], ref=ref)
    assert cp["own_n"] == 0 and cp["matched_n"] == 1 and cp["matched_acc"] == 1.0
    assert EV.hard_lists(ref, ["q"], lab) == ["q"]


def test_eval_dry_run_reproduces_valhard():
    if not EV.BASE["k32"].exists() or not EV.LABELS["k64r"].exists():
        pytest.skip("VALHARD / repair files not on this machine")
    out = EV.main(["--dry-run"])
    ctl = out["controls"]
    assert all(v == 0 for k, v in ctl.items() if "mismatch" in k)
    assert ctl["valhard_k32_reproduces_59.05"] and ctl["valhard_k64r_reproduces_59.55"]
    assert out["primary"]["A"]["delta"] == 0 and out["primary"]["B"]["delta"] == 0
    assert out["secondary"]["hard_lists"]["n"] == 137
    assert out["secondary"]["close_pairs"]["VALHARD"] == {"own_n": 73, "own_acc": 0.6301}


# ---- staging --------------------------------------------------------------------------------------------------------
def _lists(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(i, db):
    return {"id": i, "db_id": db, "question": "q", "context": "", "gold_sql": "g",
            "cands": [{"sql": "s1", "runs": True, "nonnull": True, "ok": True, "rank": 0},
                      {"sql": "s2", "runs": True, "nonnull": True, "ok": False, "rank": 1}]}


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    pool = tmp_path / "pool.jsonl"
    _lists(pool, [{**_row("bird_train_train_730", "movie"), "in_slots": True, "covered": True}])
    monkeypatch.setattr(ST, "K32", pool)
    monkeypatch.setattr(ST, "K64R", pool)
    cards = tmp_path / "extra" / "cards"
    cards.mkdir(parents=True)
    (cards / "syn_db.json").write_text("{}")
    monkeypatch.setattr(ST, "CARD_DIRS", [cards])
    return tmp_path


def test_stage_strips_labels_and_checks(stage_env):
    t, v, out = stage_env / "t.jsonl", stage_env / "v.jsonl", stage_env / "out"
    _lists(t, [_row("syn_1", "syn_db"), _row("bird_train_train_1", "authors")])
    _lists(v, [_row("syn_2", "syn_db")])
    assert ST.main(["--train", str(t), "--val", str(v), "--out", str(out)]) == 0
    for f in ("k32_pool.jsonl", "k64r_pool.jsonl"):
        txt = (out / f).read_text()
        assert '"ok"' not in txt and "gold_sql" not in txt and "in_slots" not in txt and '"covered"' not in txt
    assert (out / "cards/cards/syn_db.json").exists() and not (out / "cards/cards/authors.json").exists()
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["counts"]["train_usable"] == 2 and man["counts"]["cards_staged"] == 1
    sums = (out / "MANIFEST.sha256").read_text().splitlines()
    assert len(sums) == 5 and all(len(s.split("  ")[0]) == 64 for s in sums)
    # re-stage is idempotent
    assert ST.main(["--train", str(t), "--val", str(v), "--out", str(out)]) == 0


@pytest.mark.parametrize("bad", [_row("bird_train_train_730", "authors"),   # a DECISION id
                                 _row("syn_9", "movie"),                      # a held-out database
                                 _row("syn_9", "no_card_db"),                 # no schema card
                                 _row("syn_2", "syn_db")])                    # id also in val
def test_stage_refuses(stage_env, bad, capsys):
    t, v, out = stage_env / "t.jsonl", stage_env / "v.jsonl", stage_env / "out"
    _lists(t, [_row("syn_1", "syn_db"), bad])
    _lists(v, [_row("syn_2", "syn_db")])
    assert ST.main(["--train", str(t), "--val", str(v), "--out", str(out)]) == 1
    assert "VAL3_STAGE_FAIL" in capsys.readouterr().out and not out.exists()


def test_estimate_monotone():
    e1, e2 = ST.estimate(44700, 400), ST.estimate(100000, 400)
    for arm in "AB":
        assert e2[arm]["max_hours"] > e1[arm]["max_hours"] >= 1.3 * e1[arm]["train_h"] - 0.05
        assert e1[arm]["cap_usd"] > e1[arm]["box_usd"]
    assert e1["B"]["train_h"] > e1["A"]["train_h"]


# ---- box script (DRY) -----------------------------------------------------------------------------------------------
def _box(tmp_path, stage_dir, arm, seed, extra_env=None):
    d = tmp_path / "d"
    (d / "valhard").mkdir(parents=True, exist_ok=True)
    (d / "sys1").mkdir(parents=True, exist_ok=True)
    for link, target in ((d / "valhard/val3", stage_dir), (d / "sys1/bird_train", ROOT / "data/sys1/bird_train")):
        if not link.exists():
            link.symlink_to(target)
    env = {**os.environ, "DRY": "1", "ARM": arm, "SEED": str(seed), "MAX_HOURS": "3.8", "REMOTE_ROOT": str(tmp_path / "box"),
           "REMOTE_D": str(d), "REPO": str(ROOT), "PY": sys.executable, **(extra_env or {})}
    return subprocess.run(["bash", str(ROOT / "scripts/chains/val3_box.sh")], env=env, capture_output=True, text=True)


@pytest.fixture
def staged(stage_env):
    if not (ROOT / "data/sys1/bird_train/cards").exists():
        pytest.skip("no data/sys1/bird_train/cards")
    t, v, out = stage_env / "t.jsonl", stage_env / "v.jsonl", stage_env / "out"
    _lists(t, [_row("syn_1", "syn_db")])
    _lists(v, [_row("syn_2", "syn_db")])
    assert ST.main(["--train", str(t), "--val", str(v), "--out", str(out)]) == 0
    return out


def test_box_dry_arms(tmp_path, staged):
    a = _box(tmp_path, staged, "A", 1)
    assert "VAL3_DONE" in a.stdout, a.stdout
    assert "--init-seed 1" in a.stdout and "--joint-results" not in a.stdout and "--avg-last 4" in a.stdout
    assert "--tok-budget 60000" in a.stdout and "rr_A_s1" in a.stdout and "k32_by_A_s1.jsonl" in a.stdout
    b = _box(tmp_path, staged, "B", 1)
    assert "VAL3_DONE" in b.stdout and "--joint-results --joint-orders 2" in b.stdout and "rr_B_s1" in b.stdout


def test_box_refuses_unstamped_model_and_labels(tmp_path, staged):
    rr = tmp_path / "box/val3/rr_A_s2"
    rr.mkdir(parents=True)
    (rr / "rerank_head.pt").write_bytes(b"x")
    r = _box(tmp_path, staged, "A", 2)
    assert "VAL3_FAIL" in r.stdout and "without a matching VAL3 stamp" in r.stdout
    shutil.copyfile(staged / "train_lists.jsonl", staged / "k32_pool.jsonl")  # a pool with labels (and a stale sha)
    r = _box(tmp_path, staged, "A", 1)
    assert "VAL3_FAIL" in r.stdout
    r = _box(tmp_path, staged, "C", 1)
    assert "ARM must be A or B" in r.stdout
