"""VAL4 (runs/phase1/val4/PREREG.md): fold split, fold-list helpers, staging, and DRY runs of both box scripts.
CPU only; fixtures are synthetic or BIRD-train DB names (no held-out label, BIRD dev or sealed file is read)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELD = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
PINNED = {"citeseer", "disney", "professional_basketball", "trains"}
FORBIDDEN = ("bird_dev", "dev_databases", "sealed", "gate", "dev.json", "spider_dk", "kaggledbqa", "realistic")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FO = _mod("val4_folds", "scripts/val4/folds.py")
LI = _mod("val4_lists", "scripts/val4/lists.py")
ST = _mod("val4_stage", "scripts/val4/stage.py")
have_bird = (ROOT / "data/raw/bird_train/train.json").exists() and (ROOT / "data/sys1/rrdbs").is_dir()
need_bird = pytest.mark.skipif(not have_bird, reason="BIRD-train / rrdbs not on this machine")


# ---- fold split -------------------------------------------------------------------------------------------------------
@need_bird
def test_folds_deterministic_balanced_no_heldout():
    a, b = FO.build(0), FO.build(0)
    assert a == b
    f1, f2 = set(a["folds"]["1"]["dbs"]), set(a["folds"]["2"]["dbs"])
    assert not (f1 & f2) and len(f1 | f2) == 60 and a["n_fold_dbs"] == 60
    assert not ((f1 | f2) & HELD)
    assert PINNED <= f1
    q1, q2 = a["folds"]["1"]["questions"], a["folds"]["2"]["questions"]
    assert abs(q1 - q2) / (q1 + q2) < 0.02
    s1, s2 = a["folds"]["1"]["questions_with_sqlite"], a["folds"]["2"]["questions_with_sqlite"]
    assert abs(s1 - s2) / (s1 + s2) < 0.02
    for k in "12":  # dev DBs: the fold's DBs that have an SQLite on the box
        f = a["folds"][k]
        assert set(f["dev_dbs"]) <= set(f["dbs"]) and not (set(f["dev_dbs"]) & set(f["no_sqlite_on_box"]))


@need_bird
def test_fold_files_on_disk_match_and_drop_lists_hold_heldout():
    assert FO.main(["--check"]) == 0, "data/val4 is stale: run python3 scripts/val4/folds.py"
    folds = json.load(open(ROOT / "data/val4/folds.json"))["folds"]
    for k, other in (("1", "2"), ("2", "1")):
        drop = json.load(open(ROOT / f"data/val4/fold{k}_drop.json"))
        dev = json.load(open(ROOT / f"data/val4/fold{k}_dev.json"))
        assert isinstance(drop, list) and isinstance(dev, list)  # plain lists: train_sq2 --drop-dbs / --dev-dbs, gen --dev-dbs
        assert HELD <= set(drop) and set(folds[k]["dbs"]) <= set(drop)
        assert not (set(drop) & set(folds[other]["dbs"]))
        assert set(dev) <= set(folds[k]["dbs"]) and not (set(dev) & HELD)
        # gen for fold k excludes the other fold's drop list: what remains on the box is exactly fold k's dev DBs
        rr = FO.sqlite_dbs()
        genx = set(json.load(open(ROOT / f"data/val4/fold{other}_drop.json")))
        assert rr - genx == set(dev)
    assert not (set(json.load(open(ROOT / "data/sys1/contaminated_dbs.json"))["dbs"]) & set(drop))  # R1 kept them


def test_split_synthetic_pinned_together_and_seeded():
    counts = {f"db{i}": 10 + 7 * i for i in range(20)}
    counts.update({p: 50 for p in PINNED})
    counts["heldx"] = 999
    f1, f2 = FO.split(counts, {"heldx"}, set(counts), seed=3)
    assert PINNED <= set(f1) and "heldx" not in f1 + f2 and len(f1) + len(f2) == 24
    assert (f1, f2) == FO.split(counts, {"heldx"}, set(counts), seed=3)
    tot = sum(v for k, v in counts.items() if k != "heldx")
    assert abs(sum(counts[d] for d in f1) - sum(counts[d] for d in f2)) <= 0.02 * tot


def test_drop_list_parsers_accept_the_format(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841 (train_sq2 imports it)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import train_sq2
    p = tmp_path / "fold1_drop.json"
    p.write_text(json.dumps(sorted(HELD | {"citeseer"})))
    assert train_sq2.load_drop_dbs(str(p)) == HELD | {"citeseer"}
    assert set(json.loads(p.read_text())) == HELD | {"citeseer"}  # --dev-dbs (train_sq2 and rerank_sq2 gen)


# ---- fold lists -> pool / merge ---------------------------------------------------------------------------------------
def _q(i, db, greedy_ok=True, weight=1.0):
    return {"id": f"bird_train_train_{i}", "db_id": db, "question": "q", "gold_sql": "g", "weight": weight,
            "cands": [{"sql": "a", "rank": 0, "ok": greedy_ok, "runs": True, "nonnull": True, "sig": "A"},
                      {"sql": "b", "rank": 1, "ok": not greedy_ok, "runs": True, "nonnull": True, "sig": "B"}]}


def _w(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


@pytest.fixture
def fold_dir(tmp_path):
    d = tmp_path / "folds"
    d.mkdir()
    (d / "fold1_dev.json").write_text(json.dumps(["citeseer", "a1"]))
    (d / "fold2_dev.json").write_text(json.dumps(["b1"]))
    return d


def test_pool_rule_and_checks(tmp_path, fold_dir):
    f1 = _w(tmp_path / "f1.jsonl", [_q(1, "citeseer", False), _q(2, "a1", True)])
    f2 = _w(tmp_path / "f2.jsonl", [_q(3, "b1", False), _q(4, "b1", True)])
    st = LI.pool({"1": f1, "2": f2}, tmp_path / "p.jsonl", fold_dir=fold_dir)
    assert st["status"] == "PASS" and st["greedy_right"] == 50.0 and len((tmp_path / "p.jsonl").read_text().splitlines()) == 4
    f2b = _w(tmp_path / "f2b.jsonl", [_q(3, "b1"), _q(4, "b1"), _q(5, "b1")])
    assert LI.pool({"1": f1, "2": f2b}, tmp_path / "p.jsonl", fold_dir=fold_dir)["status"] == "STOP"  # 80% > 68%
    held = sorted(HELD)[0]
    bad = _w(tmp_path / "bad.jsonl", [_q(6, held)])
    assert LI.pool({"1": f1, "2": bad}, tmp_path / "p2.jsonl", fold_dir=fold_dir)["status"] == "FAIL"
    wrongfold = _w(tmp_path / "wf.jsonl", [_q(7, "a1")])
    st = LI.pool({"1": f1, "2": wrongfold}, tmp_path / "p3.jsonl", fold_dir=fold_dir)
    assert st["status"] == "FAIL" and not (tmp_path / "p3.jsonl").exists()


def test_merge_val_only_pinned(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _w(work / "train_lists.jsonl", [_q(1, "a1", False, 3.0), _q(2, "b1")])
    _w(work / "val_lists.jsonl", [_q(3, "citeseer")])
    old = _w(tmp_path / "old.jsonl", [_q(1, "a1"), _q(9, "c1")])
    st = LI.merge(work, old, tmp_path / "out")
    assert st["status"] == "OK" and st["train"]["lists"] == 4 and st["val"]["lists"] == 1
    assert st["ids_in_both_sources"] == 1 and st["train_fold"]["list_greedy_right"] == 50.0
    _w(old, [_q(9, "trains")])  # a pinned val DB in VALHARD's train lists
    assert LI.merge(work, old, tmp_path / "out2")["status"] == "FAIL"
    _w(old, [_q(9, "c1")])
    _w(work / "val_lists.jsonl", [_q(3, "a1")])  # a val list off the pinned DBs
    assert LI.merge(work, old, tmp_path / "out3")["status"] == "FAIL"


# ---- staging for the validator box ------------------------------------------------------------------------------------
@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    if not (ROOT / "data/sys1/bird_train/cards").exists():
        pytest.skip("no data/sys1/bird_train/cards")
    data = tmp_path / "data"
    data.mkdir()
    _w(data / "train_lists.jsonl", [_q(1, "books", False, 3.0), _q(1, "books"), _q(2, "authors")])
    _w(data / "val_lists.jsonl", [_q(3, "citeseer")])
    (data / "unseen_check.json").write_text(json.dumps({"status": "PASS", "greedy_right": 60.0}))
    pool = _w(tmp_path / "pool.jsonl", [{**_q(4, "books"), "in_slots": True, "covered": True}])
    monkeypatch.setattr(ST, "K32", pool)
    monkeypatch.setattr(ST, "K64R", pool)
    monkeypatch.setattr(ST.V3, "SPLITS", [])
    return data


def test_stage_strips_labels_allows_repeat_ids(stage_env):
    out = stage_env / "box"
    assert ST.main(["--data", str(stage_env), "--out", str(out)]) == 0
    for f in ("k32_pool.jsonl", "k64r_pool.jsonl"):
        txt = (out / f).read_text()
        assert '"ok"' not in txt and "gold_sql" not in txt and "in_slots" not in txt and '"covered"' not in txt
    names = [l.split("  ")[1] for l in (out / "MANIFEST.sha256").read_text().splitlines()]
    assert names == ["train_lists.jsonl", "val_lists.jsonl", "k32_pool.jsonl", "k64r_pool.jsonl"]
    e = ST.estimate(90000)
    assert e["max_hours"] >= 1.3 * e["train_h"] - 0.05 and e["max_hours"] >= 6.5


def test_stage_refuses(stage_env, capsys):
    (stage_env / "unseen_check.json").write_text(json.dumps({"status": "STOP"}))
    assert ST.main(["--data", str(stage_env), "--out", str(stage_env / "b1")]) == 1
    (stage_env / "unseen_check.json").write_text(json.dumps({"status": "PASS"}))
    _w(stage_env / "val_lists.jsonl", [_q(3, "books")])  # val off the pinned DBs (and a train DB)
    assert ST.main(["--data", str(stage_env), "--out", str(stage_env / "b2")]) == 1
    _w(stage_env / "val_lists.jsonl", [_q(3, sorted(HELD)[0])])  # a held-out DB
    assert ST.main(["--data", str(stage_env), "--out", str(stage_env / "b3")]) == 1
    assert "VAL4_STAGE_FAIL" in capsys.readouterr().out
    assert not (stage_env / "b1").exists() and not (stage_env / "b3").exists()


# ---- box scripts (DRY) ------------------------------------------------------------------------------------------------
def _commands(out: str) -> list[str]:
    return [l for l in out.splitlines() if "DRY: " in l or "] train: " in l or "] gen: " in l]


def _fold_box(tmp_path, fold, extra=None):
    d = tmp_path / "d"
    (d / "sys1").mkdir(parents=True, exist_ok=True)
    (d / "val4").mkdir(exist_ok=True)
    for f in ("fold1_drop.json", "fold2_drop.json", "fold1_dev.json", "fold2_dev.json"):
        if not (d / "val4" / f).exists():
            (d / "val4" / f).write_bytes((ROOT / "data/val4" / f).read_bytes())
    for link, target in ((d / "sys1/rrdbs", ROOT / "data/sys1/rrdbs"), (d / "rare1", ROOT / "data/rare1")):
        if not link.exists() and target.exists():
            link.symlink_to(target)
    env = {**os.environ, "DRY": "1", "FOLD": str(fold), "REMOTE_ROOT": str(tmp_path / "box"), "REMOTE_D": str(d),
           "REPO": str(ROOT), "PY": sys.executable, **(extra or {})}
    return subprocess.run(["bash", str(ROOT / "scripts/chains/val4_fold_box.sh")], env=env, capture_output=True, text=True)


@need_bird
@pytest.mark.parametrize("fold", [1, 2])
def test_fold_box_dry(tmp_path, fold):
    r = _fold_box(tmp_path, fold)
    out = r.stdout
    assert r.returncode == 0 and "VAL4FOLD_DONE" in out, out + r.stderr
    other = 3 - fold
    cmds = _commands(out)
    train = next(c for c in cmds if "DRY: " in c and "train_sq2.py" in c)
    gen = next(c for c in cmds if "DRY: " in c and "rerank_sq2.py gen" in c)
    d = tmp_path / "d"
    for flag in (f"--plans {d}/sys1/bird_train {d}/sys1/spider_train {d}/sys1/synsql_train {d}/rare1_fold{fold} --db-roots",
                 f"--drop-dbs {d}/val4/fold{fold}_drop.json", f"--dev-dbs {d}/val4/fold{fold}_dev.json",
                 "--repeat bird_train=3", "--max-steps 6714", "--cand-order v2", "--value-side", "--col-mentions",
                 "--budget-chars 9000", "--eval-every 1000", "--base seeklhy/OmniSQL-7B", f"--vocab {d}/sys1/templates.json",
                 "--max-oom-skip 0.05", f"--db-roots {d}/sys1/rrdbs --base"):
        assert flag in train, flag
    for bad in ("--gate", "--seed", "--grammar", "--extra-train", "--hours", "--bird-desc"):
        assert bad not in train, bad
    for flag in ("--k 32", "--distinct", "--split train", f"--dev-dbs {d}/val4/fold{other}_drop.json",
                 f"--plans {d}/sys1/bird_train --db-roots {d}/sys1/rrdbs", f"--out {tmp_path}/box/val4/fold{fold}/cands_fold.jsonl"):
        assert flag in gen, flag
    assert "--max-q" not in gen and "--all-questions" not in gen
    for c in cmds:
        for f in FORBIDDEN:
            assert f not in c.replace(f"fold{fold}_dev.json", "").replace(f"fold{other}_dev.json", ""), (f, c)
    if (ROOT / "data/rare1").exists():
        assert f"fold {fold}: " in out and "rare1 = R1's rows" in out  # the system-python fold check ran
        dev = set(json.load(open(ROOT / f"data/val4/fold{fold}_dev.json")))
        rows = [json.loads(l) for l in open(d / f"rare1_fold{fold}/plans.jsonl")]
        assert rows and not any(r["db_id"] in dev for r in rows)  # the fold dev set holds BIRD-train rows only
        assert len(rows) + int(out.split("removed ")[1].split()[0]) == 5483
    assert "PROJECTION fold:" in out


@need_bird
def test_fold_box_refuses(tmp_path):
    r = _fold_box(tmp_path, 3)
    assert "VAL4FOLD_FAIL" in r.stdout and "FOLD must be 1 or 2" in r.stdout
    (tmp_path / "d").mkdir(exist_ok=True)
    (tmp_path / "d" / "dev_databases").mkdir()
    r = _fold_box(tmp_path, 1)
    assert "VAL4FOLD_FAIL" in r.stdout and "dev_databases is on this box" in r.stdout
    os.rmdir(tmp_path / "d" / "dev_databases")
    (tmp_path / "d/val4/fold2_drop.json").write_text(json.dumps(["books"]))  # gen exclusion without the held-out DBs
    r = _fold_box(tmp_path, 1)
    if (ROOT / "data/rare1").exists():
        assert "VAL4FOLD_FAIL" in r.stdout and "fold list check failed" in r.stdout


def _val_box(tmp_path, staged: Path, seed, extra=None):
    d = tmp_path / "d"
    (d / "valhard").mkdir(parents=True, exist_ok=True)
    (d / "sys1").mkdir(parents=True, exist_ok=True)
    for link, target in ((d / "valhard/val4", staged), (d / "sys1/bird_train", ROOT / "data/sys1/bird_train")):
        if not link.exists():
            link.symlink_to(target)
    env = {**os.environ, "DRY": "1", "SEED": str(seed), "REMOTE_ROOT": str(tmp_path / "box"), "REMOTE_D": str(d),
           "REPO": str(ROOT), "PY": sys.executable, **(extra or {})}
    return subprocess.run(["bash", str(ROOT / "scripts/chains/val4_box.sh")], env=env, capture_output=True, text=True)


@pytest.fixture
def val_staged(stage_env):
    out = stage_env / "box"
    assert ST.main(["--data", str(stage_env), "--out", str(out)]) == 0
    flat = stage_env / "flat"  # the box's /root/d/valhard/val4: lists + pools + manifest in one dir
    flat.mkdir()
    for f in ("train_lists.jsonl", "val_lists.jsonl"):
        (flat / f).write_bytes((stage_env / f).read_bytes())
    for f in ("k32_pool.jsonl", "k64r_pool.jsonl", "MANIFEST.sha256", "MANIFEST.json"):
        (flat / f).write_bytes((out / f).read_bytes())
    return flat


def test_val_box_dry(tmp_path, val_staged):
    r = _val_box(tmp_path, val_staged, 2)
    out = r.stdout
    assert r.returncode == 0 and "VAL4_DONE" in out, out + r.stderr
    cmds = _commands(out)
    train = next(c for c in cmds if "rerank_sq2.py train" in c)
    for flag in ("--seed 2 --init-seed 2", "--loss group", "--label-smooth 0.05", "--tok-budget 120000", "--epochs 2",
                 "--sched cosine", "--lora-r 16", "--max-hours 6.5", "--base seeklhy/OmniSQL-7B",
                 f"--train-file {tmp_path}/d/valhard/val4/train_lists.jsonl", f"--val-file {tmp_path}/d/valhard/val4/val_lists.jsonl",
                 f"--out {tmp_path}/box/val4/rr_s2"):
        assert flag in train, flag
    for flag in ("--eval-every", "--avg-last", "--joint-results", "--warmup"):
        assert flag not in train, flag  # VALHARD's recipe exactly (+ the seed)
    scores = [c for c in cmds if "rerank_sq2.py score" in c]
    assert len(scores) == 2
    assert "k32_pool.jsonl" in scores[0] and f"{tmp_path}/box/val4/k32_by_s2.jsonl" in scores[0]
    assert "k64r_pool.jsonl" in scores[1] and f"{tmp_path}/box/val4/k64r_by_s2.jsonl" in scores[1]
    for c in cmds:
        for f in FORBIDDEN:
            assert f not in c, (f, c)


def test_val_box_refuses(tmp_path, val_staged):
    assert "SEED must be" in _val_box(tmp_path, val_staged, "x").stdout
    (val_staged / "k32_pool.jsonl").write_bytes((val_staged / "train_lists.jsonl").read_bytes())  # labels + stale sha
    r = _val_box(tmp_path, val_staged, 1)
    assert "VAL4_FAIL" in r.stdout and "VAL4_DONE" not in r.stdout


def test_scripts_reference_no_bird_dev_outputs():
    for p in ("scripts/chains/val4_fold_box.sh", "scripts/chains/run_val4_fold_box.sh", "scripts/chains/val4_box.sh",
              "scripts/chains/run_val4_box.sh", "scripts/val4/build_lists.sh", "scripts/val4/folds.py",
              "scripts/val4/lists.py", "scripts/val4/stage.py"):
        t = (ROOT / p).read_text()
        for f in ("cands_bird_dev", "bird_dev_scored", "gate.json", "data/eval/sealed", "data/sys1/sealed", "leak_patterns",
                  "RESERVED_USED"):
            assert f not in t, (p, f)


def test_launcher_dry_paths_and_leakcheck_allow():
    t = (ROOT / "scripts/vast/leakcheck.sh").read_text()
    assert "|data/val4|" in t and "# data/val4:" in t
    vb = (ROOT / "scripts/chains/valhard_box.sh").read_text()
    assert "VH_TAG=r1" in vb and "val4" not in vb  # VALHARD's frozen script untouched
