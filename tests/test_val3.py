"""VAL3 data v3 (scripts/val3): question-source rules, the label-noise mask, list weights, the generic list builder on a
tiny SQLite DB, the big-DB fetcher's extraction, and the box scripts' syntax / dry run. CPU, a few seconds."""
import collections
import json
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/val3"))
import build as B  # noqa: E402
import difficulty as DF  # noqa: E402
import fetch_big_dbs as FB  # noqa: E402
import sources as SRC  # noqa: E402


def cand(rank, sig, ok, runs=True, nonnull=True, **kw):
    return {"rank": rank, "cost": float(rank), "sql": f"SELECT {rank}", "runs": runs, "nonnull": nonnull, "ok": ok,
            "sig": sig, **kw}


# ---- sources.py ----------------------------------------------------------------------------------------------------

def test_bird_kind_rules():
    tr, tw, rare, txt = {"q1"}, {"q2"}, {"q3"}, {("db", "same text")}
    k = lambda i, row, q="x": SRC.bird_kind(i, row, tr, tw, rare, txt, "db", q)  # noqa: E731
    assert k("q1", {"outcome": "plan"}) == ("a", "trained")
    assert k("q2", {"outcome": "plan"}) == ("x", "twin_trained")
    assert k("q3", None) == ("x", "rare1_mutant")
    assert k("q4", None, "Same  TEXT") == ("x", "dup_trained_text")
    assert k("q5", None) == ("b", "norow")
    assert k("q6", {"outcome": "unsupported"}) == ("b", "unsupported")
    assert k("q7", {"outcome": "plan", "empty_result": True}) == ("x", "empty_gold")
    assert k("q8", {"outcome": "plan"}) == ("b", "outside_slots")


def test_gold_ok_rules():
    fx = {1: {"SQL": "select 1", "grading_method": "set"}, 2: {"SQL": "select 2b", "grading_method": "list"},
          3: {"SQL": "select  3"}}
    noise = {"bird_train_train_0", "bird_train_train_1"}
    assert SRC.gold_ok(0, "bird_train_train_0", noise, fx, "select 0") == "gold_noise"   # noise, no Verified entry
    assert SRC.gold_ok(1, "bird_train_train_1", noise, fx, "select 0") is None           # noise but corrected (set)
    assert SRC.gold_ok(2, "bird_train_train_2", noise, fx, "select 2") == "verified_not_set_graded"
    assert SRC.gold_ok(3, "bird_train_train_3", noise, fx, "select 3") is None           # same SQL up to spaces


def test_pick_synsql_quota_and_fill():
    rows = [{"id": f"s{i}", "sql_complexity": c} for i, c in enumerate(["Simple"] * 6 + ["Moderate"] * 1 + ["Complex"] * 3)]
    got = SRC.pick_synsql(rows, 4, {"Simple": 0.2, "Moderate": 0.4, "Complex": 0.3, "Highly Complex": 0.1}, "x")
    cx = collections.Counter(r["sql_complexity"] for r in got)
    assert len(got) == 4 and cx["Moderate"] == 1 and cx["Complex"] >= 1 and cx["Simple"] >= 1
    assert SRC.pick_synsql(rows, 4, {"Simple": .2, "Moderate": .4, "Complex": .3, "Highly Complex": .1}, "x") == got
    assert len(SRC.pick_synsql(rows[:2], 4, {"Simple": 1.0}, "x")) == 2


def test_excluded_dbs_cover_sealed_and_heldout():
    ex = SRC.excluded_dbs()
    for d in json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")):
        assert ex[d.lower()] == "heldout"
    for d in json.load(open(ROOT / "data/sys1/sealed/kaggledbqa.dbs.json")):
        assert d.lower() in ex
    spider_dev = {q["db_id"].lower() for q in json.load(open(ROOT / "data/raw/spider_data/dev.json"))}
    assert spider_dev <= set(ex)
    assert "formula_1" in ex and "soccer_1" in ex


# ---- build.py: mask, hardness, bins, weights -----------------------------------------------------------------------

def lst(cands, **kw):
    return {"id": "bird_train_train_5", "db_id": "d", "cands": cands, "weight": 1.0, **kw}


def test_greedy_and_hard():
    raw = {"cands": [cand(0, "a", False), cand(1, "b", True)]}
    L = lst([cand(0, "a", False), cand(1, "b", True)])
    assert B.greedy(raw)["sig"] == "a" and B.is_hard(L, raw)
    assert not B.is_hard(lst([cand(0, "a", True), cand(1, "b", False)]), raw)  # relabelled: list label wins
    assert B.is_hard(L, {"cands": [cand(0, "a", True, nonnull=False)]})        # dead greedy
    assert B.is_hard(L, None)


def test_mask_rule_all_conditions():
    L = lst([cand(0, "a", False), cand(1, "b", True), cand(2, "c", False)])
    r1 = {"cands": [cand(0, "a", False), cand(1, "b", True)]}
    r2 = {"cands": [cand(0, "a", False)]}
    assert B.mask_sig(L, r1, r2, verified=False) == "a"
    assert B.mask_sig(L, r1, r2, verified=True) is None                                  # (3) Verified entry
    assert B.mask_sig(dict(L, gold_corrected=True), r1, r2, verified=False) is None      # (4) corrected gold
    assert B.mask_sig(L, r1, {"cands": [cand(0, "c", False)]}, verified=False) is None   # (1) R2 disagrees
    assert B.mask_sig(L, r1, None, verified=False) is None                               # no R2 list
    assert B.mask_sig(L, {"cands": [cand(0, "a", False, nonnull=False)]}, r2, False) is None  # (1) dead greedy
    Lr = lst([cand(0, "a", True), cand(1, "b", False)])
    assert B.mask_sig(Lr, r1, r2, verified=False) is None                                # (2) gold agrees
    m = B.apply_mask(L, "a")
    assert [c["sig"] for c in m["cands"]] == ["b", "c"]
    assert B.apply_mask(lst([cand(0, "a", False), cand(1, "b", True)]), "a") is None      # no wrong left -> drop
    pair = lst([cand(0, "a", False), cand(1, "b", True), cand(350, "a", False, prov="pair"), cand(2, "c", False)])
    assert all(c["sig"] != "a" for c in B.apply_mask(pair, "a")["cands"])                 # pair members go too


def test_class_order_and_bins():
    raw = {"cands": [cand(0, "a", False), cand(1, "a", False), cand(2, "x", False, runs=False), cand(3, "b", False),
                     cand(4, "c", True), cand(5, "d", False)]}
    L = lst([cand(0, "a", False), cand(4, "c", True)])
    assert B.class_order(L, raw) == [False, False, True, False]  # b not in list: gen-time label (wrong)
    L2 = lst([cand(0, "a", False), cand(3, "b", True)], gold_corrected=True)
    assert B.class_order(L2, raw) == [False, True, False, False]  # corrected gold: c not in list -> wrong
    assert B.diff_bin([False, False, True, False]) == ("2-3", "2-4")
    assert B.diff_bin([True]) == ("0", "1")
    assert B.diff_bin([False] * 4 + [True] + [False] * 6) == ("4+", "10+")
    assert B.diff_bin([False, False]) == ("none", "2-4")


def test_bin_ratios_and_weights():
    src = [("0", "5-9")] * 8 + [("1", "5-9")] * 2
    tgt = [("0", "5-9")] * 5 + [("1", "5-9")] * 5
    r = B.bin_ratios(src, tgt, alpha=0.0)
    assert r[("0", "5-9")] == pytest.approx(0.625) and r[("1", "5-9")] == pytest.approx(2.5)
    assert B.bin_ratios([("0", "1")] * 99 + [("4+", "10+")], [("4+", "10+")] * 50, alpha=0.0)[("4+", "10+")] == 3.0
    # x3 only on verified or out-of-sample hard lists, never when masked
    assert B.list_weight(True, False, True, False, None) == (3.0, 3.0)
    assert B.list_weight(True, True, False, False, 1.0) == (3.0, 3.0)
    assert B.list_weight(True, False, False, False, 1.0) == (1.0, 1.0)   # unverified in-sample hard: x3 dropped
    assert B.list_weight(True, False, True, True, None) == (1.0, 1.0)    # masked
    assert B.list_weight(False, False, False, False, 2.0) == (1.0, 2.0)
    assert B.list_weight(True, True, False, False, 2.0) == (3.0, 3.0)    # capped at 3
    assert B.list_weight(False, False, False, False, 0.1) == (1.0, pytest.approx(1 / 3))
    assert B.ess([1, 1, 1, 1]) == 4 and B.ess([1, 0, 0, 0]) == 1
    assert B.tv({"a": 1}, {"a": 2}) == 0 and B.tv({"a": 1}, {"b": 1}) == 1


def test_strip_for_val():
    L = lst([cand(0, "a", False), cand(1, "b", True), cand(100, "m", False, mutant=True, prov="mut"),
             cand(300, "e", False, prov="edit"), cand(350, "a", False, prov="pair")], weight=3.0)
    v = B.strip_for_val(L)
    assert [c["sig"] for c in v["cands"]] == ["a", "b"] and v["weight"] == 1.0
    assert B.strip_for_val(lst([cand(0, "a", True), cand(100, "m", False, mutant=True)])) is None


# ---- build.py: generic builder (SynSQL path) on a tiny DB ----------------------------------------------------------

@pytest.fixture
def tiny_db(tmp_path):
    p = tmp_path / "shop" / "shop.sqlite"
    p.parent.mkdir()
    con = sqlite3.connect(p)
    con.executescript("""
      CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
      CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), amount REAL);
      INSERT INTO customers VALUES (1,'ann','paris'),(2,'bob','rome'),(3,'cy','paris');
      INSERT INTO orders VALUES (1,1,10),(2,1,20),(3,2,5),(4,3,7);""")
    con.commit()
    con.close()
    return p


def test_build_generic_and_mix(tiny_db):
    sys.path.insert(0, str(ROOT / "scripts/val2"))
    import execq
    gold = 'SELECT COUNT(T1."id") FROM "orders" AS T1 INNER JOIN "customers" AS T2 ON T1."customer_id" = T2."id" WHERE T2."city" = \'paris\''
    sqls = [gold.replace("COUNT(T1.\"id\")", "COUNT(DISTINCT T2.\"id\")"), gold,
            gold.replace("'paris'", "'rome'"), 'SELECT COUNT(*) FROM "orders"']
    res = execq.run_tasks({"shop": [("synsql_train_data_1", gold, sqls)]}, {"shop": str(tiny_db)}, 5, 3, workers=1)
    st_, _, rr = res["synsql_train_data_1"]
    assert st_ == "ok"
    cands = [dict(rank=k, cost=float(k), sql=s, runs=r[0], nonnull=r[1], ok=r[2], sig=r[3], res=r[4])
             for k, (s, r) in enumerate(zip(sqls, rr))]
    rec = {"id": "synsql_train_data_1", "db_id": "shop", "question": "q", "context": "", "gold_sql": gold,
           "in_slots": True, "covered": True, "cands": cands}
    st = collections.Counter()
    train, val, hard = B.build_generic({rec["id"]: rec}, {"shop": str(tiny_db)}, lambda i: "train", 0, 1, st)
    assert len(train) == 1 and not val
    assert any(c["ok"] for c in train[0]["cands"]) and any(not c["ok"] for c in train[0]["cands"])
    assert train[0]["weight"] == 3.0  # greedy (rank 0) wrong
    mixed = B.mix_lists(train, hard)
    assert len(mixed) == 1 and {c.get("prov") for c in mixed[0]["cands"]} >= {"base"}
    tv_, vv, _ = B.build_generic({rec["id"]: rec}, {"shop": str(tiny_db)}, lambda i: "val", 0, 1, st)
    assert not tv_ and len(vv) == 1 and not any(c.get("mutant") for c in vv[0]["cands"])


# ---- difficulty.py --------------------------------------------------------------------------------------------------

def test_difficulty_feats():
    r = {"gold_sql": "SELECT 0", "cands": [cand(0, "a", False), cand(1, "a", False), cand(2, "b", False),
                                          cand(3, "c", True)]}
    f = DF.feats(r)
    assert f["first_right"] == 2 and f["ncls"] == 3 and f["mixed"] and not f["greedy"] and f["oracle"]
    assert f["gold_string"]  # rank 0 is "SELECT 0"
    s = DF.summarize([f])
    assert s["mixed_right_pos_ge2"] == 100.0 and s["mixed_right_pos_ge4"] == 0.0


# ---- fetch_big_dbs.py -----------------------------------------------------------------------------------------------

def test_parts_cover_everything():
    for total, n in ((10, 3), (8919543554, 8), (5, 8)):
        ps = FB.parts(total, n)
        assert ps[0][0] == 0 and ps[-1][1] == total - 1
        assert all(b[0] == a[1] + 1 for a, b in zip(ps, ps[1:]))


def test_extract_from_inner_zip(tmp_path):
    work, out = tmp_path / "work", tmp_path / "out"
    work.mkdir()
    with zipfile.ZipFile(work / "train_databases.zip", "w") as z:
        z.writestr("train_databases/aa/aa.sqlite", b"x" * 100)
        z.writestr("train_databases/bb/bb.sqlite", b"y" * 7)
        z.writestr("train_databases/cc/cc.sqlite", b"z")
    want = tmp_path / "want.json"
    want.write_text(json.dumps({"aa": 100, "bb": 7}))
    rc = subprocess.call([sys.executable, str(ROOT / "scripts/val3/fetch_big_dbs.py"), "--want", str(want),
                          "--out", str(out), "--work", str(work), "--min-free-gb", "0"])
    assert rc == 0 and (out / "FETCH_OK").exists()
    assert (out / "aa/aa.sqlite").stat().st_size == 100 and not (out / "cc").exists()
    want.write_text(json.dumps({"aa": 100, "bb": 8}))   # size mismatch -> FETCH_FAIL
    (out / "bb/bb.sqlite").unlink()
    with zipfile.ZipFile(work / "train_databases.zip", "w") as z:
        z.writestr("train_databases/bb/bb.sqlite", b"y" * 7)
    rc = subprocess.call([sys.executable, str(ROOT / "scripts/val3/fetch_big_dbs.py"), "--want", str(want),
                          "--out", str(out), "--work", str(work), "--min-free-gb", "0"])
    assert rc == 1 and (out / "FETCH_FAIL").exists() and not (out / "FETCH_OK").exists()


# ---- box scripts ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("f", ["scripts/chains/val3_lists_box.sh", "scripts/chains/run_val3_lists_box.sh"])
def test_box_scripts_parse(f):
    assert subprocess.call(["bash", "-n", str(ROOT / f)]) == 0


def test_box_script_dry_run(tmp_path):
    q = tmp_path / "d/val3/q"
    q.mkdir(parents=True)
    env = {"PATH": "/usr/bin:/bin", "DRY": "1", "REMOTE_ROOT": str(tmp_path / "r"), "REMOTE_D": str(tmp_path / "d"),
           "REPO": str(ROOT), "PY": sys.executable}
    p = subprocess.run(["bash", str(ROOT / "scripts/chains/val3_lists_box.sh")], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout
    assert "VAL3_LISTS_DONE" in out and "_FAIL" not in out
    assert "gen runs/r1r2/r1 32 bird_rr" in out and "gen runs/r1r2/r1 32 synsql" in out
    assert "gen runs/r1r2/r2 1 bird_rr" in out


def test_preflight_before_setup_uses_system_python_only():
    """Two boxes died running venv6's python before setup6 created it: nothing before setup may call $V."""
    s = (ROOT / "scripts/chains/val3_lists_box.sh").read_text()
    pre = s.split("# ---- preflight")[1].split("# ---- setup ---")[0]   # preflight + the big-DB download start
    body = "\n".join(l for l in pre.splitlines() if not l.lstrip().startswith("#"))
    assert "$V" not in body and "venv6" not in body
    assert "python3 -c" in body and "python3 -" in body
