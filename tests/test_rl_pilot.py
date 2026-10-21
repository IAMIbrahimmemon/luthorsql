"""rl_train.py pilot readiness (scripts/rl): list anchor + RL gradients, out-of-memory recovery (forward
OOM retried in halves with the same gradient, backward OOM drops the step), the off-policy buffer's held-out filter,
--drop-ids, the noisy-gold rule, parallel SQLite execution with per-query deadlines, the timing projection, and a
2-step end-to-end run on the tiny model. CPU only; the end-to-end test takes about a minute."""
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rl"))
import rl_core as rc  # noqa: E402
import rl_data  # noqa: E402
import rl_train  # noqa: E402
import luthor.sq2.train as tr  # noqa: E402
from luthor.sq.plan import compile_plan  # noqa: E402
from test_sq2_r1_levers import _card, _rows, _tiny  # noqa: E402
from _global_state import isolated  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate():
    """rl_train.main sets R2's template vocabulary and candidate order process-wide: put the module globals and the
    random state back after every test, so later test files see what they would see alone."""
    with isolated(seed=0, pristine=False):
        yield


# ---- fixtures ----------------------------------------------------------------------------------------------

def _shop_db(path: Path, n_codes: int = 8) -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT)")
    con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT, amount REAL)")
    con.execute("CREATE TABLE codes (id INTEGER PRIMARY KEY, " + ", ".join(f"flag{k} TEXT" for k in range(n_codes)) + ")")
    con.executemany("INSERT INTO customers VALUES (?, ?, ?)", [(1, "a", "Egypt"), (2, "b", "Kenya"), (3, "c", "Egypt")])
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)",
                    [(i, 1 + i % 3, ["shipped", "cancelled"][i % 2], 10.0 * i) for i in range(1, 13)])
    con.executemany("INSERT INTO codes VALUES (?, " + ", ".join("?" * n_codes) + ")",
                    [(i, *[f"c{k}v{(i + k) % 12}" for k in range(n_codes)]) for i in range(1, 25)])
    con.commit()
    con.close()
    return path


def _setup(tmp_path, warm=15, **over):
    card = _card()
    rows = _rows()
    g = card.graph()
    gold = {}
    for r in rows:
        from luthor.sq.plan import Plan
        sql = compile_plan(Plan.from_dict(r["plan"]), g)
        gold[r["id"]] = {**r, "gold_sql": sql, "compiled_sql": sql}
    exs = [tr.make_example(r, card, random.Random(0), order="v2") for r in rows]
    db = _shop_db(tmp_path / "shop.sqlite")
    a = rl_train.parse_args(["--plans", "x", "--db-roots", str(tmp_path), "--out", str(tmp_path / "out"),
                             "--device", "cpu", "--G", "8", "--workers", "1", "--sft-weight", "0.5",
                             "--list-weight", "0.5", "--beta", "0.01"])
    for k, v in over.items():
        setattr(a, k, v)
    m = _tiny(value_side=True)
    opt, rnd = torch.optim.AdamW(m.parameters(), lr=3e-3), random.Random(0)
    for _ in range(warm):  # a little SFT, so plans compile and the lists mix right and wrong plans
        tr.train_db(m, card, exs, 9000, rnd)
        opt.step()
        opt.zero_grad(set_to_none=True)
    m.eval()
    ref = rl_train.add_reference(m, None, tiny=True)
    return a, m, ref, card, exs, gold, rc.ExecReward({"shop": db}, 2.0, 1)


def _grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def _run(a, m, ref, card, exs, gold, rw, ledger=None):
    import train_sq2
    log = __import__("collections").Counter()
    ledger = ledger or train_sq2.OOMLedger(0.5, min_seen=10**9)
    n = rl_train.run_rl_db(a, m, ref, card, "shop", exs, gold, rw, random.Random(0), 1234, log, ledger, 1.0)
    return n, log, ledger


# ---- list anchor + RL gradients ------------------------------------------------------------------------------

def test_list_anchor_and_rl_gradients_finite(tmp_path):
    a, m, ref, card, exs, gold, rw = _setup(tmp_path)
    n, log, _ = _run(a, m, ref, card, exs, gold, rw)
    assert n == 3 and log["q"] == 3
    assert log["list_groups"] > 0, "the list anchor never had a mixed right/wrong list"
    gr = _grads(m)
    assert gr and all(torch.isfinite(v).all() for v in gr.values())
    assert float(m.W_q.weight.grad.abs().sum()) > 0
    for k in ("t_forward", "t_sample", "t_kbest", "t_exec", "t_logp", "t_list", "t_backward"):
        assert log[k] >= 0
    assert log["t_backward"] > 0


def test_list_anchor_alone_moves_the_heads(tmp_path):
    a, m, ref, card, exs, gold, rw = _setup(tmp_path, rl_weight=0.0, sft_weight=0.0, list_weight=1.0)
    _, log, _ = _run(a, m, ref, card, exs, gold, rw)
    assert log["plans"] == 0 and log["list_groups"] > 0
    assert float(m.W_q.weight.grad.abs().sum()) > 0


# ---- out-of-memory recovery ------------------------------------------------------------------------------------

def test_forward_oom_split_keeps_the_gradient(tmp_path, monkeypatch):
    a, m, ref, card, exs, gold, rw = _setup(tmp_path)
    _run(a, m, ref, card, exs, gold, rw)
    g_full = _grads(m)
    m.zero_grad(set_to_none=True)
    real = tr.encode_db

    def oom_if_big(model, card_, group, *args, **kw):
        if torch.is_grad_enabled() and len(group) > 1:
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return real(model, card_, group, *args, **kw)
    monkeypatch.setattr(tr, "encode_db", oom_if_big)
    n, log, ledger = _run(a, m, ref, card, exs, gold, rw)
    assert n == 3 and log["q"] == 3 and ledger.retries == 2 and ledger.skipped == 0
    g_split = _grads(m)
    assert g_full.keys() == g_split.keys()
    for k in g_full:
        assert torch.allclose(g_full[k], g_split[k], rtol=1e-3, atol=1e-6), k


def test_forward_oom_on_single_question_is_skipped_and_counted(tmp_path, monkeypatch):
    a, m, ref, card, exs, gold, rw = _setup(tmp_path)

    def always(*args, **kw):
        raise RuntimeError("MPS backend out of memory (test)")
    monkeypatch.setattr(tr, "encode_db", always)
    n, log, ledger = _run(a, m, ref, card, exs, gold, rw)
    assert n == 0 and ledger.skipped == 3 and log["oom_skipped"] == 3
    import train_sq2
    with pytest.raises(SystemExit):  # past --max-oom-skip the run stops
        _run(a, m, ref, card, exs, gold, rw, train_sq2.OOMLedger(0.1, min_seen=0))


def test_backward_oom_raises_for_the_step_to_be_dropped(tmp_path, monkeypatch):
    a, m, ref, card, exs, gold, rw = _setup(tmp_path)

    def boom(self, *args, **kw):
        raise torch.OutOfMemoryError("CUDA out of memory (test)")
    monkeypatch.setattr(torch.Tensor, "backward", boom)
    with pytest.raises(tr.BackwardOOM):
        _run(a, m, ref, card, exs, gold, rw)


def test_other_errors_are_not_swallowed(tmp_path, monkeypatch):
    a, m, ref, card, exs, gold, rw = _setup(tmp_path)
    monkeypatch.setattr(tr, "encode_db", lambda *x, **k: (_ for _ in ()).throw(ValueError("bug")))
    with pytest.raises(ValueError):
        _run(a, m, ref, card, exs, gold, rw)


# ---- buffer, drop ids, noisy gold ----------------------------------------------------------------------------

def _buf_row(i, db, ok=(True, False)):
    return {"id": f"bv_train_{i}", "db_id": db, "question": "q", "context": "", "gold_sql": "SELECT 1",
            "in_slots": False, "cands": [{"sql": f"SELECT {k}", "ok": o} for k, o in enumerate(ok)]}


def test_buffer_filters_heldout_dbs(tmp_path):
    p = tmp_path / "train_cands.jsonl"
    p.write_text("".join(json.dumps(_buf_row(i, db)) + "\n" for i, db in enumerate(["a", "held", "b", "held"])))
    buf, st = rl_data.load_buffer([p], {"held"})
    assert set(buf) == {"bird_train_train_0", "bird_train_train_2"}
    assert st["dropped_heldout"] == 2 and st["kept"] == 2
    assert all(v["gold_source"] == "bird_verified" for v in buf.values())
    assert buf["bird_train_train_0"]["ok_verified"] == [True, False]
    bad = tmp_path / "score_bird_dev.jsonl"
    bad.write_text("")
    with pytest.raises(ValueError):
        rl_data.load_buffer([bad], set())


def test_real_buffer_drops_the_337_heldout_questions():
    if not all(p.exists() for p in rl_data.BUFFER_FILES) or not rl_data.HELDOUT_DBS.exists():
        pytest.skip("no sel_cands files")
    held = rl_data.heldout_dbs()
    buf, st = rl_data.load_buffer(rl_data.BUFFER_FILES, held)
    assert st["dropped_heldout"] == 337 and st["kept"] == 2121
    assert not any(v["db_id"] in held for v in buf.values())
    assert all(k.startswith("bird_train_train_") for k in buf)


def test_drop_ids_formats(tmp_path):
    ids = ["bird_train_train_1", "bv_val_7"]
    want = {"bird_train_train_1", "bird_train_train_7"}
    for i, obj in enumerate([ids, {"ids": ids}, {"drop": ids}, {k: "empty" for k in ids}]):
        (tmp_path / f"d{i}.json").write_text(json.dumps(obj))
        assert rl_data.load_drop_ids(tmp_path / f"d{i}.json") == want
    (tmp_path / "d.jsonl").write_text("".join(json.dumps({"id": k, "why": "error"}) + "\n" for k in ids))
    assert rl_data.load_drop_ids(tmp_path / "d.jsonl") == want


def test_noisy_gold_rule():
    assert rc.noisy_gold(None) == "error"
    assert rc.noisy_gold(frozenset()) == "empty"
    assert rc.noisy_gold(frozenset({(None,)})) == "empty"
    assert rc.noisy_gold(frozenset({(0,)})) == "single_zero"
    assert rc.noisy_gold(frozenset({(0.0,)})) == "single_zero"
    assert rc.noisy_gold(frozenset({(3,)})) is None
    assert rc.noisy_gold(frozenset({("0",)})) is None
    assert rc.noisy_gold(frozenset({(0,), (1,)})) is None
    assert rc.noisy_gold(frozenset({(0, 0)})) is None


def test_skip_noisy_gold_removes_the_question_from_the_loss(tmp_path):
    a, m, ref, card, exs, gold, rw = _setup(tmp_path, skip_noisy_gold=True)
    gold["b"] = {**gold["b"], "gold_sql": "SELECT COUNT(*) FROM codes WHERE flag0 = 'nope'"}  # a single 0
    _, log, _ = _run(a, m, ref, card, exs, gold, rw)
    assert log["gold_noisy"] == 1 and log["gold_noisy_single_zero"] == 1 and log["q"] == 3


# ---- SQLite execution ---------------------------------------------------------------------------------------------

def test_exec_parallel_chunks_and_deadline(tmp_path):
    db = _shop_db(tmp_path / "shop.sqlite")
    rw = rc.ExecReward({"shop": db}, seconds=0.5, workers=2, min_chunk=2)
    gold = "SELECT COUNT(*) FROM orders"
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    sqls = [gold, "SELECT COUNT(id) FROM orders", "SELECT 1", "SELECT nope FROM orders", slow, None] + \
        [f"SELECT {i} + 100" for i in range(10)]
    t = time.time()
    out = rw.rewards([("q", "shop", gold, s) for s in sqls])
    took = time.time() - t
    rw.close()
    assert out[0] == (1.0, True, True) and out[1] == (1.0, True, True)
    assert out[2] == (0.1, True, False) and out[3] == (0.0, False, False)
    assert out[4] == (0.0, False, False)  # interrupted at its deadline
    assert out[5] == (0.0, False, False)
    assert rw.stats["chunks"] >= 2 and rw.stats["queries"] == 15
    assert took < 30
    assert rw.gold_rows("shop", gold) == frozenset({(12,)})


def test_timing_projection():
    a = rl_train.parse_args(["--plans", "x", "--db-roots", "y", "--out", "z", "--project-questions", "8000"])
    sp = {k[2:]: 0.1 for k in rl_train.PHASES}
    hist = [{"q": 32, "step_s": 99.0, "s_per_q": sp}] + [{"q": 32, "step_s": 32.0, "s_per_q": sp}] * 3
    rep = rl_train.timing_report(hist, a, startup_s=360.0)
    assert rep["steps_measured"] == 3 and rep["s_per_question"] == 1.0  # step 0 (warm-up) left out
    assert rep["gpu_h"]["1_epoch"] == round((8000 + 360) / 3600, 3)
    assert rep["gpu_h"]["2_epoch"] == round((16000 + 360) / 3600, 3)


# ---- end to end (tiny backbone, real BIRD-train plans, 2 steps) -----------------------------------------------

def test_end_to_end_two_steps(tmp_path):
    plans, dbs = ROOT / "data/sys1/bird_train", ROOT / "data/sys1/rrdbs/trains"
    if not (plans / "plans.jsonl").exists() or not dbs.exists():
        pytest.skip("no BIRD-train plans / trains database")
    drop = tmp_path / "drop.json"
    drop.write_text(json.dumps(["bird_train_train_690"]))
    out = tmp_path / "run"
    hist = rl_train.main(["--tiny", "--plans", str(plans), "--db-roots", str(dbs), "--only-db", "trains",
                          "--tiny-sft-steps", "10", "--timing-smoke", "2", "--G", "8", "--q-per-step", "8",
                          "--beta", "0.01", "--sft-weight", "0.5", "--list-weight", "0.5", "--buffer",
                          "--drop-ids", str(drop), "--skip-noisy-gold", "--device", "cpu", "--workers", "2",
                          "--seed", "3", "--out", str(out)])
    assert len(hist) == 2 and all(r["q"] > 0 and not r["nonfinite"] for r in hist)
    assert all(r["list_groups"] > 0 for r in hist)
    assert all(r["grad_norm_heads"] > 0 for r in hist)
    meta = json.loads((out / "rl_args.json").read_text())
    assert meta["seed"] == 3 and meta["git_rev"] and meta["filters"]["drop_ids"] == 1
    assert meta["buffer"]["gold_source"] == "bird_verified" and meta["buffer"]["dropped_heldout"] == 337
    tim = json.loads((out / "timing.json").read_text())
    assert tim["gpu_h"]["2_epoch"] > tim["gpu_h"]["1_epoch"] > 0
    assert json.loads((out / "oom.json").read_text())["questions_skipped"] == 0
