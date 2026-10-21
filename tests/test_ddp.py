"""train_sq2.py data parallelism (torchrun), on a tiny randomly initialised Qwen2 (OmniSQL's architecture), CPU:
- a step's databases split between processes, each weighted 1/len(chunk), sum to the single-process gradient;
- two real processes (torchrun, gloo) take the same AdamW step as one process over the whole chunk, and stay equal;
- an OOM in backward on ONE process drops the step on BOTH (no deadlock, weights untouched, questions counted);
- one process passing --max-oom-skip stops both;
- --auto-resume's state round trip.
The two-process test runs this file as the torchrun worker (python -m torch.distributed.run ... test_ddp.py OUT)."""
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_sq2 as ts  # noqa: E402
import luthor.sq2.train as tr  # noqa: E402
from luthor.sq2.train import BackwardOOM  # noqa: E402
from test_sq2_r1_levers import _card, _rows, _tiny  # noqa: E402


class NoShuffle(random.Random):
    """train_tables shuffles a schema's table order 30% of the time from the step's rnd; the processes' streams
    differ from the single process's, so the equivalence checks keep the order fixed."""

    def random(self):
        return 0.99


def _setup(n_db: int = 4):
    cards, chunk = {}, []
    for j in range(n_db):
        card = _card()
        card.db_id = f"shop{j}"
        cards[card.db_id] = card
        rows = [{**r, "id": f"{r['id']}{j}", "db_id": card.db_id} for r in _rows()]
        chunk.append((card.db_id, [tr.make_example(r, card, random.Random(0), order="v2") for r in rows]))
    return cards, chunk


def _model():
    m = _tiny(value_side=True, seed=0)
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    return m, params, opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)


def _record(m, setattr_=setattr) -> dict:
    """The gradients train_step clips (after the all-reduce, before clipping, which would hide a scale error, as
    AdamW would), by parameter name."""
    got, clip = {}, torch.nn.utils.clip_grad_norm_

    def rec(ps, *a, **k):
        got.clear()
        got.update({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
        return clip(ps, *a, **k)
    setattr_(torch.nn.utils, "clip_grad_norm_", rec)
    return got


def _flat(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_shards_sum_to_the_single_process_gradient():
    cards, chunk = _setup()
    m, params, _, _ = _model()
    led = ts.OOMLedger(1.0)
    ts.step_grads(m, chunk, chunk, cards, 10**6, NoShuffle(), led)
    full = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    for rank in range(2):  # the two processes' shares, accumulated (what the all-reduce sums)
        ts.step_grads(m, chunk, chunk[rank::2], cards, 10**6, NoShuffle(), led)
    summed = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    assert full.keys() == summed.keys() and len(full) > 10
    for n in full:
        assert torch.allclose(full[n], summed[n], rtol=1e-4, atol=1e-7), n


def test_single_process_dist_is_a_no_op(monkeypatch):
    for v in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(v, raising=False)
    d = ts.Dist().setup()
    assert d.world == 1 and d.main and d.shard([1, 2, 3]) == [1, 2, 3] and d.reduce([1.0, 2.0]) == [1.0, 2.0]
    assert not torch.distributed.is_initialized()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_two_processes_match_one(tmp_path, monkeypatch):
    cards, chunk = _setup()
    m, params, opt, sched = _model()
    g_ref = _record(m, monkeypatch.setattr)
    ref = ts.train_step(m, opt, sched, chunk, cards, 10**6, NoShuffle(), ts.OOMLedger(1.0), ts.Dist(), params)
    assert ref[0] == "ok"
    want = _flat(m)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    for v in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        env.pop(v, None)
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_addr=127.0.0.1",
                        f"--master_port={_free_port()}", __file__, str(tmp_path)],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = [json.loads((tmp_path / f"rank{k}.json").read_text()) for k in range(2)]
    after = [torch.load(tmp_path / f"rank{k}_step1.pt") for k in range(2)]
    grads = [torch.load(tmp_path / f"rank{k}_grads1.pt") for k in range(2)]
    assert grads[0].keys() == g_ref.keys()  # also: a parameter no process used keeps grad None (AdamW skips it)
    for n in g_ref:  # summed, not averaged: the single-process gradient
        assert torch.equal(grads[0][n], grads[1][n]), n
        assert torch.allclose(grads[0][n], g_ref[n], rtol=1e-4, atol=1e-7), n
    for n in want:  # identical across the processes, and the single-process step
        assert torch.equal(after[0][n], after[1][n]), n
        assert torch.allclose(after[0][n], want[n], rtol=1e-4, atol=1e-6), n
    assert all(x["step1"] == "ok" for x in res)
    assert res[0]["loss1"] == pytest.approx(ref[1], rel=1e-5)  # the logged loss is the whole chunk's
    # OOM in backward on rank 1 only: both drop the step, nobody hangs, weights unchanged, all 12 questions counted
    assert [x["step2"] for x in res] == ["dropped", "dropped"]
    assert all(x["unchanged2"] and x["dropped_steps"] == 1 for x in res)
    assert sum(x["skipped"] for x in res) == sum(len(es) for _, es in chunk)
    assert OOMLedgerMerged(res)["dropped_steps"] == 1
    # rank 1 alone over --max-oom-skip: both stop
    assert all(x["step3"].startswith("SystemExit") for x in res)
    assert "another process" in res[0]["step3"] and "OOM:" in res[1]["step3"]


def OOMLedgerMerged(res):
    return ts.OOMLedger.merged([x["ledger"] for x in res])


def _worker(out: Path):
    d = ts.Dist().setup(5)
    assert d.world == 2
    cards, chunk = _setup()
    m, params, opt, sched = _model()
    d.broadcast_params(params)
    grads = _record(m)
    led = ts.OOMLedger(1.0, defer=True)
    res = {}
    res["step1"], res["loss1"], _ = ts.train_step(m, opt, sched, chunk, cards, 10**6, NoShuffle(), led, d, params)
    torch.save(_flat(m), out / f"rank{d.rank}_step1.pt")
    torch.save(dict(grads), out / f"rank{d.rank}_grads1.pt")

    real = ts.train_db
    bad = chunk[1][0]  # rank 1's database

    def flaky(model, card, es, *a, **k):
        if card.db_id == bad:
            raise BackwardOOM(bad)
        return real(model, card, es, *a, **k)
    ts.train_db = flaky
    before = _flat(m)
    res["step2"] = ts.train_step(m, opt, sched, chunk, cards, 10**6, NoShuffle(), led, d, params)[0]
    res["unchanged2"] = all(torch.equal(before[n], p) for n, p in _flat(m).items())
    res["dropped_steps"], res["skipped"], res["ledger"] = led.dropped_steps, led.skipped, led.summary()
    ts.train_db = real

    led3 = ts.OOMLedger(0.001, min_seen=1, defer=True)
    if d.rank == 1:
        led3.seen, led3.skipped = 10, 5
    try:
        ts.train_step(m, opt, sched, chunk, cards, 10**6, NoShuffle(), led3, d, params)
        res["step3"] = "ok"
    except SystemExit as e:
        res["step3"] = f"SystemExit: {e}"
    (out / f"rank{d.rank}.json").write_text(json.dumps(res))
    d.close()


def test_resume_state_round_trip(tmp_path):
    """save_resume / load_resume: weights, optimizer, schedule, random streams and position come back."""
    from peft import LoraConfig, get_peft_model
    from luthor.sq.model import Backbone
    from luthor.sq2.model import LuthorSQ2
    from transformers import Qwen2Config, Qwen2Model
    tok = _tiny().backbone.tok

    def build():
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2)
        body = get_peft_model(Qwen2Model(cfg), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear"))
        m = LuthorSQ2(Backbone(body, tok), d_proj=32, col_mentions=True, value_side=True)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3)
        return m, params, opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)

    cards, chunk = _setup(2)
    a = ts.parser().parse_args(["--plans", "p", "--dev-dbs", "d", "--out", str(tmp_path), "--auto-resume"])
    m, params, opt, sched = build()
    rnd = random.Random(5)
    assert ts.train_step(m, opt, sched, chunk, cards, 10**6, rnd, ts.OOMLedger(1.0), ts.Dist(), params)[0] == "ok"
    led = ts.OOMLedger(1.0)
    led.seen = 6
    ts.save_resume(tmp_path, m, opt, sched, ts.Dist(), rnd, a, step=1, best=0.25, epoch=0, pos=2, total=10, span=10,
                   elapsed=3.0, ledger=led.summary())
    x = rnd.random()
    m2, p2, opt2, sched2 = build()
    rnd2 = random.Random(9)
    st = ts.load_resume(tmp_path, m2, opt2, sched2, ts.Dist(), rnd2, a)
    assert (st["step"], st["pos"], st["best"], st["ledger"]["questions_seen"]) == (1, 2, 0.25, 6)
    assert rnd2.random() == x and sched2.last_epoch == sched.last_epoch
    for (n, p), q in zip(m.named_parameters(), m2.parameters()):
        assert torch.equal(p, q), n
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert s1.keys() == s2.keys() and all(torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1)
    b = ts.parser().parse_args(["--plans", "p", "--dev-dbs", "d", "--out", str(tmp_path), "--auto-resume", "--seed", "3"])
    with pytest.raises(SystemExit, match="seed"):
        ts.load_resume(tmp_path, *build()[0:1], opt2, sched2, ts.Dist(), rnd2, b)


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
