"""Validator trainer v2 (scripts/rerank_sq2.py train --tok-budget ...): the chunked gradient-cache path gives the same
loss and gradients as one un-chunked pass (tiny randomly initialised Qwen2 + LoRA Reranker, OmniSQL's architecture),
both with the schema prefix detached into leaves (full attention) and with the prefix recomputed per chunk; the
out-of-memory fallback restores gradients before retrying; the multipos loss equals listwise_loss with one positive.
CPU, a few seconds."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from luthor.sq2.rerank import gradcache_backward, listwise_loss, multipos_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rerank_sq2 as R  # noqa: E402


# ---- losses ---------------------------------------------------------------------------------------------------------

def test_multipos_equals_listwise_with_one_positive():
    g = torch.Generator().manual_seed(0)
    for n in (2, 5, 12):
        s = torch.randn(n, generator=g)
        for p in range(n):
            ok = [i == p for i in range(n)]
            assert float(multipos_loss(s, ok)) == pytest.approx(float(listwise_loss(s, ok)), rel=1e-6, abs=1e-6)


def test_multipos_is_minus_log_positive_mass_and_smoothing():
    s = torch.tensor([1.0, -0.5, 2.0, 0.3])
    ok = [True, False, True, False]
    p = torch.softmax(s, 0)
    assert float(multipos_loss(s, ok)) == pytest.approx(-math.log(float(p[0] + p[2])), rel=1e-6)
    eps = 0.05
    want = (1 - eps) * -math.log(float(p[0] + p[2])) + eps * float(-torch.log(p).mean())
    assert float(multipos_loss(s, ok, eps)) == pytest.approx(want, rel=1e-6)


# ---- gradient cache on a small stand-in scorer ------------------------------------------------------------------------

class Toy(torch.nn.Module):
    """prefix P = f(schema features); score_j = head(tanh(x_j W + g(P))); dropout on both paths."""

    def __init__(self, p_drop=0.0):
        super().__init__()
        torch.manual_seed(0)
        self.pre = torch.nn.Linear(6, 8)
        self.row = torch.nn.Linear(5, 8)
        self.mix = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 1)
        self.drop = torch.nn.Dropout(p_drop)
        self.schema = torch.randn(3, 6)
        self.x = torch.randn(11, 5)

    def prefix(self):
        return [self.drop(torch.tanh(self.pre(self.schema)))]

    def rows(self, P, idx):
        h = torch.tanh(self.drop(self.row(self.x[idx])) + self.mix(P[0].mean(0)))
        return self.head(h).squeeze(-1)


def _grads(m):
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def _toy_loss(s):
    oks = [[True, False, False, True], [False, True, False], [True, False, False, False]]
    out, o = 0, 0
    for ok in oks:
        out = out + multipos_loss(s[o:o + len(ok)], ok, 0.05)
        o += len(ok)
    return out / 3


@pytest.mark.parametrize("sizes", [[11], [4, 4, 3], [1] * 11, [6, 5]])
def test_gradcache_matches_one_pass_on_toy(sizes):
    m = Toy()
    m.zero_grad()
    full = _toy_loss(m.rows(m.prefix(), list(range(11))))
    full.backward()
    want = _grads(m)
    m.zero_grad()
    bounds = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    loss, s = gradcache_backward(m.prefix, lambda P, c: m.rows(P, list(range(bounds[c], bounds[c + 1]))),
                                 len(sizes), _toy_loss)
    assert loss == pytest.approx(float(full.detach()), rel=1e-6)
    got = _grads(m)
    assert want.keys() == got.keys() and "pre.weight" in got  # the prefix trains through the cached leaves
    for n in want:
        assert torch.allclose(want[n], got[n], rtol=1e-5, atol=1e-7), n


def test_gradcache_replays_dropout_masks():
    """With dropout, pass 2 recomputes each chunk with pass 1's masks (RNG replay), so its scores equal pass 1's."""
    m = Toy(p_drop=0.3).train()
    seen = []

    def chunk(P, c):
        s = m.rows(P, list(range(4 * c, min(4 * c + 4, 11))))
        seen.append(s.detach().clone())
        return s
    gradcache_backward(m.prefix, chunk, 3, _toy_loss)
    assert len(seen) == 6
    for a, b in zip(seen[:3], seen[3:]):
        assert torch.equal(a, b)
    seen.clear()  # replay_prefix: the prefix recomputed inside each chunk gets the same masks every time

    def chunk2(P, c):
        P = m.prefix()
        seen.append(P[0].detach().clone())
        return m.rows(P, list(range(4 * c, min(4 * c + 4, 11))))
    gradcache_backward(lambda: [], chunk2, 3, _toy_loss, replay_prefix=True)
    assert all(torch.equal(seen[0], p) for p in seen)


# ---- the real Reranker (tiny Qwen2 + LoRA) through db_step_v2 ------------------------------------------------------------

def _rr():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model

    from luthor.sq.model import Backbone
    from luthor.sq2.rerank import Reranker
    from test_sq2_r1_levers import _tok_dir
    tok = AutoTokenizer.from_pretrained(_tok_dir())
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(0)
    body = get_peft_model(Qwen2Model(cfg), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear"))
    rr = Reranker(Backbone(body, tok), lora_r=4)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in rr.named_parameters():
            if ".rerank." in n and "lora_B" in n:
                p.normal_(0, 0.05, generator=g)
        rr.score.weight.normal_(0, 0.5, generator=g)
    rr.activate(train=True)
    rr.eval()  # LoRA dropout off: one pass and the chunks then draw no masks, so gradients compare exactly
    return rr


def _recs():
    def c(sql, ok, **kw):
        return {"sql": sql, "runs": True, "nonnull": True, "ok": ok, **kw}
    return [
        {"id": "q1", "db_id": "shop", "question": "total amount of orders from Egypt", "context": "Egypt is a country",
         "weight": 3.0, "cands": [c('SELECT SUM("amount") FROM "orders"', False),
                                   c('SELECT SUM(T1."amount") FROM "orders" AS T1 JOIN "customers" AS T2 ON T1."customer_id" = T2."id" WHERE T2."country" = \'Egypt\'', True),
                                   c('SELECT COUNT(*) FROM "orders"', False),
                                   c('SELECT SUM(DISTINCT T1."amount") FROM "orders" AS T1', False, mutant=True),
                                   c('SELECT 1', False, runs=False)]},  # dead: not in the list
        {"id": "q2", "db_id": "shop", "question": "how many customers", "context": "",
         "cands": [c('SELECT COUNT(*) FROM "customers"', True), c('SELECT COUNT("id") FROM "customers"', True),
                   c('SELECT COUNT(DISTINCT "country") FROM "customers"', False)]},
        {"id": "q3", "db_id": "shop", "question": "names of customers in Kenya", "context": "",
         "cands": [c('SELECT "name" FROM "customers" WHERE "country" = \'Kenya\'', True),
                   c('SELECT "name" FROM "customers"', False), c('SELECT "id" FROM "customers"', False),
                   c('SELECT "name", "id" FROM "customers" WHERE "country" = \'Kenya\'', False)]},
    ]


def _step(rr, budget, loss="multipos", monkeypatch=None, hybrid=False):
    from test_sq2_r1_levers import _card
    if hybrid:
        monkeypatch.setattr(R, "full_attention", lambda body: False)
    a = SimpleNamespace(budget_chars=9000, tok_budget=budget, loss=loss, label_smooth=0.05)
    rr.zero_grad(set_to_none=True)
    stats = R.new_stats()
    l, n = R.db_step_v2(rr, _card(), _recs(), a, 1.0 / 3, 1.5, stats)
    grads = {k: p.grad.clone() for k, p in rr.named_parameters() if p.grad is not None}
    return l, n, grads, stats


@pytest.mark.parametrize("loss", ["multipos", "listwise"])
@pytest.mark.parametrize("hybrid", [False, True])
def test_db_step_gradcache_matches_one_pass(loss, hybrid, monkeypatch):
    rr = _rr()
    l0, n0, g0, s0 = _step(rr, 10**9, loss)
    assert s0["direct"] == 1 and s0["gradcache"] == 0 and n0 == 3
    l1, n1, g1, s1 = _step(rr, 1, loss, monkeypatch, hybrid)  # budget 1: every row its own chunk
    assert s1["gradcache"] == 1 and s1["chunks"] == 11 and s1["oom"] == 0 and not s1["skipped"]
    assert l1 == pytest.approx(l0, rel=1e-5) and n1 == n0
    assert g0.keys() == g1.keys() and any(".rerank." in k for k in g0) and "score.weight" in g0
    assert not any(".default." in k for k in g0)  # only the rerank adapter and the head train
    _same(g0, g1)


def _same(g0, g1, shift=0.0):
    """Equal up to float32 rounding (the chunks run other batch shapes): relative error ~2e-5 observed."""
    for k in g0:
        d = g1[k] - shift - g0[k]
        assert torch.allclose(g0[k], g1[k] - shift, rtol=1e-3, atol=1e-5), k
        assert k == "score.bias" or float(d.norm() / g0[k].norm()) < 1e-4, k


def test_db_step_middle_budget_and_oom_fallback(monkeypatch):
    """A budget that packs several rows per chunk gives the same gradients; an out-of-memory error in the first
    (one-pass) try restores the gradients already accumulated and retries with a halved budget."""
    rr = _rr()
    _, _, g0, _ = _step(rr, 10**9)
    from test_sq2_r1_levers import _card
    card = _card()
    P = len(rr.backbone.tok(R.schema_text(card, None).text, add_special_tokens=False)["input_ids"])
    _, _, g1, s1 = _step(rr, R.chunk_cost(4, 60, P))
    assert 1 < s1["chunks"] < 11
    _same(g0, g1)
    real, calls = type(rr).scores, []

    def flaky(self, *args, **kw):
        calls.append(1)
        if len(calls) == 1:
            out = real(self, *args, **kw)
            out.sum().backward()  # a partial backward before failing: these gradients must be undone
            raise torch.OutOfMemoryError("simulated")
        return real(self, *args, **kw)
    monkeypatch.setattr(type(rr), "scores", flaky)
    rows = R.row_ids(rr.backbone.tok, [R.item(r, c) for r in _recs() for c in R.list_items(r)])
    assert len(rows) == 11
    # fits one pass exactly; after the simulated out-of-memory the halved budget goes through the gradient cache
    a = SimpleNamespace(budget_chars=9000, tok_budget=R.chunk_cost(11, max(map(len, rows)), P), loss="multipos",
                        label_smooth=0.05)
    rr.zero_grad(set_to_none=True)
    for p in rr.parameters():  # gradients from an earlier database in the same step
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    stats = R.new_stats()
    R.db_step_v2(rr, card, _recs(), a, 1.0 / 3, 1.5, stats)
    assert stats["oom"] == 1 and stats["gradcache"] == 1 and not stats["skipped"]
    _same(g0, {k: p.grad for k, p in rr.named_parameters() if k in g0}, shift=1.0)


# ---- batching helpers ---------------------------------------------------------------------------------------------------

def test_plan_chunks_cover_rows_within_budget():
    lens, P = [37, 120, 5, 80, 80, 300, 12, 64], 900
    for budget in (1, 5000, 20000, 60000, 10**9):
        chunks = R.plan_chunks(lens, P, budget)
        assert sorted(i for c in chunks for i in c) == list(range(len(lens)))
        for c in chunks:
            assert len(c) == 1 or R.chunk_cost(len(c), max(lens[i] for i in c), P) <= budget
    assert len(R.plan_chunks(lens, P, 10**9)) == 1 and len(R.plan_chunks(lens, P, 1)) == len(lens)


def test_folds_stable_and_partition():
    dbs = [f"db{i}" for i in range(200)]
    f = [R.fold_of(d, 2) for d in dbs]
    assert f == [R.fold_of(d, 2) for d in dbs] and 60 < sum(f) < 140
    assert R.fold_of("works_cycles", 2) == int(__import__("hashlib").md5(b"works_cycles").hexdigest(), 16) % 2
    assert R.parse_fold("1/2") == (1, 2)
    with pytest.raises(SystemExit):
        R.parse_fold("2/2")


def test_v2_only_behind_new_flags():
    old = SimpleNamespace(train_file=None, tok_budget=0, loss="listwise", sched="legacy", val_file=None, fold="")
    assert not R.v2_requested(old)
    for k, v in [("train_file", Path("x")), ("tok_budget", 1000), ("loss", "multipos"), ("sched", "cosine"),
                 ("val_file", Path("v")), ("fold", "0/2")]:
        assert R.v2_requested(SimpleNamespace(**{**vars(old), k: v}))


def test_cosine_schedule_reaches_zero():
    a = SimpleNamespace(sched="cosine", warmup=0.05)
    f = [R.lr_factor(a, s, 200) for s in range(201)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0) and max(f) == pytest.approx(1.0)
    assert f[200] == pytest.approx(0.0, abs=1e-12) and all(x >= y for x, y in zip(f[10:], f[11:]))
