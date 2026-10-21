"""VAL3 trainer flags (scripts/rerank_sq2.py, docs/paper_notes/research/validator/val3_trainer.md): --seed / --init-seed,
--eval-every + --avg-last (tail checkpoint average), --joint-results (results table in the shared prefix). Tiny random
Qwen2 + LoRA Reranker on the CPU (the real trainer end to end, Backbone loading and .cuda() patched); a few seconds.
Defaults must be byte-identical to the trainer before these flags."""
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from luthor.sq2 import rerank as RR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import rerank_sq2 as R  # noqa: E402


# ---- defaults: prompts and scores unchanged --------------------------------------------------------------------------

def test_candidate_text_frozen():
    assert RR.candidate_text("how many shops", "shop = store", "SELECT 1") == (
        "\nQuestion: how many shops\nContext: shop = store\nCandidate SQL: SELECT 1\n"
        "Does this SQL answer the question exactly?")
    assert RR.candidate_text("q", "", "SELECT 2", "1 row x 1 col (integer); first: (3)") == (
        "\nQuestion: q\nCandidate SQL: SELECT 2\nResult: 1 row x 1 col (integer); first: (3)\n"
        "Does this SQL answer the question exactly?")


def _old_scores(self, card, items, tables=None, max_tok=384):
    """Reranker.scores before the scores_ids refactor (verbatim)."""
    from luthor.sq2.prefix_cache import expand_cache, release_cache
    st = RR.schema_text(card, tables)
    tok = self.backbone.tok
    ids = tok(st.text, add_special_tokens=False)["input_ids"]
    out = self.body(input_ids=torch.tensor([ids], device=self.device), use_cache=True)
    rows = [tok(RR.candidate_text(*it), add_special_tokens=False)["input_ids"][-max_tok:] for it in items]
    n, T = len(rows), max(len(r) for r in rows)
    x = torch.full((n, T), tok.pad_token_id, dtype=torch.long, device=self.device)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r, device=self.device)
    cache = expand_cache(out.past_key_values, n)
    pos = torch.arange(len(ids), len(ids) + T, device=self.device)[None].expand(n, -1)
    try:
        H = self.body(input_ids=x, past_key_values=cache, position_ids=pos, use_cache=True).last_hidden_state
    finally:
        release_cache(cache)
    last = torch.tensor([len(r) - 1 for r in rows], device=self.device)
    h = H[torch.arange(n, device=self.device), last].float()
    return self.score(h).squeeze(-1)


def test_default_scores_and_prompts_unchanged():
    import test_rerank_gradcache as T
    from test_sq2_r1_levers import _card
    rr, card = T._rr(), _card()
    items = [R.item(r, c) for r in T._recs() for c in R.list_items(r)]
    with torch.no_grad():
        assert torch.equal(rr.scores(card, items), _old_scores(rr, card, items))
    assert not R.JOINT["on"]
    a = SimpleNamespace(budget_chars=9000, tok_budget=10**9, loss="group", label_smooth=0.05)
    seen = []
    real = type(rr).scores_ids

    def spy(self, ids, rows):
        seen.append((ids, rows))
        return real(self, ids, rows)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(rr), "scores_ids", spy)
        R.db_step_v2(rr, card, T._recs(), a, 1.0, 1.0, R.new_stats())
    tok = rr.backbone.tok
    want_rows = [tok(RR.candidate_text(*it), add_special_tokens=False)["input_ids"][-384:] for it in items]
    assert seen == [(tok(RR.schema_text(card, None).text, add_special_tokens=False)["input_ids"], want_rows)]


def test_new_flags_do_not_switch_v2_by_default_and_seed_does_not():
    old = SimpleNamespace(train_file=None, tok_budget=0, loss="listwise", sched="legacy", val_file=None, fold="",
                          seed=3, joint_results=False, eval_every=0.0, avg_last=0)
    assert not R.v2_requested(old)
    for k, v in [("joint_results", True), ("eval_every", 0.5), ("avg_last", 4)]:
        assert R.v2_requested(SimpleNamespace(**{**vars(old), k: v}))


# ---- seed -------------------------------------------------------------------------------------------------------------

def test_seed_defaults_keep_rng_stream():
    a = SimpleNamespace(seed=0, init_seed=None)
    assert R.init_seed(a) == 0
    torch.manual_seed(0)
    torch.rand(3)
    st = torch.get_rng_state()
    R.reseed_after_init(a)  # seed == init seed: no reseed, the stream continues exactly as before
    assert torch.equal(torch.get_rng_state(), st)
    b = SimpleNamespace(seed=1, init_seed=0)
    assert R.init_seed(b) == 0 and R.init_seed(SimpleNamespace(seed=2, init_seed=None)) == 2
    R.reseed_after_init(b)
    torch.manual_seed(1)
    want = torch.rand(4)
    R.reseed_after_init(b)
    assert torch.equal(torch.rand(4), want)


def _tiny_backbone(*args, **kw):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model

    from luthor.sq.model import Backbone
    from test_sq2_r1_levers import _tok_dir
    tok = AutoTokenizer.from_pretrained(_tok_dir())
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    body = get_peft_model(Qwen2Model(cfg), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear"))
    return Backbone(body, tok)


class _CpuReranker(RR.Reranker):
    def cuda(self, *a, **k):
        return self


def _lists():
    import test_rerank_gradcache as T
    recs = T._recs()
    for r in recs:
        for k, c in enumerate(r["cands"]):
            c["sig"] = f"{r['id']}_{k}"
            c["res"] = f"{k + 1} rows x 1 col (integer); first: ({k})"
    recs[1]["cands"][1]["sig"] = recs[1]["cands"][0]["sig"]
    recs[1]["cands"][1].pop("res")  # a pair member: its class line comes from the other member
    return recs


def _run(tmp_path, monkeypatch, name, *extra):
    from test_sq2_r1_levers import _card
    d = tmp_path / "cards" / "cards"
    d.mkdir(parents=True, exist_ok=True)
    _card().save(d / "shop.json")
    lists = tmp_path / "lists.jsonl"
    lists.write_text("".join(json.dumps(r) + "\n" for r in _lists()))
    out = tmp_path / name
    monkeypatch.setattr(R.Backbone, "from_pretrained", _tiny_backbone)
    monkeypatch.setattr(R, "Reranker", _CpuReranker)
    monkeypatch.setattr(sys, "argv", ["rerank_sq2.py", "train", "--cards", str(tmp_path / "cards"), "--train-file",
                                      str(lists), "--val-file", str(lists), "--out", str(out), "--loss", "group",
                                      "--label-smooth", "0.05", "--sched", "cosine", "--tok-budget", "60000",
                                      "--q-per-db", "1", "--dbs-per-step", "1", "--lr", "5e-3", "--head-lr", "5e-2",
                                      *extra])
    R.main()
    head = torch.load(out / "rerank_head.pt")
    return out, head, json.loads((out / "val_history.json").read_text())


def test_seed_determinism_end_to_end(tmp_path, monkeypatch):
    _, h0, v0 = _run(tmp_path, monkeypatch, "s0a", "--epochs", "1")
    _, h0b, v0b = _run(tmp_path, monkeypatch, "s0b", "--epochs", "1", "--seed", "0")
    _, h1, _ = _run(tmp_path, monkeypatch, "s1", "--epochs", "1", "--seed", "1", "--init-seed", "0")
    assert all(torch.equal(h0[k], h0b[k]) for k in h0) and v0 == v0b
    assert not all(torch.equal(h0[k], h1[k]) for k in h0)  # other data order (same init)


# ---- half-epoch evaluation + tail average -------------------------------------------------------------------------------

def test_eval_points():
    assert R.eval_points(244, 0) == {} and R.eval_points(244, 1.0) == {}
    assert R.eval_points(244, 0.5) == {122: 0.5}
    assert R.eval_points(10, 0.25) == {2: 0.25, 5: 0.5, 8: 0.75}  # round half to even
    assert R.eval_points(1, 0.5) == {}  # an epoch of one step: the epoch end only


def test_average_states_arithmetic():
    g = torch.Generator().manual_seed(0)
    sts = [{"a": torch.randn(3, 4, generator=g).to(torch.bfloat16), "b": torch.randn(5, generator=g)} for _ in range(4)]
    avg = RR.average_states(sts)
    assert avg["a"].dtype == torch.bfloat16 and avg["b"].dtype == torch.float32
    want_a = (sum(s["a"].float() for s in sts) / 4).to(torch.bfloat16)
    assert torch.equal(avg["a"], want_a)
    assert torch.allclose(avg["b"], torch.stack([s["b"] for s in sts]).mean(0), atol=1e-7)
    assert torch.equal(RR.average_states(sts[:1])["b"], sts[0]["b"])
    with pytest.raises(ValueError):
        RR.average_states([])


def test_half_epoch_eval_and_tail_average_end_to_end(tmp_path, monkeypatch):
    from peft import set_peft_model_state_dict  # noqa: F401  (peft present)
    from safetensors.torch import load_file
    out, head, hist = _run(tmp_path, monkeypatch, "avg", "--epochs", "2", "--eval-every", "0.5", "--avg-last", "2")
    # 3 lists, one per step: 3 steps per epoch, val at step 2 (0.5 epoch, rounded) and 3 (epoch end), ...
    assert [h["tag"] for h in hist] == ["epoch 0.5", "epoch 1", "epoch 1.5", "epoch 2", "avg_last2"]
    assert [h["step"] for h in hist][:4] == [2, 3, 5, 6] and all("group_nll" in h for h in hist)
    ck = sorted(p.name for p in (out / "ckpt").iterdir())
    assert ck == ["step000005", "step000006"] and json.loads((out / "avg_steps.json").read_text()) == [5, 6]
    heads = [torch.load(out / "ckpt" / c / "rerank_head.pt") for c in ck]
    for k in head:
        assert torch.allclose(head[k], (heads[0][k] + heads[1][k]) / 2, atol=1e-6)
    loras = [load_file(str(out / "ckpt" / c / "rerank_lora" / "rerank" / "adapter_model.safetensors")) for c in ck]
    got = load_file(str(out / "rerank_lora" / "rerank" / "adapter_model.safetensors"))
    assert got.keys() == loras[0].keys() and len(got) > 0
    for k in got:
        want = ((loras[0][k].float() + loras[1][k].float()) / 2).to(got[k].dtype)
        assert torch.equal(got[k], want), k
    assert (out / "best" / "rerank_head.pt").exists() and (out / "last" / "rerank_head.pt").exists()
    st = json.loads((out / "settings.json").read_text())
    for sub in ("best", "last", "ckpt/step000006"):
        assert json.loads((out / sub / "settings.json").read_text()) == st
    assert st["avg_last"] == 2 and st["eval_every"] == 0.5 and st["seed"] == 0 and st["joint_results"] is False


# ---- joint scoring --------------------------------------------------------------------------------------------------------

def _c(sql, sig, res=None, ok=False, runs=True, **kw):
    return {"sql": sql, "sig": sig, "res": res, "ok": ok, "runs": runs, "nonnull": runs, **kw}


def test_result_classes_and_joint_texts():
    cands = [_c("S1", "a", "1 row x 1 col (integer); first: (3)", ok=True), _c("S2", "b", "0 rows"),
             _c("S3", "a"),  # pair member of class a, no preview of its own
             _c("S4", None, runs=False), _c("S5", "m", "2 rows x 1 col (text); first: ('x')", mutant=True, runs=False)]
    keys, lines, of = RR.result_classes(cands)
    assert keys == ["a", "b", "m"] and of == ["a", "b", "a", None, "m"]
    assert lines == ["1 row x 1 col (integer); first: (3)", "0 rows", "2 rows x 1 col (text); first: ('x')"]
    head, rows = RR.joint_texts("how many", "evidence here", cands, [2, 0])  # b not shown
    assert head == ("\nQuestion: how many\nContext: evidence here" + RR.JOINT_HEAD
                    + "R1: 2 rows x 1 col (text); first: ('x')\nR2: 1 row x 1 col (integer); first: (3)\n")
    assert rows == [RR.joint_row_text("S1", "R2"), RR.joint_row_text("S2", "not shown"), RR.joint_row_text("S3", "R2"),
                    RR.joint_row_text("S4", "error"), RR.joint_row_text("S5", "R1")]
    assert rows[0] == "\nCandidate SQL: S1\nResult: R2\nDoes this SQL answer the question exactly?"


def test_joint_truncation():
    long = "x" * 500
    cands = [_c(f"S{k}", f"g{k}", long) for k in range(25)]
    _, lines, _ = RR.result_classes(cands)
    assert all(len(line) == RR.JOINT_RES_CHARS for line in lines)
    shown = R.joint_train_shown(25, random.Random(0))
    assert len(shown) == RR.JOINT_MAX_RESULTS == len(set(shown)) and set(shown) <= set(range(25))
    assert sorted(R.joint_train_shown(5, random.Random(0))) == [0, 1, 2, 3, 4]
    head, rows = RR.joint_texts("q" * 5000, "", cands, shown)
    assert head.startswith("\nQuestion: " + "q" * (RR.JOINT_Q_CHARS - len("\nQuestion: ")) + RR.JOINT_HEAD)
    assert sum(r.endswith("not shown\nDoes this SQL answer the question exactly?") for r in rows) == 25 - 16
    assert head.count("\nR") == 16
    # structured rows (when a list carries them): columns, rows and cells capped
    c = {"runs": True, "sig": "s", "rows": [list(range(9)), ["y" * 99] * 9, [None] * 9, [1] * 9], "n_rows": 40}
    line = RR.result_line(c)
    assert line.startswith("40 rows x 9 cols (integer, integer, integer, integer, integer, integer); rows: (0, 1, 2, 3, 4, 5, ...)")
    assert len(line) <= RR.JOINT_RES_CHARS and line.count("(") <= 1 + RR.JOINT_ROWS
    assert RR.result_line({"runs": True, "rows": []}) == "0 rows"


def test_joint_orders_deterministic_and_reversed():
    r = {"id": "bird_train_7"}
    o = R.joint_orders(r, 9, 2)
    assert o == R.joint_orders(r, 9, 2) and o[1] == o[0][::-1] and sorted(o[0]) == list(range(9))
    assert R.joint_orders(r, 9, 1) == o[:1] and len(R.joint_orders(r, 9, 3)) == 3


def test_joint_prefix_ids_and_token_size():
    from test_sq2_r1_levers import _card, _tok_dir
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_tok_dir())
    card = _card()
    cands = [_c(f"S{k}", f"g{k}", "y" * 500) for k in range(40)]
    head, _ = RR.joint_texts("q", "", cands, list(range(16)))
    ids = RR.joint_prefix_ids(tok, card, None, head)
    base = len(tok(RR.schema_text(card, None).text, add_special_tokens=False)["input_ids"])
    assert ids[:5] == tok(RR.schema_text(card, None).text, add_special_tokens=False)["input_ids"][:5]
    assert len(ids) - base < 16 * (RR.JOINT_RES_CHARS // 2 + 8) + 60  # the table's worst case stays bounded


def _joint_step(rr, budget, loss="group", hybrid=False, monkeypatch=None):
    from test_sq2_r1_levers import _card
    if hybrid:
        monkeypatch.setattr(R, "full_attention", lambda body: False)
    a = SimpleNamespace(budget_chars=9000, tok_budget=budget, loss=loss, label_smooth=0.05, joint_results=True, seed=0)
    rr.zero_grad(set_to_none=True)
    stats = R.new_stats()
    l, n = R.db_step_v2(rr, _card(), _lists(), a, 1.0 / 3, 1.5, stats)
    return l, n, {k: p.grad.clone() for k, p in rr.named_parameters() if p.grad is not None}, stats


@pytest.mark.parametrize("hybrid", [False, True])
def test_joint_step_group_loss_gradcache_matches_one_pass(hybrid, monkeypatch):
    import test_rerank_gradcache as T
    rr = T._rr()
    l0, n0, g0, s0 = _joint_step(rr, 10**9)
    assert n0 == 3 and s0["direct"] == 3 and s0["gradcache"] == 0  # one prefix (and pass) per question
    l1, n1, g1, s1 = _joint_step(rr, 1, hybrid=hybrid, monkeypatch=monkeypatch)
    assert s1["gradcache"] == 3 and s1["chunks"] == 11 and not s1["skipped"]
    assert l1 == pytest.approx(l0, rel=1e-5) and n1 == n0
    assert any(".rerank." in k for k in g0) and "score.weight" in g0
    T._same(g0, g1)
    # the joint loss is a different function of the weights than the pointwise one
    a = SimpleNamespace(budget_chars=9000, tok_budget=10**9, loss="group", label_smooth=0.05)
    from test_sq2_r1_levers import _card
    lp, _ = R.db_step_v2(rr, _card(), _lists(), a, 1.0 / 3, 1.5, R.new_stats())
    assert abs(lp - l0) > 1e-5


def test_joint_score_file_averages_orders(monkeypatch, tmp_path):
    import test_rerank_gradcache as T
    from test_sq2_r1_levers import _card
    rr = T._rr()
    d = tmp_path / "cards"
    d.mkdir()
    _card().save(d / "shop.json")
    paths = {"shop": d / "shop.json"}
    recs = _lists()
    monkeypatch.setitem(R.JOINT, "on", True)
    for n in (1, 2):
        monkeypatch.setitem(R.JOINT, "orders", n)
        R.score_file(rr, recs, paths, 9000)
        if n == 1:
            one = [c["rr"] for r in recs for c in R.unique(r["cands"])]
    two = [c["rr"] for r in recs for c in R.unique(r["cands"])]
    assert len(one) == len(two) == 12 and one != two
    # the 2-order score is the mean of the two single orders, each scored by hand
    tok = rr.backbone.tok
    rr.eval()  # score_file leaves the model in train mode (LoRA dropout on)
    r = recs[2]
    cs = R.unique(r["cands"])
    o = R.joint_orders(r, len(RR.result_classes(cs)[0]), 2)
    per = []
    for shown in o:
        head, texts = RR.joint_texts(r["question"], r["context"], cs, shown)
        rows = [tok(t, add_special_tokens=False)["input_ids"][-384:] for t in texts]
        with torch.no_grad():
            per.append(rr.scores_ids(RR.joint_prefix_ids(tok, _card(), None, head), rows).float())
    got = torch.tensor([c["rr"] for c in cs])
    assert torch.allclose(got, (per[0] + per[1]) / 2, atol=1e-5)
    monkeypatch.setitem(R.JOINT, "on", False)
    R.score_file(rr, recs, paths, 9000)
    assert [c["rr"] for r in recs for c in R.unique(r["cands"])] != two


def test_joint_train_end_to_end_writes_settings(tmp_path, monkeypatch):
    out, _, hist = _run(tmp_path, monkeypatch, "joint", "--epochs", "1", "--joint-results")
    st = json.loads((out / "settings.json").read_text())
    assert st["joint_results"] is True and st["joint_orders"] == 2 and st["joint_max_results"] == RR.JOINT_MAX_RESULTS
    assert hist and hist[-1]["tag"] == "epoch 1"
    R.JOINT.update(on=False, orders=2)
