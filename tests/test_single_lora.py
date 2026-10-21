"""scripts/single_lora.py end to end on CPU (tiny Qwen2 + LoRA): an R1-style generator run is the init; the single-LoRA
run trains one adapter on generator replay + validator lists, and its out dir loads both as a generator run
(eval_sq2.load_model) and as a shared-adapter validator (rerank_sq2.load_validator) that holds no second adapter."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import rerank_sq2 as R  # noqa: E402
import single_lora as SL  # noqa: E402
from test_train_sq2_main import _data, _run, cpu  # noqa: E402,F401


def _lists(tmp: Path) -> tuple[Path, Path]:
    from test_sq2_r1_levers import _card
    from test_val3_trainer import _lists as lists
    d = tmp / "vcards" / "cards"
    d.mkdir(parents=True)
    _card().save(d / "shop.json")
    f = tmp / "lists.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in lists()))
    return tmp / "vcards", f


def _single(tmp, monkeypatch, init, base, *extra):
    cards, lists = _lists(tmp)
    out = tmp / "single"
    monkeypatch.setattr(sys, "argv", ["single_lora.py", "--init", str(init), "--train-file", str(lists),
                                      "--val-file", str(lists), "--cards", str(cards), "--out", str(out),
                                      "--base", str(base), "--epochs", "2", "--tok-budget", "60000", "--q-per-db", "1",
                                      "--dbs-per-step", "1", "--lr", "5e-3", "--head-lr", "5e-2",
                                      "--gen-head-lr", "5e-3", *extra])
    SL.main()
    return out, lists


def _lora(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path / "best_lora" / "adapter_model.safetensors"))


@pytest.fixture
def r1(tmp_path, cpu, monkeypatch):
    base, d, _ = _data(tmp_path)
    init = tmp_path / "r1"
    _run(monkeypatch, base, d, init)
    return base, init


def test_one_adapter_trains_and_loads_both_ways(tmp_path, monkeypatch, r1):
    import eval_sq2
    base, init = r1
    out, lists = _single(tmp_path, monkeypatch, init, base)
    st = json.loads((out / "settings.json").read_text())
    assert st["shared_adapter"] and st["steps_done"] == 6  # 3 lists, 1 per step, 2 epochs
    assert R.shared_adapter(out) and not R.shared_adapter(init)
    head = torch.load(out / "rerank_head.pt")
    assert set(head) == {"score.weight", "score.bias"} and head["score.weight"].abs().sum() > 0
    a, b = _lora(init), _lora(out)
    assert a.keys() == b.keys() and any(not torch.equal(a[k], b[k]) for k in a)
    hist = json.loads((out / "history.json").read_text())
    assert [h["tag"] for h in hist] == ["init (R1)", "epoch 1", "epoch 2"] and "val_pick_acc" in hist[-1]
    # a generator run: the slot model loads from the same dir
    m = eval_sq2.load_model(out, str(base), "", "cpu")
    assert list(m.backbone.model.peft_config) == ["default"]
    # a validator: the same adapter, no second one, scores the lists
    monkeypatch.setattr(R.Reranker, "cuda", lambda self, *x, **k: self, raising=False)
    rr = R.load_validator(out, str(base), "", 16)
    assert rr.adapter == "default" and list(rr.backbone.model.peft_config) == ["default"]
    recs = [json.loads(l) for l in open(lists)]
    R.score_file(rr, recs, R.load_cards([tmp_path / "vcards"]), 9000)
    assert all("rr" in c and torch.isfinite(torch.tensor(c["rr"])) for r in recs for c in R.unique(r["cands"]))


def test_no_replay_leaves_slot_heads(tmp_path, monkeypatch, r1):
    base, init = r1
    out, _ = _single(tmp_path, monkeypatch, init, base, "--gen-weight", "0")
    h0, h1 = torch.load(init / "best_heads.pt"), torch.load(out / "best_heads.pt")
    assert h0.keys() == h1.keys()
    assert all(torch.equal(h0[k], h1[k]) for k in h0)  # only the validator loss: the slot heads get no gradient


def test_shared_reranker_adds_no_adapter():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model

    from luthor.sq.model import Backbone
    from luthor.sq2.rerank import Reranker
    from test_sq2_r1_levers import _tok_dir
    tok = AutoTokenizer.from_pretrained(_tok_dir())
    body = get_peft_model(Qwen2Model(Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                                 num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)),
                          LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
    rr = Reranker(Backbone(body, tok), adapter="default")
    assert list(body.peft_config) == ["default"]
    rr.activate(train=True)
    names = [n for n, p in rr.named_parameters() if p.requires_grad]
    assert names and all(".default." in n or n.startswith("score.") for n in names)
    with pytest.raises(ValueError):
        Reranker(Backbone(body, tok), adapter="other")
    Reranker(Backbone(body, tok))  # the old two-adapter validator still adds "rerank"
    assert set(body.peft_config) == {"default", "rerank"}


def test_shared_validator_on_loaded_slot_model_matches_load_validator(tmp_path, monkeypatch, r1):
    """predict.py's single-LoRA path: the validator built on the already loaded slot model (one backbone) gives the
    same scores as the standalone loader (its own backbone copy)."""
    import eval_sq2
    base, init = r1
    out, lists = _single(tmp_path, monkeypatch, init, base)
    monkeypatch.setattr(R.Reranker, "cuda", lambda self, *x, **k: self, raising=False)
    cards = R.load_cards([tmp_path / "vcards"])
    a = [json.loads(l) for l in open(lists)]
    b = [json.loads(l) for l in open(lists)]
    rr1 = R.load_validator(out, str(base), "", 16).float()
    rr1.eval()
    R.score_file(rr1, a, cards, 9000)
    m = eval_sq2.load_model(out, str(base), "", "cpu").float()
    rr2 = R.shared_validator(m, out)
    assert rr2.backbone is m.backbone and list(m.backbone.model.peft_config) == ["default"]
    rr2.eval()
    R.score_file(rr2, b, cards, 9000)
    for ra, rb in zip(a, b):
        for ca, cb in zip(R.unique(ra["cands"]), R.unique(rb["cands"])):
            assert abs(ca["rr"] - cb["rr"]) < 1e-4
    assert m.backbone.model.training  # score_file leaves train mode on: predict.py calls m.eval() after it


def test_soup_is_the_exact_average_of_two_runs(tmp_path, monkeypatch, r1):
    """scripts/soup_lora.py: the stacked rank-2r LoRA adds exactly the mean of the two runs' weight changes, heads are
    averaged, and the soup loads as a generator run and as a shared-adapter validator."""
    import eval_sq2
    import soup_lora as SP
    from safetensors.torch import load_file
    base, init = r1
    runs = []
    for seed in (0, 1):
        d = tmp_path / f"s{seed}"
        d.mkdir()
        out, _ = _single(d, monkeypatch, init, base, "--seed", str(seed))
        runs.append(out)
    soup = tmp_path / "soup"
    monkeypatch.setattr(sys, "argv", ["soup_lora.py", "--runs", *map(str, runs), "--out", str(soup)])
    SP.main()
    cfg = json.loads((runs[0] / "best_lora" / "adapter_config.json").read_text())
    s_in = cfg["lora_alpha"] / cfg["r"]
    s_out = SP.LOADER_ALPHA / (2 * cfg["r"])
    L = [load_file(str(r / "best_lora" / "adapter_model.safetensors")) for r in runs]
    S = load_file(str(soup / "best_lora" / "adapter_model.safetensors"))
    for k in [k for k in S if ".lora_A." in k]:
        kb = k.replace(".lora_A.", ".lora_B.")
        mean = sum(s_in * l[kb].float() @ l[k].float() for l in L) / 2
        got = s_out * S[kb].float() @ S[k].float()
        assert torch.allclose(got, mean, atol=1e-5, rtol=1e-4), k
    h = [torch.load(r / "rerank_head.pt") for r in runs]
    hs = torch.load(soup / "rerank_head.pt")
    assert torch.allclose(hs["score.weight"], (h[0]["score.weight"] + h[1]["score.weight"]) / 2)
    assert json.loads((soup / "args.json").read_text())["lora_r"] == str(2 * cfg["r"])
    m = eval_sq2.load_model(soup, str(base), "", "cpu")
    assert list(m.backbone.model.peft_config) == ["default"] and m.backbone.model.peft_config["default"].r == 2 * cfg["r"]
    monkeypatch.setattr(R.Reranker, "cuda", lambda self, *x, **k: self, raising=False)
    rr = R.load_validator(soup, str(base), "", 16)
    assert rr.adapter == "default" and R.shared_adapter(soup)


def test_mlp_rerank_head_trains_and_loads(tmp_path, monkeypatch, r1):
    """Option 3: --rerank-head mlp saves a 2-layer head, records it in settings.json, and both loaders rebuild it."""
    import eval_sq2
    base, init = r1
    out, lists = _single(tmp_path, monkeypatch, init, base, "--rerank-head", "mlp")
    assert json.loads((out / "settings.json").read_text())["rerank_head"] == "mlp"
    head = torch.load(out / "rerank_head.pt")
    assert {"score.0.weight", "score.0.bias", "score.2.weight", "score.2.bias"} <= set(head)
    assert head["score.2.weight"].abs().sum() > 0
    monkeypatch.setattr(R.Reranker, "cuda", lambda self, *x, **k: self, raising=False)
    rr = R.load_validator(out, str(base), "", 16)
    assert isinstance(rr.score, torch.nn.Sequential)
    m = eval_sq2.load_model(out, str(base), "", "cpu")
    rr2 = R.shared_validator(m, out)
    assert torch.equal(rr2.score[2].weight, head["score.2.weight"].to(rr2.score[2].weight.dtype))
