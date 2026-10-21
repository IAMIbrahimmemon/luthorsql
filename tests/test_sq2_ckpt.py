"""--grad-ckpt with the schema prefix cache (prefix_cache.prefix_cache / question_cache), on a tiny randomly
initialised Qwen2 + LoRA (OmniSQL's architecture): same loss and gradients as the plain run, far fewer saved
activations, and inference unchanged. CPU, a few seconds."""
import random

import pytest
import torch

import luthor.sq2.train as tr
from luthor.sq.model import Backbone
from luthor.sq2.model import LuthorSQ2
from test_sq2_r1_levers import _card, _rows, _tok_dir


def _pair():
    """Two models with identical weights: plain, and with gradient checkpointing (as Backbone.from_pretrained
    enables it: on the base model, before LoRA)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
    tok = AutoTokenizer.from_pretrained(_tok_dir())
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2)
    out = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        body = Qwen2Model(cfg)
        if ckpt:
            body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        body = get_peft_model(body, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear"))
        for n, p in body.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.02, generator=torch.Generator().manual_seed(hash(n) % 2**31))
        torch.manual_seed(1)
        out.append(LuthorSQ2(Backbone(body, tok), d_proj=32, col_mentions=True, value_side=True))
    out[1].load_state_dict(out[0].state_dict())
    assert not out[0].ckpt_cache and out[1].ckpt_cache
    return out


def _loss_and_grads(m, exs, card):
    m.train()
    m.zero_grad(set_to_none=True)
    saved = [0]

    def pack(t):
        saved[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        st, oix, enc = tr.encode_db(m, card, exs)
        loss = sum(tr.example_loss(m, x, e.t, oix)[0] for e, x in zip(exs, enc)) / len(exs)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    return float(loss), grads, saved[0]


def test_grad_ckpt_matches_plain_run():
    card = _card()
    exs = [tr.make_example(r, card, random.Random(0), order="v2") for r in _rows()]
    plain, ckpt = _pair()
    l0, g0, s0 = _loss_and_grads(plain, exs, card)
    l1, g1, s1 = _loss_and_grads(ckpt, exs, card)
    assert l1 == pytest.approx(l0, rel=1e-5)
    assert g0.keys() == g1.keys() and any("lora_" in n for n in g0)  # LoRA and the schema prefix both train
    for n in g0:
        assert torch.allclose(g0[n], g1[n], rtol=1e-4, atol=1e-6), n
    assert s1 < 0.5 * s0, (s0, s1)  # activations saved for backward: the reason --grad-ckpt exists


def test_grad_ckpt_twice_is_stable():
    """A second step reuses nothing from the first (the overwrite cache never accumulates states)."""
    card = _card()
    exs = [tr.make_example(r, card, random.Random(0), order="v2") for r in _rows()]
    _, ckpt = _pair()
    a = _loss_and_grads(ckpt, exs, card)
    b = _loss_and_grads(ckpt, exs, card)
    assert a[0] == pytest.approx(b[0], rel=1e-6)
    for n in a[1]:
        assert torch.allclose(a[1][n], b[1][n], rtol=1e-5, atol=1e-7), n


def test_grad_ckpt_inference_unchanged():
    card = _card()
    exs = [tr.make_example(r, card, random.Random(0), order="v2") for r in _rows()]
    plain, ckpt = _pair()
    for m in (plain, ckpt):
        m.eval()
    with torch.no_grad():
        a = tr.encode_db(plain, card, exs)[2]
        b = tr.encode_db(ckpt, card, exs)[2]
    for x, y in zip(a, b):
        assert torch.allclose(x.S, y.S, atol=1e-6) and torch.allclose(x.k, y.k, atol=1e-6)
    assert tr.predict(plain, card, exs) == tr.predict(ckpt, card, exs)


def test_grad_ckpt_refuses_linear_attention():
    from types import SimpleNamespace

    from luthor.sq2.prefix_cache import allow_cache_under_checkpointing
    body = SimpleNamespace(config=SimpleNamespace(layer_types=["linear_attention", "full_attention"]),
                           modules=lambda: [])
    with pytest.raises(NotImplementedError):
        allow_cache_under_checkpointing(body)
