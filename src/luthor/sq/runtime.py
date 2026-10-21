"""Inference: one backbone pass per question, cached option vectors per database.

Connecting a database = `index(card)`: every option text and every indexed value is encoded
once (offline) and kept. Answering = encode the question once (token states + pooled),
score against the cache, decode the plan, compile it. On Apple Silicon the backbone runs
in MLX (the LoRA merged into the weights first); the heads are small and run in torch on CPU.

    rt = Runtime.load("runs/luthor_sq/final", backend="mlx")
    rt.index(SchemaCard.load("cards/acme.json"))
    plan, sql = rt.plan("acme", "revenue by country last year")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import Example, question_text, span_token_ranges
from .candidates import find_candidates
from .model import LuthorSQ
from .plan import compile_plan
from .schema_card import OPT_TEMPLATE, SchemaCard, value_text
from .train import decode


class MLXEncoder:
    """The merged backbone in MLX (encoder, plus token states for span pooling)."""

    def __init__(self, ckpt: str, bits: int = 8, group_size: int = 64):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.qwen3_5 import Model, ModelArgs
        from transformers import AutoTokenizer

        self.mx = mx
        cfg = json.loads((Path(ckpt) / "config.json").read_text())
        self.model = Model(ModelArgs.from_dict(cfg))
        weights = mx.load(str(Path(ckpt) / "model.safetensors"))
        weights = self.model.sanitize({k: v for k, v in weights.items() if not k.startswith("score.")})
        self.model.load_weights(list(weights.items()), strict=False)
        if bits:
            nn.quantize(self.model, group_size=group_size, bits=bits)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, texts: list[str], max_len: int = 384, offsets: bool = False):
        mx = self.mx
        enc = [self.tok(t, truncation=True, max_length=max_len, return_offsets_mapping=offsets) for t in texts]
        ids = [e["input_ids"] for e in enc]
        width = max(len(x) for x in ids)
        arr = mx.array([x + [self.tok.pad_token_id] * (width - len(x)) for x in ids])
        H = self.model.language_model.model(arr)  # right padding is safe: causal + recurrent left-to-right
        last = mx.array([len(x) - 1 for x in ids])
        h = H[mx.arange(len(ids)), last]
        mx.eval(H, h)
        H = torch.from_numpy(np.array(H.astype(mx.float32)))
        h = torch.from_numpy(np.array(h.astype(mx.float32)))
        return H, h, ([e["offset_mapping"] for e in enc] if offsets else None)


class TorchEncoder:
    def __init__(self, model: LuthorSQ):
        self.model = model

    @torch.no_grad()
    def __call__(self, texts: list[str], max_len: int = 384, offsets: bool = False):
        enc = self.model.backbone.tokenize(texts, offsets=offsets, max_len=max_len)
        dev = next(self.model.parameters()).device
        H, h = self.model.backbone(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        return H.float().cpu(), h.float().cpu(), (enc["offset_mapping"].tolist() if offsets else None)


class Runtime:
    def __init__(self, heads: LuthorSQ, encoder, bs: int = 64, serving: dict | None = None):
        """`serving`: thresholds calibrated on held-out training databases (serving.json)."""
        self.serving = serving or {}
        self.heads = heads.eval()
        self.encoder = encoder
        self.bs = bs
        self.cache: dict[str, dict] = {}

    @torch.no_grad()
    def _encode_pooled(self, texts: list[str]) -> torch.Tensor:
        out = [self.encoder(texts[i:i + self.bs], max_len=96)[1] for i in range(0, len(texts), self.bs)]
        return torch.cat(out) if out else torch.zeros(0, self.heads.backbone.hidden)

    def index(self, card: SchemaCard) -> float:
        """Pre-compute option and value vectors for a database. Returns seconds taken."""
        t0 = time.perf_counter()
        opts = card.options()
        ids = [o["id"] for o in opts]
        O = self._encode_pooled([OPT_TEMPLATE.format(h=o["text"]) for o in opts])
        # values are encoded lazily, the first time a question's candidates need them, then cached
        self.cache[card.db_id] = {"card": card, "ids": ids, "O": O, "vals": {}, "graph": card.graph()}
        return time.perf_counter() - t0

    @torch.no_grad()
    def plan(self, db_id: str, question: str, context: str = ""):
        c = self.cache[db_id]
        t0 = time.perf_counter()
        text, off = question_text(question, context)
        cands = find_candidates(text[off:], c["card"])
        ex = Example("q", db_id, text, off, cands, None)
        H, h, offsets = self.encoder([text], offsets=True)
        t_enc = time.perf_counter() - t0
        opt_index = {oid: i for i, oid in enumerate(c["ids"])}
        vt = [OPT_TEMPLATE.format(h=value_text(x.col, str(x.value))) for x in cands if x.kind == "value"]
        missing = [t for t in dict.fromkeys(vt) if t not in c["vals"]]
        if missing:
            for t, v in zip(missing, self._encode_pooled(missing)):
                c["vals"][t] = v
        V = torch.stack([c["vals"][t] for t in vt]) if vt else torch.zeros(0, c["O"].shape[1])
        out = self.heads.score(h[0], H[0], c["O"], V, span_token_ranges(ex, offsets[0]),
                               [x.kind == "value" for x in cands],
                               [opt_index.get(x.col, -1) if x.kind == "value" else -1 for x in cands])
        # Luthor always answers: abstaining is off unless a serving config explicitly turns it on
        # (limits belong to the harness). The outcome head's score is returned as a confidence signal.
        thr = self.serving.get("unsupported_threshold") if self.serving.get("abstain", False) else 1.01
        plan = decode(self.heads, out, c["ids"], cands, unsupported_threshold=thr)
        sql = compile_plan(plan, c["graph"]) if plan is not None else None
        p_unsup = float(torch.softmax(out.choice["outcome"], -1)[1])
        return plan, sql, {"encode_ms": t_enc * 1e3, "total_ms": (time.perf_counter() - t0) * 1e3,
                           "p_outside_grammar": round(p_unsup, 3)}
