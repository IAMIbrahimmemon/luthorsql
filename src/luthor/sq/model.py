"""LuthorSQ: a dynamic bi-encoder that fills a bounded query plan in one forward pass.

An earlier closed-catalog planner learned one embedding per metric, so its options lived in the
weights. Here NOTHING is per-option: every table / column / value is encoded from its description by the
same backbone, so a database never seen in training is just a new set of option texts.

    question ──backbone──► H (token states), h (last-token pooled)
    options  ──backbone──► O (pooled), cached per database at inference

    column slots   s_r(q, o) = exp(τ_r) · cos(W_r h, W_o o) + b_r        r ∈ ROLES
    choice heads   softmax(W_c h)            outcome, #proj, compose, distinct, #group,
                                              #filter, #cond, grain, order_dir, limit_kind
    pair heads     MLP[W_r h, W_o o, W_r h ⊙ W_o o]   aggregation, operator, date part
    value pointer  MLP[W_v h, W_o o_f, c, W_o o_f ⊙ c, same_col] over literal candidates c:
                   stored values (encoded like options) or question spans (mean of H)

Backbone: an OpenJev checkpoint (Qwen3.5-4B NLI, MIT) with LoRA, used as an encoder
(last-token pooling). Only public data trains it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plan import AGGS, COMPOSE, GRAINS, LIMIT_KINDS, MAX_FILTERS, MAX_GROUP, MAX_PROJ, OPS, ORDER_DIRS, PARTS

_LOG = logging.getLogger("luthorsql")

# positional single-choice slots (proj0..2, cond0..1, order, time) and set slots (group, filter)
ROLES = ("proj0", "proj1", "proj2", "group", "filter", "order", "time", "cond0", "cond1", "via", "join")
SINGLE = ("proj0", "proj1", "proj2", "order", "time", "cond0", "cond1")
# via: bridge tables the join must pass through (scored on tables' rows options);
# join: columns whose FK edges the join should use. Both are hints to the compiler, which
# still builds every join itself.
SET = ("group", "filter", "via", "join")
CHOICES = {
    "outcome": ("plan", "unsupported"),
    "n_proj": tuple(str(i) for i in range(1, MAX_PROJ + 1)),
    "compose": COMPOSE,
    "distinct": ("no", "yes"),
    "n_group": tuple(str(i) for i in range(MAX_GROUP + 1)),
    "n_filter": tuple(str(i) for i in range(MAX_FILTERS + 1)),
    "cond_mask": ("none", "p0", "p1", "both"),  # which projections are conditional aggregates
    "grain": GRAINS,
    "order_dir": ORDER_DIRS,
    "limit_kind": LIMIT_KINDS,
}


class Backbone(nn.Module):
    """Any HF decoder used as an encoder: last non-pad token pooled, plus all token states."""

    def __init__(self, model: nn.Module, tokenizer, max_len: int = 384):
        super().__init__()
        self.model = model
        self.tok = tokenizer
        self.max_len = max_len
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.hidden = cfg.hidden_size

    @classmethod
    def from_pretrained(cls, path: str, subfolder: str | None = None, lora_r: int = 32, lora_alpha: int = 64,
                        dtype=torch.bfloat16, grad_ckpt: bool = True) -> "Backbone":
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        kw = {"subfolder": subfolder} if subfolder else {}
        tok = AutoTokenizer.from_pretrained(path, **kw)
        try:  # OpenJev ships as a sequence classifier; keep only its transformer body
            full = AutoModelForSequenceClassification.from_pretrained(path, dtype=dtype, **kw)
            body = getattr(full, full.base_model_prefix)
            body = getattr(body, "language_model", body)
        except Exception:
            _LOG.debug("%s: no sequence-classification wrapper, loading the base model", path, exc_info=True)
            body = AutoModel.from_pretrained(path, dtype=dtype, **kw)
        for n, p in body.named_parameters():
            if "visual" in n:
                p.requires_grad = False
        if grad_ckpt:
            body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora_r:
            from peft import LoraConfig, get_peft_model

            body = get_peft_model(body, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                                                   target_modules="all-linear"))
        return cls(body, tok)

    def tokenize(self, texts: list[str], offsets: bool = False, max_len: int | None = None) -> dict:
        return self.tok(texts, truncation=True, max_length=max_len or self.max_len, padding=True,
                        return_tensors="pt", return_offsets_mapping=offsets)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        H = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1
        return H, H[torch.arange(H.shape[0], device=H.device), last]


def _pair_mlp(d: int, n_out: int, hidden: int = 256, extra: int = 0) -> nn.Module:
    return nn.Sequential(nn.Linear(3 * d + extra, hidden), nn.GELU(), nn.Linear(hidden, n_out))


@dataclass
class ExampleOut:
    """Scores for one question against its database's options and literal candidates."""
    choice: dict[str, torch.Tensor]    # head -> (n_classes,)
    slots: dict[str, torch.Tensor]     # role -> (n_opts,)
    agg: dict[str, torch.Tensor]       # proj0..2 / order -> (n_opts, |AGGS|)
    op: dict[str, torch.Tensor]        # filter / cond0 / cond1 -> (n_opts, |OPS|)
    part: dict[str, torch.Tensor]      # filter / cond0 / cond1 -> (n_opts, |PARTS|)
    qv: torch.Tensor                   # (d_proj,) value-pointer query
    ql: torch.Tensor                   # (d_proj,) LIMIT-literal query
    ov: torch.Tensor                   # (n_opts, d_proj) projected options
    cands: torch.Tensor                # (n_cands, d_proj) projected literal candidates
    cand_col: torch.Tensor             # (n_cands,) option index of a stored value's column, -1 for spans
    mentioned: torch.Tensor            # (n_cands,) 1 if the candidate occurs in the question text


class LuthorSQ(nn.Module):
    def __init__(self, backbone: Backbone, d_proj: int = 512):
        super().__init__()
        self.backbone = backbone
        d = backbone.hidden
        self.d_proj = d_proj
        self.q_proj = nn.ModuleDict({r: nn.Linear(d, d_proj) for r in ROLES + ("value", "limit")})
        self.o_proj = nn.Linear(d, d_proj)   # tables' rows options and columns
        self.v_proj = nn.Linear(d, d_proj)   # stored-value candidates
        self.s_proj = nn.Linear(d, d_proj)   # question-span candidates
        self.log_scale = nn.ParameterDict({r: nn.Parameter(torch.tensor(math.log(20.0))) for r in ROLES})
        self.bias = nn.ParameterDict({r: nn.Parameter(torch.tensor(-2.0)) for r in ROLES})
        self.choice = nn.ModuleDict({k: nn.Linear(d, len(v)) for k, v in CHOICES.items()})
        self.agg_head = _pair_mlp(d_proj, len(AGGS))
        self.order_agg_head = _pair_mlp(d_proj, len(AGGS))  # proj0..2 share agg_head
        self.op_head = _pair_mlp(d_proj, len(OPS))
        self.part_head = _pair_mlp(d_proj, len(PARTS))
        self.value_head = _pair_mlp(d_proj, 1, extra=d_proj + 2)  # + same-column bit + mentioned bit
        self.limit_col = nn.Parameter(torch.zeros(d_proj))  # stands in for "the column" of a LIMIT literal

    # ---- encoding ---------------------------------------------------------------------
    opt_max_len = 64  # option / value texts are short; capping them bounds the per-step cost

    def encode_texts(self, texts: list[str], bs: int = 256, max_len: int | None = None) -> torch.Tensor:
        """Pooled vectors for option / value texts (with grad when training). Texts are sorted
        by length so each chunk pads little; results come back in the input order."""
        max_len = max_len or self.opt_max_len
        dev = self.o_proj.weight.device
        if not texts:
            return torch.zeros(0, self.backbone.hidden, device=dev)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        outs = []
        for i in range(0, len(order), bs):
            chunk = [texts[j] for j in order[i:i + bs]]
            enc = self.backbone.tokenize(chunk, max_len=max_len)
            _, pooled = self.backbone(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
            outs.append(pooled.float())
        sorted_vecs = torch.cat(outs)
        inv = torch.empty(len(order), dtype=torch.long, device=dev)
        inv[torch.tensor(order, device=dev)] = torch.arange(len(order), device=dev)
        return sorted_vecs[inv]

    def encode_questions(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, list]:
        dev = next(self.parameters()).device
        enc = self.backbone.tokenize(texts, offsets=True)
        H, h = self.backbone(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        return H.float(), h.float(), enc["offset_mapping"].tolist()

    # ---- scoring ----------------------------------------------------------------------
    @staticmethod
    def _pair(q: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        q = q.expand_as(o)
        return torch.cat([q, o, q * o], -1)

    def score(self, h: torch.Tensor, H: torch.Tensor, O: torch.Tensor, value_vecs: torch.Tensor,
              span_tok: list[tuple[int, int] | None], cand_kind_value: list[bool], cand_col: list[int]) -> ExampleOut:
        """One question: h (d,), H (T, d); O (n_opts, d) option vectors of its database;
        value_vecs (n_value_cands, d) in candidate order; span_tok[i]: token range where
        candidate i occurs in the question (None for an enumerated stored value).
        A stored value = its encoding + (if mentioned) the question's states over the mention;
        a span = the question's states over it."""
        ov = F.normalize(self.o_proj(O), dim=-1)
        slots = {}
        qn = {}
        for r in ROLES:
            qn[r] = F.normalize(self.q_proj[r](h), dim=-1)
            slots[r] = self.log_scale[r].exp() * (ov @ qn[r]) + self.bias[r]
        choice = {k: head(h) for k, head in self.choice.items()}
        # candidates, vectorised: span vectors = mean of H over each span (one cumsum), stored
        # values = W_v(value vector) (+ its mention's span vector when mentioned)
        n = len(cand_kind_value)
        dev = h.device
        if n:
            spans = [sp if sp is not None else (0, 0) for sp in span_tok]
            s_idx = torch.tensor([a for a, _ in spans], device=dev)
            e_idx = torch.tensor([max(b, a + 1) if sp is not None else 0 for (a, b), sp in zip(spans, span_tok)],
                                 device=dev)
            men = torch.tensor([sp is not None for sp in span_tok], device=dev)
            cs = torch.cat([H.new_zeros(1, H.shape[-1]), H.cumsum(0)])
            span_mean = (cs[e_idx] - cs[s_idx]) / (e_idx - s_idx).clamp(min=1)[:, None].float()
            span_vec = self.s_proj(span_mean) * men[:, None].float()
            is_val = torch.tensor(cand_kind_value, device=dev)
            vv = H.new_zeros(n, self.d_proj)
            if value_vecs.shape[0]:
                vv = vv.index_put((is_val.nonzero().squeeze(1),), self.v_proj(value_vecs))
            C = F.normalize(vv + span_vec, dim=-1)
            mentioned = men.float()
        else:
            C = ov.new_zeros(0, self.d_proj)
            mentioned = ov.new_zeros(0)
        agg = {r: self.agg_head(self._pair(qn[r], ov)) for r in ("proj0", "proj1", "proj2")}
        agg["order"] = self.order_agg_head(self._pair(qn["order"], ov))
        return ExampleOut(
            choice=choice, slots=slots, agg=agg,
            op={r: self.op_head(self._pair(qn[r], ov)) for r in ("filter", "cond0", "cond1")},
            part={r: self.part_head(self._pair(qn[r], ov)) for r in ("filter", "cond0", "cond1")},
            qv=F.normalize(self.q_proj["value"](h), dim=-1),
            ql=F.normalize(self.q_proj["limit"](h), dim=-1), ov=ov, cands=C,
            cand_col=torch.tensor(cand_col, device=h.device, dtype=torch.long),
            mentioned=mentioned,
        )

    def value_scores(self, out: ExampleOut, col_vec: torch.Tensor, col_idx: int,
                     query: torch.Tensor | None = None, cands: torch.Tensor | None = None,
                     cand_col: torch.Tensor | None = None, mentioned: torch.Tensor | None = None) -> torch.Tensor:
        """Score literal candidates as the value of the filter on option `col_idx`.
        `cands` / `cand_col` default to the example's own; the minimal-pair loss passes a
        twin's candidates to cross-score them."""
        C = out.cands if cands is None else cands
        cc = out.cand_col if cand_col is None else cand_col
        if C.shape[0] == 0:
            return C.new_zeros(0)
        men = (out.mentioned if mentioned is None else mentioned)[:, None]
        same = (cc == col_idx).float()[:, None]
        q = (out.qv if query is None else query).expand_as(C)
        c = col_vec.expand_as(C)
        return self.value_head(torch.cat([q, c, C, c * C, same, men], -1)).squeeze(-1)

    def limit_scores(self, out: ExampleOut) -> torch.Tensor:
        if out.cands.shape[0] == 0:
            return out.cands.new_zeros(0)
        return self.value_scores(out, F.normalize(self.limit_col, dim=-1), -2, query=out.ql)

    # ---- persistence ------------------------------------------------------------------
    def head_state(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}
