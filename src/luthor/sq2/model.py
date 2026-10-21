"""LuthorSQ2: schema-in-context slot filling, one forward pass per question.

    [schema text] ──backbone (once per database, cached)──► Hs, cache
    [question] [slot embeddings] ──backbone over the cache──► Hc

    option o     = mean of Hs over o's line         (tables' rows lines and column lines)
    candidate c  = W_m(mean of Hc over its mention in the question)   if mentioned
                 + W_c(mean of Hs where the schema lists the value)    if listed
                 + W_x(mean of Hc over its entry in "Values: ...")     value-side input (after the question)
    slot s       = Hc at s's position (one learned input embedding per slot)

    pointer      s_r(s, o) = exp(τ_r) · cos(W_q s, W_k o) + b_r     one (τ, b) per slot
    choice       softmax(W_ch s_plan)          per block: counts, distinct, order, limit, ...
    pair heads   MLP[W_q s, W_k o, ⊙]         aggregation, operator, date part, OR group, subquery
    value        MLP[W_v s, W_k o, C, W_k o ⊙ C, same column, mentioned] over candidates
    limit        the value MLP with a learned stand-in column

The same backbone as v1 (OpenJev Qwen3.5 NLI + LoRA). The schema prefix is shared across the
questions of a database with prefix_cache.expand_cache, so training pays for it once per
database per step, and serving once per database.
"""
from __future__ import annotations

import collections
import math
import os
import random
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sq.candidates import Candidate
from ..sq.model import Backbone
from .prefix_cache import allow_cache_under_checkpointing, checkpointing, expand_cache, prefix_cache, question_cache, release_cache
from .serialize import SLOTS, VALUE_ENTRIES, SchemaText, char_to_token_ranges, question_text
from .targets import CHOICES, MAIN_CHOICES, PAIRS

SLOT_IX = {s: i for i, s in enumerate(SLOTS)}


def _mlp(d_in: int, n_out: int, hidden: int = 256) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


@dataclass
class SchemaEnc:
    O: torch.Tensor                      # (n_opts, d) option states
    V: torch.Tensor                      # (n_listed_values, d) states of listed values
    vix: dict[tuple[str, str], int]      # (column id, value) -> row of V
    cache: object
    L: int


@dataclass
class Prepared:
    """Tokenised continuation of one question (slot positions follow `ids`)."""
    ids: list[int]
    ment_tok: list[tuple[int, int] | None]  # each candidate's mention in the question, if any
    text: str = ""                          # question text as tokenised (for column-name mentions)
    offsets: list | None = None
    val_tok: list | None = None  # value-side: each candidate's entry in the Values section (None: not written)


@dataclass
class Encoded:
    """One question's states against its database's options and literal candidates."""
    S: torch.Tensor           # (n_slots, d)
    O: torch.Tensor           # (n_opts, d)
    M: torch.Tensor           # (n_cands, d) mention states (zeros when not mentioned)
    Vc: torch.Tensor          # (n_cands, d) listed-value states (zeros when not listed)
    mentioned: torch.Tensor   # (n_cands,)
    listed: torch.Tensor      # (n_cands,)
    cand_col: torch.Tensor    # (n_cands,) option index of a stored value's column, -1 otherwise
    OM: torch.Tensor | None = None  # (n_opts, d) question states where an option's name is written (zeros if not)
    om: torch.Tensor | None = None  # (n_opts,) 1 where the option's name appears in the question or context
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    cv: torch.Tensor | None = None
    X: torch.Tensor | None = None   # (n_cands, d) value-side: states over each candidate's Values entry


class LuthorSQ2(nn.Module):
    def __init__(self, backbone: Backbone, d_proj: int = 512, joint_layers: int = 0, col_mentions: bool = False,
                 mention_weight: str = "flat", mention_drop: float = 0.0, value_side: bool = False,
                 value_entries: int = VALUE_ENTRIES):
        super().__init__()
        self.backbone = backbone
        d = backbone.hidden
        # --grad-ckpt: checkpointed layers keep the schema prefix cache (prefix_cache.py); used only while training
        # (an unsupported backbone, e.g. Qwen3.5's linear attention, fails only when a checkpointed training pass runs)
        self.ckpt_cache, self._ckpt_err = checkpointing(backbone.model), None
        if self.ckpt_cache:
            try:
                allow_cache_under_checkpointing(backbone.model)
            except NotImplementedError as e:
                self._ckpt_err = e
        self.d_proj = d_proj
        emb = self.body.get_input_embeddings().weight
        init = emb[torch.randint(0, emb.shape[0], (len(SLOTS),))].detach().float().clone()
        self.slot_emb = nn.Parameter(init)   # one input embedding per slot position
        self.W_q = nn.Linear(d, d_proj)
        self.W_k = nn.Linear(d, d_proj)
        self.W_v = nn.Linear(d, d_proj)
        self.W_c = nn.Linear(d, d_proj)     # candidate: where the schema lists it
        self.W_m = nn.Linear(d, d_proj)     # candidate: its mention in the question
        self.log_scale = nn.Parameter(torch.full((len(SLOTS),), math.log(20.0)))
        self.bias = nn.Parameter(torch.full((len(SLOTS),), -2.0))
        heads = {}
        for b in ("m", "s", "o"):
            for k, v in CHOICES.items():
                heads[f"{b}.{k}"] = len(v)
        for k, v in MAIN_CHOICES.items():
            heads[f"m.{k}"] = len(v)
        self.choice = nn.ModuleDict({k.replace(".", "_"): nn.Linear(d, n) for k, n in heads.items()})
        self.pair = nn.ModuleDict({h: _mlp(3 * d_proj, len(v)) for h, v in PAIRS.items()})
        self.value_head = _mlp(4 * d_proj + 2, 1)
        self.value_role = nn.Parameter(torch.zeros(4, d_proj))  # 0 = the filter's own values, 1..3 = template {V0}..{V2}
        # small random, not zeros: it is L2-normalised, and normalising a zero vector has a ~1/eps gradient (one
        # wasted, clipped step at the first LIMIT target of a fresh run). Trained checkpoints load over it.
        self.limit_col = nn.Parameter(torch.randn(d_proj) * 0.02)
        # Optional joint head (idea from Cloudflare Clef's "joint schema head"): a small transformer over all
        # slot, option and candidate states together, so slots can coordinate before they are scored. Its
        # outputs are added through zero-initialised projections: with joint_layers > 0 the model starts out
        # exactly like the plain heads. Kept only if a paired dev comparison shows it raises accuracy.
        self.joint_layers = joint_layers
        # Column mentions: the schema is encoded before the question, so option states never see it. Where a
        # column's (or table's) name is written in the question or BIRD evidence ("A3 refers to district names"),
        # the question's own state there is added to that option's key, plus a per-slot bonus. Zero-initialised.
        self.col_mentions = col_mentions
        self.mention_weight = mention_weight  # "inv": a name matching N options counts 1/N (raceId in 9 tables)
        self.mention_drop = mention_drop  # training only: hide this share of written names, so linking does not
        # lean on exact string matches (Dr.Spider: such features break when a question rewords the column)
        if col_mentions:
            self.W_om = nn.Linear(d, d_proj)
            nn.init.zeros_(self.W_om.weight)
            nn.init.zeros_(self.W_om.bias)
            self.om_bias = nn.Parameter(torch.zeros(len(SLOTS)))
        # Value-side input: the selected candidates are written after the question ("Values: ..."), so their states
        # there have read the question and evidence; W_x of those states is added to each candidate's vector (a sum
        # keeps the mention / listed terms and needs no new head input). Off: no W_x, outputs as before.
        self.value_side, self.value_entries = value_side, value_entries  # entries: the most relevant N candidates
        if value_side:
            self.W_x = nn.Linear(d, d_proj)
        if joint_layers:
            self.j_in_s, self.j_in_o = nn.Linear(d, d_proj), nn.Linear(d, d_proj)
            self.j_type = nn.Parameter(torch.zeros(3, d_proj))
            layer = nn.TransformerEncoderLayer(d_proj, 8, 4 * d_proj, dropout=0.0, batch_first=True, norm_first=True)
            self.joint = nn.TransformerEncoder(layer, joint_layers)
            self.j_out_s, self.j_out_o, self.j_out_c = nn.Linear(d_proj, d), nn.Linear(d_proj, d_proj), nn.Linear(d_proj, d_proj)
            for lin in (self.j_out_s, self.j_out_o, self.j_out_c):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    @property
    def body(self):
        return self.backbone.model

    @property
    def device(self):
        return self.W_q.weight.device

    def _ckpt_now(self) -> bool:
        on = self.ckpt_cache and self.training and torch.is_grad_enabled()
        if on and self._ckpt_err is not None:
            raise self._ckpt_err
        return on

    def _text_config(self):
        cfg = self.body.config
        return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg

    # ---- tokenisation ----------------------------------------------------------------
    def prepare_schema(self, st: SchemaText):
        enc = self.backbone.tok(st.text, return_offsets_mapping=True, add_special_tokens=False)
        keys = list(st.value_spans)
        vt = char_to_token_ranges(enc["offset_mapping"], [st.value_spans[k] for k in keys])
        return enc["input_ids"], char_to_token_ranges(enc["offset_mapping"], st.spans), dict(zip(keys, vt))

    def prepare_question(self, question: str, cands: list[Candidate], context: str = "",
                         names: dict[str, str] | None = None) -> Prepared:
        """`names` (value-side, renamed schema): identifiers as the schema shows them, for the Values section."""
        qt = question_text(question, context, cands if self.value_side else None, names, self.value_entries)
        enc = self.backbone.tok(qt.text, return_offsets_mapping=True, add_special_tokens=False)
        ment = [(c.span[0] + qt.q_offset, c.span[1] + qt.q_offset) if c.span else None for c in cands]
        rng = char_to_token_ranges(enc["offset_mapping"], ment)
        if not self.value_side:
            return Prepared(enc["input_ids"], [r if m else None for r, m in zip(rng, ment)], qt.text, enc["offset_mapping"])
        # column-name mentions read only the question part (a column written in the Values section is not a mention)
        return Prepared(enc["input_ids"], [r if m else None for r, m in zip(rng, ment)], qt.text[:qt.q_end],
                        enc["offset_mapping"], [r if sp else None for r, sp in
                                                zip(char_to_token_ranges(enc["offset_mapping"], qt.v_spans), qt.v_spans)])

    # ---- encoding ----------------------------------------------------------------------
    @staticmethod
    def _span_means(H: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
        if not ranges:
            return H.new_zeros(0, H.shape[-1], dtype=torch.float32)
        cs = torch.cat([H.new_zeros(1, H.shape[-1], dtype=torch.float32), H.float().cumsum(0)])
        s = torch.tensor([a for a, _ in ranges], device=H.device)
        e = torch.tensor([max(b, a + 1) for a, b in ranges], device=H.device)
        return (cs[e] - cs[s]) / (e - s)[:, None].float()

    def encode_schema(self, prepared) -> SchemaEnc:
        ids, opt_tok, val_tok = prepared
        x = torch.tensor([ids], device=self.device)
        if self._ckpt_now():  # under checkpointing HF does not return the cache it filled: keep our own reference
            cache = prefix_cache(self._text_config())
            out = self.body(input_ids=x, use_cache=True, past_key_values=cache)
        else:
            out = self.body(input_ids=x, use_cache=True)
            cache = out.past_key_values
        Hs = out.last_hidden_state[0]
        keys = list(val_tok)
        return SchemaEnc(self._span_means(Hs, opt_tok), self._span_means(Hs, [val_tok[k] for k in keys]),
                         {k: i for i, k in enumerate(keys)}, cache, len(ids))

    def encode_questions(self, se: SchemaEnc, preps: list[Prepared], cands: list[list[Candidate]],
                         oix: dict[str, int], mix: dict[str, int] | None = None) -> list[Encoded]:
        """Continue the schema cache with N questions at once. Each row is [question tokens,
        slot embeddings, right padding]: attention is causal, so no state that is read out
        ever sees padding, and a batch gives exactly what one question alone would."""
        n = len(preps)
        ns = len(SLOTS)
        T = max(len(p.ids) for p in preps) + ns
        pad = self.backbone.tok.pad_token_id
        x = torch.full((n, T), pad, dtype=torch.long, device=self.device)
        is_slot = torch.zeros(n, T, dtype=torch.bool, device=self.device)
        slot_idx = torch.zeros(n, T, dtype=torch.long, device=self.device)
        for i, p in enumerate(preps):
            q = len(p.ids)
            x[i, :q] = torch.tensor(p.ids, device=self.device)
            is_slot[i, q:q + ns] = True
            slot_idx[i, q:q + ns] = torch.arange(ns, device=self.device)
        emb = self.body.get_input_embeddings()(x)
        inp = torch.where(is_slot[..., None], self.slot_emb.to(emb.dtype)[slot_idx], emb)
        cache = question_cache(se.cache, n, self._text_config()) if self._ckpt_now() else expand_cache(se.cache, n)
        pos = torch.arange(se.L, se.L + T, device=self.device)[None].expand(n, -1)
        Hc = self.body(inputs_embeds=inp, past_key_values=cache, position_ids=pos, use_cache=True).last_hidden_state
        release_cache(cache)
        se.cache = None
        outs = []
        d = Hc.shape[-1]
        for i, (p, cs) in enumerate(zip(preps, cands)):
            H = Hc[i]
            q = len(p.ids)
            S = H[q:q + ns].float()
            M = self._span_means(H, [r or (0, 1) for r in p.ment_tok]) if cs else H.new_zeros(0, d, dtype=torch.float32)
            men = torch.tensor([r is not None for r in p.ment_tok], device=self.device, dtype=torch.float32)
            vi = [se.vix.get((c.col, str(c.value)), -1) if c.kind == "value" else -1 for c in cs]
            lst = torch.tensor([v >= 0 for v in vi], device=self.device, dtype=torch.float32)
            if cs and se.V.shape[0]:
                Vc = se.V[torch.tensor([max(v, 0) for v in vi], device=self.device, dtype=torch.long)]
            else:
                Vc = H.new_zeros(len(cs), d, dtype=torch.float32)
            cc = torch.tensor([oix.get(c.col, -1) if c.kind == "value" and c.col else -1 for c in cs],
                              device=self.device, dtype=torch.long)
            if cs:
                M, Vc = M * men[:, None], Vc * lst[:, None]
            OM = om = None
            if self.col_mentions:
                hits = option_mentions(p.text, mix if mix is not None else oix)
                if self.training and self.mention_drop > 0 and hits:
                    gone = {sp for _, sp in hits if random.random() < self.mention_drop}
                    hits = [(o, sp) for o, sp in hits if sp not in gone]
                OM = H.new_zeros(se.O.shape[0], d, dtype=torch.float32)
                om = torch.zeros(se.O.shape[0], device=self.device)
                if hits:
                    rngs = char_to_token_ranges(p.offsets, [sp for _, sp in hits])
                    means = self._span_means(H, [(a, b) for a, b in rngs])
                    idx = torch.tensor([o for o, _ in hits], device=self.device)
                    if self.mention_weight == "inv":  # a name written once but matching N options counts 1/N for each
                        per = collections.Counter(sp for _, sp in hits)
                        w = torch.tensor([1.0 / per[sp] for _, sp in hits], device=self.device)
                    else:
                        w = torch.ones(len(hits), device=self.device)
                    OM = OM.index_add(0, idx, means * w[:, None])
                    cnt = torch.zeros(se.O.shape[0], device=self.device).index_add(0, idx, w)
                    om = torch.zeros(se.O.shape[0], device=self.device).index_reduce(0, idx, w, "amax", include_self=True)
                    OM = OM / cnt.clamp(min=1e-6)[:, None] * (cnt > 0).float()[:, None]
            X = None
            if self.value_side:
                X = H.new_zeros(0, d, dtype=torch.float32)
                if cs:  # candidates past the entry budget keep only their mention / listed terms (X = 0)
                    shown = torch.tensor([r is not None for r in p.val_tok], device=self.device, dtype=torch.float32)
                    X = self._span_means(H, [r or (0, 1) for r in p.val_tok]) * shown[:, None]
            outs.append(Encoded(S, se.O, M, Vc, men, lst, cc, OM, om, X=X))
        return outs

    # ---- heads ------------------------------------------------------------------------
    def project(self, e: Encoded) -> Encoded:
        O = e.O.float()
        cv = (self.W_m(e.M) * e.mentioned[:, None] + self.W_c(e.Vc) * e.listed[:, None]) if e.M.shape[0] \
            else e.M.new_zeros(0, self.d_proj)
        if self.value_side and e.X is not None and e.X.shape[0]:
            cv = cv + self.W_x(e.X)
        dk = dc = 0.0
        if self.joint_layers:
            ns, no = e.S.shape[0], O.shape[0]
            x = torch.cat([self.j_in_s(e.S) + self.j_type[0], self.j_in_o(O) + self.j_type[1], cv + self.j_type[2]])
            h = self.joint(x[None])[0]
            e.S = e.S + self.j_out_s(h[:ns])   # choice, pointer and value heads all read the updated slots
            dk, dc = self.j_out_o(h[ns:ns + no]), self.j_out_c(h[ns + no:])
        e.q = F.normalize(self.W_q(e.S), dim=-1)
        if self.col_mentions and e.OM is not None:
            dk = dk + self.W_om(e.OM) * e.om[:, None]
        e.k = F.normalize(self.W_k(O) + dk, dim=-1)
        e.cv = F.normalize(cv + dc, dim=-1) if e.M.shape[0] else cv
        return e

    def ptr_scores(self, e: Encoded, slot: str) -> torch.Tensor:
        i = SLOT_IX[slot]
        s = self.log_scale[i].exp() * (e.k @ e.q[i]) + self.bias[i]
        return s + self.om_bias[i] * e.om if self.col_mentions and e.om is not None else s

    def choice_logits(self, e: Encoded, key: str) -> torch.Tensor:
        b = key.split(".", 1)[0]
        return self.choice[key.replace(".", "_")](e.S[SLOT_IX[f"{b}.plan"]])

    def pair_logits(self, e: Encoded, slot: str, opt, head: str) -> torch.Tensor:
        q = e.q[SLOT_IX[slot]]
        k = e.k[opt]
        qq = q.expand_as(k)
        return self.pair[head](torch.cat([qq, k, qq * k], -1))

    def value_logits(self, e: Encoded, slot: str, opt: int | None) -> torch.Tensor:
        if e.cv.shape[0] == 0:
            return e.cv.new_zeros(0)
        slot, _, role = slot.partition("|t")
        qv = self.W_v(e.S[SLOT_IX[slot]]) + self.value_role[int(role) + 1 if role else 0]
        qv = F.normalize(qv, dim=-1).expand_as(e.cv)
        col = (e.k[opt] if opt is not None else F.normalize(self.limit_col, dim=-1)).expand_as(e.cv)
        same = ((e.cand_col == opt).float() if opt is not None else torch.zeros_like(e.mentioned))[:, None]
        x = torch.cat([qv, col, e.cv, col * e.cv, same, e.mentioned[:, None]], -1)
        return self.value_head(x).squeeze(-1)

    def head_state(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}


class TorchScores:
    """targets.Scores over one encoded question (for decode)."""

    def __init__(self, model: LuthorSQ2, e: Encoded, option_ids: list[str], cands: list[Candidate]):
        self.m, self.e, self.option_ids, self.cands = model, e, option_ids, cands

    def choice(self, key):
        return self.m.choice_logits(self.e, key).tolist()

    def ptr(self, slot):
        return self.m.ptr_scores(self.e, slot).tolist()

    def pair(self, slot, opt, head):
        return self.m.pair_logits(self.e, slot, opt, head).tolist()

    def vals(self, slot, opt):
        return self.m.value_logits(self.e, slot, opt).tolist()

    def limit(self, slot):
        return self.m.value_logits(self.e, slot, None).tolist()


_MENTION_RX: dict = {}


def _name_forms(name: str) -> set[str]:
    """Ways a column or table name is written in a question: as is, with spaces, camelCase split."""
    n = name.strip('"`[]')
    forms = {n, n.replace("_", " "), re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n).replace("_", " ")}
    return {f.lower() for f in forms if len(f) >= 2}


def option_mentions(text: str, oix: dict[str, int]) -> list[tuple[int, tuple[int, int]]]:
    """(option index, char span) for every place an option's name appears in the question text."""
    key = hash(tuple(oix.items()))  # per database (an id() could be reused by a later dict)
    if key not in _MENTION_RX:
        if len(_MENTION_RX) > 256:
            _MENTION_RX.clear()
        pats = []
        for oid, i in oix.items():
            t, _, c = oid.partition(".")
            forms = _name_forms(t if c in ("", "*") else c)
            if forms:
                alt = "|".join(sorted(map(re.escape, forms), key=len, reverse=True))
                pats.append((i, re.compile(rf"(?<![\w])(?:{alt})(?![\w])", re.I)))
        _MENTION_RX[key] = pats
    lo, hi = text.find("Question: "), text.rfind("\nPlan:")  # only the question and its context
    lo, hi = (lo + len("Question: ") if lo >= 0 else 0), (hi if hi >= 0 else len(text))
    out = []
    for i, rx in _MENTION_RX[key]:
        out += [(i, m.span()) for m in rx.finditer(text, lo, hi)]
    if os.environ.get("LUTHOR_COLMENT_OFF"):  # diagnostics: inference without mentions
        return []
    cap = int(os.environ.get("LUTHOR_COLMENT_MAXHIT", "0"))
    if cap:  # diagnostics: drop a written name that matches more than `cap` options (e.g. raceId in 9 tables)
        per = collections.Counter(sp for _, sp in out)
        out = [(i, sp) for i, sp in out if per[sp] <= cap]
    return out
