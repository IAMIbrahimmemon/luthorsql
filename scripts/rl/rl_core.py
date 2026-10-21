# Vendored (imports made absolute; ExecReward since made parallel within a database, + noisy_gold) from .claude/worktrees/wf_cfb20c27-79e-46/src/luthor/sq2/rl.py (fanout R1, never merged).
"""Training objectives beyond gold cross-entropy for LuthorSQ2 (T6 execution-reward RL, T2 teacher distillation).

The slot heads define a factorised policy over plans: every decision `decode` makes (a count, a column, an
operator, a value, ...) is an argmax over one score vector. Two pieces turn that into something RL can use:

  sampling     `SampledScores` adds Gumbel(0, 1) noise to each score vector (scaled by 1/temp) once per draw.
               argmax(s/T + g) is an exact draw from softmax(s/T), and top-k(s/T + g) an exact draw of k
               options without replacement (Plackett-Luce; the Gumbel-top-k trick). So `decode` over these
               scores IS a sample from the heads, with no change to the decoder.
  log-prob     `plan_logprob` scores a plan under the heads (temperature 1) from its slot targets: log-softmax of
               each decision the decoder consults, Plackett-Luce for set slots (exact, summed over orders, for
               sets of up to 4), and literal choices summed over candidates that give the same literal.

Known approximations (measured in tests/test_sq2_rl_tiny.py by comparing sampled frequencies with exp(log-prob)):
the projection de-duplication loop, `_fit_part` rewriting a date part, and IN-lists picked by a probability
threshold are not modelled exactly.

GRPO (`grpo_loss`): K plans per question, rewards 1 (same result as gold) / 0.1 (runs) / 0 (fails), advantages
normalised within the group, PPO-clipped ratio at the plan level (ratio = 1 on-policy), plus the k3 estimator of
KL(policy || reference). Questions whose K rewards are all equal carry no gradient and are skipped.

Distillation (`soft_targets`, `distill_loss`): teacher plans (e.g. OmniSQL-32B samples parsed to plans, weighted
by whether they ran right) become per-decision marginals; the loss is cross-entropy to those marginals, i.e.
KL(teacher || model) plus the teacher's entropy (a constant). Pair, value and template heads are conditioned on
the option the teacher chose, the same way gold targets are.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from luthor.sq.candidates import ground as _ground
from luthor.sq.plan import OPS, ROWS, Plan
from luthor.sq2.model import Encoded, LuthorSQ2
from luthor.sq2.targets import (COMPOSE, MAIN_CHOICES, PAIRS, TWO_VALUE_OPS, VOCAB, Scores, Targets, Unsupported, _n_slots,
                      build_targets, decode)

ROWS_ONLY = ("require", "via")         # set slots that point at tables' rows options
NO_ROWS = ("filter", "group", "join")  # set slots that never point at rows options
HAVING_OPS = ("=", "!=", ">", ">=", "<", "<=", "between")


# ---- sampling ----------------------------------------------------------------------------------------

class SampledScores:
    """Scores with Gumbel noise: `decode(SampledScores(...))` draws one plan from the heads at `temp`.
    `base_cache` (shared across draws of one question) holds the model's raw score lists, so K draws cost
    one evaluation of each head plus K cheap decodes."""

    def __init__(self, base: Scores, rnd: random.Random, temp: float = 1.0, base_cache: dict | None = None):
        self.base, self.rnd, self.temp = base, rnd, temp
        self.cache = base_cache if base_cache is not None else {}
        self.noisy: dict = {}
        self.option_ids, self.cands = base.option_ids, base.cands

    def _get(self, key, fn):
        if key not in self.noisy:
            if key not in self.cache:
                self.cache[key] = list(fn())
            g = lambda: -math.log(-math.log(max(self.rnd.random(), 1e-12)))
            self.noisy[key] = TracedVec([x / self.temp + g() for x in self.cache[key]], key)
        return self.noisy[key]

    def choice(self, k):
        return self._get(("c", k), lambda: self.base.choice(k))

    def ptr(self, s):
        return self._get(("p", s), lambda: self.base.ptr(s))

    def pair(self, s, o, h):
        return self._get(("q", s, o, h), lambda: self.base.pair(s, o, h))

    def vals(self, s, o):
        return self._get(("v", s, o), lambda: self.base.vals(s, o))

    def limit(self, s):
        return self._get(("l", s), lambda: self.base.limit(s))


class TracedVec(list):
    """A noisy score vector that tells decode (targets._note) which head it came from."""

    def __init__(self, xs, key):
        super().__init__(xs)
        self.key = key


@dataclass
class Draw:
    """One draw from the heads: the plan (None if decode could not finish it) and every decision decode made,
    as (head key, chosen indices in order, allowed indices or None, approximate)."""
    plan: Plan | None
    trace: list


def sample_draws(sc: Scores, k: int, temp: float = 1.0, seed: int = 0) -> list[Draw]:
    """Exactly k draws, failed ones included (a draw decode cannot finish is an outcome too: reward 0, and its
    partial trace gets the gradient). Unlike sample_plans + plan_targets, no draw is ever dropped, so the
    policy gradient is the plain score-function estimator over everything the policy did."""
    import luthor.sq2.targets as tg
    rnd, cache, out = random.Random(seed), {}, []
    for _ in range(k):
        tg._TRACE = []
        try:
            plan = decode(SampledScores(sc, rnd, temp, cache))
        except Exception:
            plan = None
        finally:
            trace, tg._TRACE = tg._TRACE, None
        out.append(Draw(plan, _dedupe(trace)))
    return out


def _dedupe(trace: list) -> list:
    """decode reads some vectors twice for the same decision (e.g. a projection's template head in the dedup loop
    and again in _dec_expr): within one draw the noise is fixed, so it is one decision, counted once."""
    seen, out = set(), []
    for ev in trace:
        if ev not in seen:
            seen.add(ev)
            out.append(ev)
    return out


def trace_conflicts(trace: list) -> int:
    """Head vectors with two different decisions in one draw (their joint probability is not the product);
    expected 0, logged by rl_train.py."""
    by: dict = {}
    for key, order, allowed, _ in trace:
        by.setdefault(key, set()).add((order, allowed))
    return sum(len(v) > 1 for v in by.values())


def _head(model, e: Encoded, key) -> torch.Tensor:
    kind = key[0]
    if kind == "c":
        return model.choice_logits(e, key[1])
    if kind == "p":
        return model.ptr_scores(e, key[1])
    if kind == "q":
        return model.pair_logits(e, key[1], key[2], key[3])
    if kind == "v":
        return model.value_logits(e, key[1], key[2])
    if kind == "l":
        return model.value_logits(e, key[1], None)
    raise KeyError(key)


def _ordered_pl(s: torch.Tensor, order: tuple, allowed: tuple | None) -> torch.Tensor:
    """log P(the top of s + Gumbel noise, restricted to `allowed`, is `order`, in that order) (Plackett-Luce)."""
    if allowed is not None:
        mask = torch.ones_like(s, dtype=torch.bool)
        mask[list(allowed)] = False
        s = s.masked_fill(mask, float("-inf"))
    lp, taken = s.new_zeros(()), torch.zeros_like(s, dtype=torch.bool)
    for i in order:
        lp = lp + s[i] - torch.logsumexp(s.masked_fill(taken, float("-inf")), 0)
        taken = taken.clone()
        taken[i] = True
    return lp


def trace_logprob(model, e: Encoded, trace: list, temp: float = 1.0) -> torch.Tensor:
    """log pi(draw) at the sampling temperature: the exact sum over the decisions decode made (differentiable).
    With model = HeadMemo, each head and each (decision, choice) term is shared by all draws of a question.
    Exact except IN-lists picked by a probability threshold (flagged approximate in the trace)."""
    terms = []
    for key, order, allowed, _ in trace:
        terms.append(_term(model, (id(e), "t", key, order, allowed, temp),
                           lambda: _ordered_pl(_head(model, e, key).float() / temp, order, allowed)))
    return torch.stack(terms).sum() if terms else e.S.new_zeros((), dtype=torch.float32)


def trace_logprobs(model, e: Encoded, draws: list, temp: float = 1.0, memo=None) -> torch.Tensor:
    memo = memo or HeadMemo(model)
    return torch.stack([trace_logprob(memo, e, d.trace, temp) for d in draws])


def trace_decisions(trace: list) -> int:
    return sum(len(order) for _, order, _, _ in trace)


def sample_plans(sc: Scores, k: int, temp: float = 1.0, seed: int = 0) -> list[Plan]:
    """k plans drawn from the heads (repeats kept: GRPO needs the policy's own frequencies)."""
    rnd, cache, out = random.Random(seed), {}, []
    for _ in range(k * 2):
        if len(out) >= k:
            break
        try:
            out.append(decode(SampledScores(sc, rnd, temp, cache)))
        except Exception:  # a draw the decoder cannot finish (e.g. too few candidates for BETWEEN)
            continue
    return out


# ---- log-probability of a plan -----------------------------------------------------------------------

def _pl_logprob(s: torch.Tensor, chosen: list[int], allowed: torch.Tensor | None) -> torch.Tensor:
    """log P(the unordered set `chosen` is the top-|chosen| of a Gumbel-perturbed `s`), restricted to `allowed`.
    Exact (sum over orders) up to 4 members; beyond that, the order of the current scores."""
    if allowed is not None:
        s = s.masked_fill(~allowed, float("-inf"))
    if not chosen:
        return s.new_zeros(())
    orders = list(itertools.permutations(chosen)) if len(chosen) <= 4 else \
        [tuple(sorted(chosen, key=lambda i: -float(s[i].detach())))]
    terms = []
    for order in orders:
        lp, mask = s.new_zeros(()), torch.zeros_like(s, dtype=torch.bool)
        for i in order:
            lp = lp + s[i] - torch.logsumexp(s.masked_fill(mask, float("-inf")), 0)
            mask = mask.clone()
            mask[i] = True
        terms.append(lp)
    return torch.logsumexp(torch.stack(terms), 0)


def _same_literal(cands, i: int) -> list[int]:
    v = repr(cands[i].value)
    return [j for j, c in enumerate(cands) if repr(c.value) == v]


def _literal_logprob(s: torch.Tensor, cands, idx: list[int]) -> torch.Tensor:
    ls = F.log_softmax(s, 0)
    if len(idx) == 1:  # a plan holds the literal, not the candidate: sum candidates that give the same literal
        return torch.logsumexp(ls[_same_literal(cands, idx[0])], 0)
    idx = list(dict.fromkeys(idx))
    combos = [c for c in itertools.islice(itertools.product(*[_same_literal(cands, i) for i in idx]), 64)
              if len(set(c)) == len(c)]
    return torch.logsumexp(torch.stack([_pl_logprob(s, list(c), None) for c in combos]), 0)


def _consulted(t: Targets) -> tuple[dict, list]:
    """Drop target entries `decode` never consults (so they are not counted as decisions): template numbers past
    the template's {N} slots, compose numbers when compose is not a template, and `fn` / `agg` on rows options."""
    choice = dict(t.choice)
    comp = MAIN_CHOICES["compose"][choice["m.compose"]] if "m.compose" in choice else ""
    n_cn = 0 if comp in COMPOSE else _n_slots(comp, "N")
    for i in range(3):
        if i >= n_cn:
            choice.pop(f"m.cn{i}", None)
    fn_of = {(s, o): c for s, o, h, c in t.pair if h == "fn"}
    pair = []
    for s, o, h, c in t.pair:
        if h in ("fn", "agg") and o.endswith(f".{ROWS}"):  # decode fixes these for rows options (count, no template)
            continue
        if h.startswith("n") and h[1:].isdigit():
            tmpl = VOCAB["templates"][fn_of.get((s, o), 0)] if "fn" in PAIRS else ""
            if int(h[1:]) >= _n_slots(tmpl, "N"):
                continue
        pair.append((s, o, h, c))
    return choice, pair


def _no_class0(name: str) -> bool:
    """Template / compose number heads (n0..n2, m.cn0..cn2): decode takes the argmax over classes 1.. only."""
    h = name.rsplit(".", 1)[-1]
    return h[:-1] in ("n", "cn") and h[-1].isdigit()


def _mask0(z: torch.Tensor) -> torch.Tensor:
    m = torch.zeros_like(z, dtype=torch.bool)
    m[0] = True
    return z.masked_fill(m, float("-inf"))


def _term(model, key, fn):
    """One decision's log-prob; memoised per (question, decision, choice) when model is a HeadMemo."""
    terms = getattr(model, "terms", None)
    if terms is None:
        return fn()
    if key not in terms:
        terms[key] = fn()
    return terms[key]


def plan_logprob(model: LuthorSQ2, e: Encoded, t: Targets, oix: dict[str, int], cands) -> torch.Tensor:
    """log pi(plan) under the heads at temperature 1 (differentiable). With model = HeadMemo, every (decision,
    choice) term is computed once per question and shared by all plans that make it (plan_logprobs)."""
    dev = e.S.device
    q = id(e)
    terms = []
    choice, pair = _consulted(t)
    ids = sorted(oix, key=oix.get)
    rows_key = (q, "rows")
    is_rows = _term(model, rows_key, lambda: torch.tensor([o.endswith(f".{ROWS}") for o in ids], device=dev))

    def lsm_choice(k, c):
        z = model.choice_logits(e, k).float()
        if _no_class0(k) and c != 0:  # decode never picks class 0 ("-") of a consulted compose number
            z = _mask0(z)
        return F.log_softmax(z, 0)[c]

    def lsm_ptr(s, o):
        sc = model.ptr_scores(e, s).float()
        if s.split(".")[1].startswith("x"):
            sc = sc.masked_fill(is_rows, float("-inf"))
        return F.log_softmax(sc, 0)[oix[o]]

    def pl_set(s, os_):
        slot = s.split(".")[1]
        allowed = is_rows if slot in ROWS_ONLY else (~is_rows if slot in NO_ROWS else None)
        return _pl_logprob(model.ptr_scores(e, s).float(), [oix[o] for o in os_], allowed)

    def lsm_pair(s, o, h, c):
        z = model.pair_logits(e, s, oix[o], h).float()
        if s == "m.having" and h == "op":
            z = z.masked_fill(torch.tensor([x not in HAVING_OPS for x in OPS], device=dev), float("-inf"))
        if _no_class0(h) and c != 0:  # nor class 0 of a consulted template number
            z = _mask0(z)
        return F.log_softmax(z, 0)[c]

    for k, c in choice.items():
        terms.append(_term(model, (q, "c", k, c), lambda: lsm_choice(k, c)))
    for s, o in t.ptr.items():
        terms.append(_term(model, (q, "p", s, o), lambda: lsm_ptr(s, o)))
    for s, os_ in t.sets.items():
        if os_:
            terms.append(_term(model, (q, "s", s, tuple(sorted(os_))), lambda: pl_set(s, os_)))
    for s, o, h, c in pair:
        terms.append(_term(model, (q, "q", s, o, h, c), lambda: lsm_pair(s, o, h, c)))
    for s, o, idx in t.vals:
        terms.append(_term(model, (q, "v", s, o, tuple(idx)),
                           lambda: _literal_logprob(model.value_logits(e, s, oix[o]).float(), cands, idx)))
    for s, i in t.limit.items():
        terms.append(_term(model, (q, "l", s, i),
                           lambda: _literal_logprob(model.value_logits(e, s, None).float(), cands, [i])))
    assert e.k.shape[0] == len(ids)
    return torch.stack(terms).sum() if terms else e.S.new_zeros((), dtype=torch.float32)


def decision_count(t: Targets) -> int:
    """Number of decisions plan_logprob sums for these targets (a set slot counts one per member)."""
    choice, pair = _consulted(t)
    return (len(choice) + len(t.ptr) + sum(len(v) for v in t.sets.values()) + len(pair) + len(t.vals)
            + len(t.limit))


def _loose_ground(value, col_id: str, cands) -> int | None:
    """ground(), then any candidate holding this literal. A sampled plan may put a literal stored in another
    column on this one (decode allows it; gold targets never do), and it still needs a log-prob."""
    i = _ground(value, col_id, cands)
    if i is not None:
        return i
    for f in (repr, str):
        for j, c in enumerate(cands):
            if f(c.value) == f(value):
                return j
    return None


def plan_targets(plan: Plan, cands) -> Targets | None:
    """Slot targets of a SAMPLED plan (literals grounded loosely, see _loose_ground); None if not expressible."""
    import luthor.sq2.targets as tg
    tg.ground = _loose_ground
    try:
        return build_targets(plan, cands)
    except (Unsupported, ValueError, KeyError, IndexError):
        return None
    finally:
        tg.ground = _ground


class HeadMemo:
    """Stands in for the model inside plan_logprob and memoises each head output per (question, head, args)
    and each decision's log-prob term per (question, decision, choice).
    The K plans of one group share most decisions (same counts, same top columns), so scoring them through
    one memo evaluates each head once per question instead of once per plan; values and gradients are
    identical (the shared tensors are reused in the graph, and backward accumulates through them)."""

    def __init__(self, model: LuthorSQ2):
        self.m, self.memo, self.calls, self.hits = model, {}, 0, 0
        self.terms: dict = {}  # (question, decision, choice) -> log-prob term (plan_logprob reads it)

    def _get(self, key, fn):
        if key in self.memo:
            self.hits += 1
        else:
            self.calls += 1
            self.memo[key] = fn()
        return self.memo[key]

    def choice_logits(self, e, k):
        return self._get((id(e), "c", k), lambda: self.m.choice_logits(e, k))

    def ptr_scores(self, e, s):
        return self._get((id(e), "p", s), lambda: self.m.ptr_scores(e, s))

    def pair_logits(self, e, s, o, h):
        return self._get((id(e), "q", s, o, h), lambda: self.m.pair_logits(e, s, o, h))

    def value_logits(self, e, s, o):
        return self._get((id(e), "v", s, o), lambda: self.m.value_logits(e, s, o))


def plan_logprobs(model: LuthorSQ2, e: Encoded, ts: list[Targets], oix: dict[str, int], cands,
                  memo: HeadMemo | None = None) -> torch.Tensor:
    """(len(ts),) log pi of several plans of one question, sharing head evaluations (see HeadMemo)."""
    memo = memo or HeadMemo(model)
    return torch.stack([plan_logprob(memo, e, t, oix, cands) for t in ts])


# ---- GRPO ----------------------------------------------------------------------------------------------

def arctic_reward(ran: bool, correct: bool) -> float:
    """Arctic-Text2SQL-R1's reward: 1 right result, 0.1 runs (wrong result), 0 does not compile or run."""
    return 1.0 if correct else (0.1 if ran else 0.0)


def group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float] | None:
    """(r - mean) / std within the group; None when every reward is equal (the group teaches nothing)."""
    if len(rewards) < 2 or max(rewards) - min(rewards) < 1e-9:
        return None
    m = sum(rewards) / len(rewards)
    sd = math.sqrt(sum((r - m) ** 2 for r in rewards) / (len(rewards) - 1))
    return [(r - m) / (sd + eps) for r in rewards]


def grpo_loss(logp: torch.Tensor, adv: list[float], old_logp: torch.Tensor | None = None,
              ref_logp: torch.Tensor | None = None, beta: float = 0.0, clip: float = 0.2,
              kl_est: str = "k2") -> tuple[torch.Tensor, dict]:
    """Plan-level clipped policy gradient over one group plus beta * KL(pi || ref).
    logp, old_logp, ref_logp: (K,) log-probs of the K sampled plans (old = at sampling time; None = on-policy).
    kl_est: "k2" = 0.5 (log pi - log ref)^2 (low variance); "k3" = exp(d) - d - 1 (unbiased, but a plan is a sum of
    ~50 decisions, so exp(d) explodes once the policy has moved: the CPU toy saw 4e5 on one step), d clamped at 10."""
    A = torch.tensor(adv, device=logp.device, dtype=logp.dtype)
    old = logp.detach() if old_logp is None else old_logp
    ratio = torch.exp(logp - old)
    pg = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean()
    kl = logp.new_zeros(())
    if ref_logp is not None:
        d = ref_logp - logp
        kl = (0.5 * d ** 2).mean() if kl_est == "k2" else (torch.exp(d.clamp(max=10.0)) - d - 1).mean()
    return pg + beta * kl, {"pg": float(pg.detach()), "kl_ref": float(kl.detach())}


def passk_advantages(correct: list[bool], k: int) -> list[float] | None:
    """Pass@k advantages (Chen et al. 2025, "Pass@k Training for Adaptively Balancing Exploration and
    Exploitation", arXiv 2508.10751), analytic form over a group of N draws with binary correctness.
    The group's pass@k is R = 1 - C(n_neg, k) / C(N, k). A right draw makes every size-k subset holding it
    right: advantage (1 - R) / sd. A wrong draw's subsets are right unless the other k - 1 are all wrong:
    advantage (1 - R - C(n_neg - 1, k - 1) / C(N - 1, k - 1)) / sd, sd = sqrt(R (1 - R)).
    k = 1 is plain GRPO (population sd). Larger k shrinks the push-down on wrong draws while right ones
    are rare, so the policy is rewarded for keeping several different plans alive (what a top-8 list +
    validator needs) instead of collapsing onto one. None when the group is all right or all wrong."""
    n, npos = len(correct), sum(bool(c) for c in correct)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return None
    k = max(1, min(k, n))
    r = 1.0 - math.comb(nneg, k) / math.comb(n, k)
    sd = math.sqrt(max(r * (1.0 - r), 1e-12))
    a_pos = (1.0 - r) / sd
    a_neg = (1.0 - r - math.comb(nneg - 1, k - 1) / math.comb(n - 1, k - 1)) / sd
    return [a_pos if c else a_neg for c in correct]


def list_loss(logp: torch.Tensor, right: list[bool]) -> torch.Tensor | None:
    """Listwise execution loss over a candidate LIST (e.g. the k-best list the serving decoder produces,
    plus the gold plan): -log( sum_{right} pi / sum_{list} pi ). It moves probability from the wrong plans
    the decoder actually ranks high (its hard negatives) to the right ones, does not penalise right plans
    that differ from gold (equivalent plans), and needs no sampling: the list is the inference list.
    None when the list is all right or all wrong (no ranking signal)."""
    m = torch.tensor(right, device=logp.device, dtype=torch.bool)
    if bool(m.all()) or not bool(m.any()):
        return None
    return torch.logsumexp(logp, 0) - torch.logsumexp(logp[m], 0)


# ---- distillation from teacher plans ---------------------------------------------------------------

@dataclass
class SoftTargets:
    """Per-decision teacher marginals (each inner dict sums to 1)."""
    choice: dict = field(default_factory=dict)  # key -> {class: p}
    ptr: dict = field(default_factory=dict)     # slot -> {option id: p}
    sets: dict = field(default_factory=dict)    # slot -> {option id: p}  (inclusion frequency, normalised)
    pair: dict = field(default_factory=dict)    # (slot, option id, head) -> {class: p}  (given that option)
    vals: dict = field(default_factory=dict)    # (slot, option id) -> {candidate idx: p}
    limit: dict = field(default_factory=dict)   # slot -> {candidate idx: p}
    n: int = 0


def soft_targets(ts: list[Targets], ws: list[float]) -> SoftTargets | None:
    """Weighted per-decision marginals of several plans' targets (weights: e.g. 1 for teacher plans that ran
    right, a small weight for the gold plan, 0 for wrong ones). Conditional heads (pair, vals) are normalised
    over the plans that reached that (slot, option)."""
    acc = {k: defaultdict(lambda: defaultdict(float)) for k in ("choice", "ptr", "sets", "pair", "vals", "limit")}
    tot = 0.0
    for t, w in zip(ts, ws):
        if t is None or w <= 0:
            continue
        tot += w
        choice, pair = _consulted(t)
        for k, c in choice.items():
            acc["choice"][k][c] += w
        for s, o in t.ptr.items():
            acc["ptr"][s][o] += w
        for s, os_ in t.sets.items():
            for o in os_:
                acc["sets"][s][o] += w / len(os_)
        for s, o, h, c in pair:
            acc["pair"][(s, o, h)][c] += w
        for s, o, idx in t.vals:
            for i in dict.fromkeys(idx):
                acc["vals"][(s, o)][i] += w / len(set(idx))
        for s, i in t.limit.items():
            acc["limit"][s][i] += w
    if tot <= 0:
        return None
    norm = lambda d: {k: {c: v / sum(dd.values()) for c, v in dd.items()} for k, dd in d.items() if sum(dd.values()) > 0}
    return SoftTargets(**{k: norm(v) for k, v in acc.items()}, n=len([w for w in ws if w > 0]))


def _soft_ce(z: torch.Tensor, q: dict, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(cross-entropy, KL) of softmax(z) against the sparse distribution q {index: p}."""
    z = z.float()  # bf16 logits under autocast: score in fp32
    if mask is not None:
        z = z.masked_fill(~mask, float("-inf"))
    ls = F.log_softmax(z, 0)
    idx = torch.tensor(list(q), device=z.device, dtype=torch.long)
    p = torch.tensor(list(q.values()), device=z.device, dtype=ls.dtype)
    ce = -(p * ls[idx]).sum()
    return ce, ce + (p * p.clamp_min(1e-12).log()).sum()


def distill_loss(model: LuthorSQ2, e: Encoded, st: SoftTargets, oix: dict[str, int]) -> tuple[torch.Tensor, dict]:
    """Mean over decisions of CE(teacher marginal, model). Returns (loss, {"kl": mean KL, "n": decisions})."""
    dev = e.S.device
    ids = sorted(oix, key=oix.get)
    is_rows = torch.tensor([o.endswith(f".{ROWS}") for o in ids], device=dev)
    ces, kls = [], []

    def add(z, q, mask=None):
        q = {k: v for k, v in q.items() if k is not None}  # options outside a trimmed schema
        tot = sum(q.values())
        q = {k: v / tot for k, v in q.items()} if tot > 0 else {}
        if q:
            ce, kl = _soft_ce(z, q, mask)
            ces.append(ce)
            kls.append(kl)

    for k, q in st.choice.items():
        add(model.choice_logits(e, k), q)
    for s, q in st.ptr.items():
        add(model.ptr_scores(e, s), {oix.get(o): p for o, p in q.items()},
            ~is_rows if s.split(".")[1].startswith("x") else None)
    for s, q in st.sets.items():
        slot = s.split(".")[1]
        add(model.ptr_scores(e, s), {oix.get(o): p for o, p in q.items()},
            is_rows if slot in ROWS_ONLY else (~is_rows if slot in NO_ROWS else None))
    for (s, o, h), q in st.pair.items():
        if o in oix:
            add(model.pair_logits(e, s, oix[o], h), q)
    for (s, o), q in st.vals.items():
        if o in oix:
            add(model.value_logits(e, s, oix[o]), q)
    for s, q in st.limit.items():
        add(model.value_logits(e, s, None), q)
    if not ces:
        z = e.S.new_zeros(())
        return z, {"kl": 0.0, "n": 0}
    return torch.stack(ces).mean(), {"kl": float(torch.stack(kls).detach().mean()), "n": len(ces)}


# ---- execution reward ------------------------------------------------------------------------------------

class ExecReward:
    """Rewards by running compiled plans on read-only SQLite (CPU). Gold results and every distinct SQL's
    result are cached, so repeats in a group (common: the heads are peaked) cost nothing.
    `rewards` takes many (question id, db id, gold sql, sql | None) at once. With workers > 1 the distinct
    queries are spread over worker processes: each database's queries are cut into up to `workers` chunks (one
    RL call is usually ONE database, so splitting by database alone kept a single worker busy). Every query has
    its own deadline (`seconds`, SQLite progress handler; rl_premise measured ~58 ms mean / 1.4 s p99 per BIRD
    query), and a pool that still does not answer within `pool_timeout` is killed and its queries count as not
    run (reward 0), so one stuck worker cannot hang a paid GPU."""

    def __init__(self, db_paths: dict, seconds: float = 5.0, workers: int = 1, min_chunk: int = 4,
                 pool_timeout: float | None = None):
        self.db_paths, self.seconds, self.workers, self.min_chunk = db_paths, seconds, workers, min_chunk
        self.pool_timeout = pool_timeout
        self.gold: dict = {}   # (db, gold sql) -> frozenset of rows | None
        self.res: dict = {}    # (db, sql) -> frozenset of rows | None (None: did not run)
        self.stats = defaultdict(float)  # queries run, chunks, wall seconds, pool timeouts
        self._pool = None

    def _need(self, items):
        by_db = defaultdict(set)
        for _, db, gsql, sql in items:
            if (db, gsql) not in self.gold:
                by_db[db].add(gsql)
            if sql is not None and (db, sql) not in self.res:
                by_db[db].add(sql)
        tasks, dbs = [], []
        for db, qs in by_db.items():
            if db not in self.db_paths:
                continue
            qs = sorted(qs)
            n = max(1, min(self.workers, -(-len(qs) // self.min_chunk)))
            for i in range(n):  # strided chunks: slow queries (similar SQL sort together) spread over workers
                part = qs[i::n]
                if part:
                    tasks.append((str(self.db_paths[db]), part, self.seconds))
                    dbs.append(db)
        return tasks, dbs

    def _map(self, tasks):
        if self.workers <= 1:
            return [_run_db(t) for t in tasks]
        if self._pool is None:
            from multiprocessing import get_context
            self._pool = get_context("spawn").Pool(self.workers)
        limit = self.pool_timeout or (self.seconds * max(len(t[1]) for t in tasks) + 60.0)
        try:
            return self._pool.map_async(_run_db, tasks).get(timeout=limit)
        except Exception as e:  # multiprocessing.TimeoutError, or a dead worker
            print(f"ExecReward: pool failed ({type(e).__name__}) on {sum(len(t[1]) for t in tasks)} queries; "
                  f"they count as not run; pool restarted", flush=True)
            self.stats["pool_failures"] += 1
            self._pool.terminate()
            self._pool = None
            return [[None] * len(t[1]) for t in tasks]

    def rewards(self, items: list[tuple]) -> list[tuple[float, bool, bool]]:
        """items: (question id, db id, gold sql, sql or None) -> per item (reward, ran, correct)."""
        tasks, dbs = self._need(items)
        if tasks:
            t0 = time.time()
            outs = self._map(tasks)
            self.stats["wall_s"] += time.time() - t0
            self.stats["queries"] += sum(len(t[1]) for t in tasks)
            self.stats["chunks"] += len(tasks)
            gold_q = {(db, g) for _, db, g, _ in items}
            for (_, qs, _), db, got in zip(tasks, dbs, outs):
                for q, r in zip(qs, got):
                    if (db, q) in gold_q:
                        self.gold[(db, q)] = r
                    self.res[(db, q)] = r
        out = []
        for _, db, gsql, sql in items:
            ref = self.gold.get((db, gsql))
            got = self.res.get((db, sql)) if sql is not None else None
            ran = got is not None
            ok = ran and ref is not None and got == ref
            out.append((arctic_reward(ran, ok), ran, ok))
        return out

    def gold_rows(self, db: str, gold_sql: str):
        """The cached gold result (after `rewards` saw this question): frozenset of rows, or None if it failed."""
        return self.gold.get((db, gold_sql))

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None


def noisy_gold(rows) -> str | None:
    """Why a gold result teaches nothing reliable, or None: the gold SQL errors / times out, returns no rows (or
    only NULLs), or returns a single 0 (BIRD's broken COUNT / SUM labels: almost any wrong plan matches them).
    The same rule as the --drop-ids list (scripts/analysis/gold_noise.py), applied to the cached gold result."""
    if rows is None:
        return "error"
    vals = [v for r in rows for v in r]
    if not rows or all(v is None for v in vals):
        return "empty"
    if len(rows) == 1 and len(vals) == 1 and vals[0] is not None and not isinstance(vals[0], (str, bytes)) \
            and vals[0] == 0:
        return "single_zero"
    return None


def _run_db(task):
    import sqlite3
    import time
    path, qs, seconds = task
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)
    out = []
    for q in qs:
        box["t"] = time.time() + seconds
        try:
            out.append(frozenset(map(tuple, con.execute(q).fetchmany(100_000))))
        except Exception:
            out.append(None)
    con.close()
    return out
