"""Losses for LuthorSQ. All slot scores are logits with a decision boundary at 0.

1. choice_loss            CE (label smoothing) on every choice head.
2. single_slot_loss       positional slots (proj0..2, order, time, cond0..1): listwise CE over
                          the database's options + a ranking margin against the hardest negatives.
3. set_slot_loss          multi-label slots (group, filter): a two-sided absolute margin around
                          the 0 boundary (positives >= +m, negatives <= -m, softplus-smoothed,
                          so "how many" is decidable by threshold) + the ranking margin
                          (every positive above every hard negative by gamma).
4. pair_loss              CE for aggregation / operator / date part at the gold option.
5. value_pointer_loss     multi-positive CE over literal candidates + ranking margin against the
                          hardest wrong candidate (same-column stored values first).
6. minimal_pair_loss      for twins (one word apart, one slot apart): each question must score
                          its own answer above its twin's answer by gamma, on that slot.
Hard negatives are mined from the model's own current scores, restricted to the options of
the question's database (plus the structural negatives the batch builder put there).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import SET, SINGLE, ExampleOut


def ranking_margin(scores: torch.Tensor, pos: torch.Tensor, gamma: float = 1.0, hard_k: int = 8) -> torch.Tensor:
    """mean over positives p and top-k hardest negatives n of relu(gamma - (s_p - s_n))."""
    neg_mask = torch.ones_like(scores, dtype=torch.bool)
    neg_mask[pos] = False
    if not neg_mask.any() or pos.numel() == 0:
        return scores.new_zeros(())
    neg = scores[neg_mask]
    hard = neg.topk(min(hard_k, neg.numel())).values
    return F.relu(gamma - (scores[pos][:, None] - hard[None, :])).mean()


def two_sided_margin(scores: torch.Tensor, pos: torch.Tensor, m: float = 2.0, hard_k: int = 16) -> torch.Tensor:
    """softplus(m - s_pos) for positives and softplus(m + s_neg) for the hardest negatives."""
    neg_mask = torch.ones_like(scores, dtype=torch.bool)
    neg_mask[pos] = False
    lp = F.softplus(m - scores[pos]).mean() if pos.numel() else scores.new_zeros(())
    neg = scores[neg_mask]
    if neg.numel() == 0:
        return lp
    hard = neg.topk(min(hard_k, neg.numel())).values
    ln = 0.5 * F.softplus(m + hard).mean() + 0.5 * F.softplus(m + neg).mean()
    return lp + ln


def single_slot_loss(scores: torch.Tensor, pos: int, gamma: float = 1.0, hard_k: int = 8) -> torch.Tensor:
    ce = F.cross_entropy(scores[None], torch.tensor([pos], device=scores.device))
    return ce + ranking_margin(scores, torch.tensor([pos], device=scores.device), gamma, hard_k)


def set_slot_loss(scores: torch.Tensor, pos: list[int], m: float = 2.0, gamma: float = 1.0) -> torch.Tensor:
    p = torch.tensor(pos, device=scores.device, dtype=torch.long)
    return two_sided_margin(scores, p, m) + ranking_margin(scores, p, gamma)


def value_pointer_loss(scores: torch.Tensor, pos: list[int], same_col: torch.Tensor | None = None,
                       gamma: float = 1.0) -> torch.Tensor:
    """Multi-positive CE: -log(sum_pos e^s / sum_all e^s), plus a margin against the hardest
    wrong candidate, where wrong stored values of the same column count double."""
    if scores.numel() == 0 or not pos:
        return scores.new_zeros(())
    p = torch.tensor(sorted(set(pos)), device=scores.device, dtype=torch.long)  # a literal used twice counts once
    ce = torch.logsumexp(scores, 0) - torch.logsumexp(scores[p], 0)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[p] = False
    if not mask.any():
        return ce
    boosted = scores + (same_col.float() * 0.5 if same_col is not None else 0.0)
    hardest = boosted[mask].max()
    return ce + F.relu(gamma - (scores[p].min() - hardest))


def choice_loss(logits: dict[str, torch.Tensor], target: dict[str, int], smoothing: float = 0.05) -> torch.Tensor:
    losses = [F.cross_entropy(logits[k][None], torch.tensor([v], device=logits[k].device), label_smoothing=smoothing)
              for k, v in target.items()]
    return torch.stack(losses).mean()


def pair_loss(logits: torch.Tensor, rows: list[int], labels: list[int]) -> torch.Tensor:
    if not rows:
        return logits.new_zeros(())
    return F.cross_entropy(logits[torch.tensor(rows, device=logits.device)],
                           torch.tensor(labels, device=logits.device))


def minimal_pair_margin(own: torch.Tensor, twin: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """own = score of this question's gold answer; twin = its score of the twin's gold answer."""
    return F.relu(gamma - (own - twin))


WEIGHTS = {"choice": 1.0, "single": 1.0, "set": 1.0, "pair": 0.5, "value": 1.0, "limit": 0.5, "twin": 0.5}


def example_loss(model, out: ExampleOut, tgt, opt_index: dict[str, int], gamma: float = 1.0) -> dict[str, torch.Tensor]:
    """All losses for one question. `opt_index` maps option id -> row in out.slots[*]."""
    z = out.ov.new_zeros(())
    L = {k: z for k in WEIGHTS}
    L["choice"] = choice_loss(out.choice, tgt.choice)
    if tgt.choice["outcome"] != 0:  # unsupported: only the outcome head learns
        return L
    singles = [single_slot_loss(out.slots[r], opt_index[oid], gamma) for r, oid in tgt.single.items()
               if r in SINGLE and oid in opt_index]
    if singles:
        L["single"] = torch.stack(singles).mean()
    sets = [set_slot_loss(out.slots[r], [opt_index[o] for o in ids if o in opt_index], gamma=gamma)
            for r, ids in tgt.sets.items() if r in SET]
    if sets:
        L["set"] = torch.stack(sets).mean()
    pairs = []
    for r, a in tgt.agg.items():
        oid = tgt.single.get(r)
        if oid in opt_index:
            pairs.append(pair_loss(out.agg[r], [opt_index[oid]], [a]))
    frows = [opt_index[o] for o in tgt.sets.get("filter", []) if o in opt_index]
    if frows and len(frows) == len(tgt.op["filter"]):
        pairs.append(pair_loss(out.op["filter"], frows, tgt.op["filter"]))
        pairs.append(pair_loss(out.part["filter"], frows, tgt.part["filter"]))
    for r in ("cond0", "cond1"):
        if r in tgt.single and tgt.single[r] in opt_index:
            i = opt_index[tgt.single[r]]
            pairs.append(pair_loss(out.op[r], [i], tgt.op[r]))
            pairs.append(pair_loss(out.part[r], [i], tgt.part[r]))
    if pairs:
        L["pair"] = torch.stack(pairs).mean()
    vals = []
    for fi, oid in enumerate(tgt.sets.get("filter", [])):
        if oid in opt_index and fi < len(tgt.values.get("filter", [])) and tgt.values["filter"][fi]:
            i = opt_index[oid]
            s = model.value_scores(out, out.ov[i], i)
            vals.append(value_pointer_loss(s, tgt.values["filter"][fi], out.cand_col == i, gamma))
    for r in ("cond0", "cond1"):
        if r in tgt.values and tgt.single.get(r) in opt_index:
            i = opt_index[tgt.single[r]]
            s = model.value_scores(out, out.ov[i], i, query=out.qv)
            vals.append(value_pointer_loss(s, tgt.values[r][0], out.cand_col == i, gamma))
    if vals:
        L["value"] = torch.stack(vals).mean()
    if tgt.limit is not None:
        L["limit"] = value_pointer_loss(model.limit_scores(out), [tgt.limit], None, gamma)
    return L


def twin_loss(model, a: tuple, b: tuple, slot: str, gamma: float = 1.0) -> torch.Tensor:
    """a, b = (ExampleOut, Target, opt_index) for a question and its twin."""
    (oa, ta, ia), (ob, tb, ib) = a, b
    if slot == "order_dir":
        da, db = ta.choice["order_dir"], tb.choice["order_dir"]
        if da == db:
            return oa.ov.new_zeros(())
        return minimal_pair_margin(oa.choice["order_dir"][da], oa.choice["order_dir"][db], gamma) + \
            minimal_pair_margin(ob.choice["order_dir"][db], ob.choice["order_dir"][da], gamma)
    if slot == "value":
        losses = []
        for fi, oid in enumerate(ta.sets.get("filter", [])):
            if oid not in ia or oid not in ib or fi >= len(tb.values.get("filter", [])):
                continue
            va, vb = ta.values["filter"][fi], tb.values["filter"][fi]
            if not va or not vb:
                continue
            i = ia[oid]  # twins share their database batch, so option rows coincide
            sa_own = model.value_scores(oa, oa.ov[i], i)[va[0]]
            sa_twin = model.value_scores(oa, oa.ov[i], i, cands=ob.cands[vb[0]:vb[0] + 1],
                                         cand_col=ob.cand_col[vb[0]:vb[0] + 1], mentioned=ob.mentioned[vb[0]:vb[0] + 1])[0]
            sb_own = model.value_scores(ob, ob.ov[i], i)[vb[0]]
            sb_twin = model.value_scores(ob, ob.ov[i], i, cands=oa.cands[va[0]:va[0] + 1],
                                         cand_col=oa.cand_col[va[0]:va[0] + 1], mentioned=oa.mentioned[va[0]:va[0] + 1])[0]
            losses.append(minimal_pair_margin(sa_own, sa_twin, gamma) + minimal_pair_margin(sb_own, sb_twin, gamma))
        return torch.stack(losses).mean() if losses else oa.ov.new_zeros(())
    return oa.ov.new_zeros(())
