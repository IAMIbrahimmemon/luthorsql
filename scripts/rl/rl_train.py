"""GRPO with execution reward on the System-1 slot model (one pass, factorised slot policy).

Policy   the slot heads: every decision `decode` makes is an argmax over one score vector; Gumbel noise on each
         vector (rl_core.SampledScores) makes `decode` an exact draw from softmax(score / T).
Rollout  ONE backbone forward per question (with grad), G plans sampled from the cached head scores (CPU work, no
         token generation), compiled by the deterministic compiler, executed on read-only SQLite in worker processes
         (each database's queries split over --workers, a deadline per query: rl_core.ExecReward).
Reward   Arctic-Text2SQL-R1's: 1 same rows as gold, 0.1 runs, 0 otherwise. Gold = the ORIGINAL training SQL, so
         questions the grammar cannot express exactly still teach whenever an expressible plan returns the same rows.
Update   group-normalised advantages (GRPO; or pass@k advantages, --adv passk), on-policy (ratio = 1, one update per
         rollout batch), log pi(draw) = exact sum over the decisions decode made (rl_core.trace_logprob; every draw
         scored, failed decodes included with reward 0), divided by the
         number of decisions (--decision-norm; the fanout-R1 CPU toy found the summed form 4-8x the SFT gradient and
         unstable), + beta * KL(pi || SFT) (k2 estimator; reference = a frozen copy of the SFT LoRA ("ref" adapter)
         and heads), + anchors:
           --sft-weight   cross-entropy on the gold plan
           --list-weight  k-best list anchor (rl_core.list_loss): the serving decoder's own distinct k-best list
                          (--list-k, topk.kbest) + the gold plan (+ right off-policy buffer plans), labelled by
                          execution against the ORIGINAL gold; -log(right mass / list mass). In the CPU toy
                          `sft+list` kept oracle@8 at 1.0 in all six logs where pure GRPO / pass@k collapsed it to 0.25
                          (reports/fanout_r1/objectives/r2/*.log).
           --distill-weight  cross-entropy to the per-decision marginals of the right buffer plans
                          (rl_core.soft_targets / distill_loss); needs --buffer.
Buffer   --buffer (04-01, default off): OmniSQL-32B x 5 SQL candidates per question (data/eval/sel_cands/sel), parsed
         to plans (luthor.sq.convert); held-out databases removed (asserted). Labels: re-executed against the ORIGINAL
         gold (--buffer-label original, default) or the file's BIRD-Verified flags (--buffer-label verified).
Filters  --drop-ids PATH (e.g. data/sys1/gold_noise_drop.json), --skip-noisy-gold (drop a question at run time when
         its ORIGINAL gold errors, returns nothing, or returns a single 0), --skip-dbs NAME..., --exclude-dbs JSON,
         --drop-dbs DB,DB|JSON (also out of the buffer; e.g. data/sys1/contaminated_dbs.json).
Grammar  --grammar v3|v4 (default: LUTHOR_GRAMMAR, else the --run's args.json, missing = v3); must match the run.
Memory   an out-of-memory error in the forward pass retries the database's questions in halves, weighted so the
         gradient is unchanged; one in backward drops the whole step; the run stops past --max-oom-skip
         (train_sq2.OOMLedger). --grad-ckpt: gradient checkpointing (the model runs in train mode with every dropout
         at 0, since HF only checkpoints in train mode).
Timing   every step logs seconds per question for backbone forward, sampling, k-best, SQLite execution, log-probs,
         list anchor and backward. --timing-smoke N runs N steps (no checkpoint) and prints projected GPU-h for 1
         and 2 epochs over --project-questions (default 8,000 BIRD-train questions) -> OUT/timing.json.

  # pilot arm A (GPU): GRPO + SFT anchor + list anchor, from R2
  python -I scripts/rl/rl_train.py --run runs/r1r2/r2 --plans data/sys1/bird_train \
      --db-roots data/sys1/dbs data/sys1/rrdbs --exclude-dbs data/sys1/dev_dbs_bird.json --skip-dbs works_cycles \
      --epochs 1 --G 32 --q-per-step 32 --sft-weight 0.5 --list-weight 0.5 --beta 0.01 --seed 0 \
      --out runs/rl1/grpo --device cuda --heldout-eval
  # arm B: the same + --adv passk --passk 8
  # RL timing test (Phase 1, cheap GPU box): ... --timing-smoke 20 --out runs/rl_timing
  # smoke test (CPU, tiny random Qwen2 backbone, ~1-2 min): proves sampling, reward, gradient flow and an update
  python -I scripts/rl/rl_train.py --tiny --plans data/sys1/bird_train --db-roots data/sys1/rrdbs \
      --only-db trains --tiny-sft-steps 40 --steps 4 --G 8 --q-per-step 8 --beta 0.01 --device cpu --out smoke

Writes OUT/rl_args.json (args, seed, git rev, file hashes, data counts) at start, OUT/rl_log.json (one record per
step), OUT/oom.json, and (7B) best_lora/ + best_heads.pt + the R2 run's args.json / vocab.json so eval_sq2 /
rerank_sq2 load it as a run. Held-out evaluation: heldout_eval.sh (k=8 lists on the 1,000 held-out questions,
first pick / oracle@8, then the frozen R2 validator).

Dev is never read here. Sealed sets are never read here.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import platform
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (HERE, R / "scripts", R / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if __name__ == "__main__":  # --grammar: LUTHOR_GRAMMAR before the first luthor.sq import (workers inherit it)
    from luthor.grammar_cli import preset
    preset()  # --grammar, else LUTHOR_GRAMMAR, else --run's args.json (missing = v3; default --run R2 is v3)
import train_sq2  # noqa: E402
from luthor import grammar_cli  # noqa: E402
import luthor.sq2.train as tr  # noqa: E402
from luthor.sq.candidates import ValueIndex, find_candidates  # noqa: E402
from luthor.sq.model import Backbone  # noqa: E402
from luthor.sq.plan import Plan, compile_plan  # noqa: E402
from luthor.sq2.model import LuthorSQ2, TorchScores  # noqa: E402
from luthor.sq2.targets import build_targets  # noqa: E402
from luthor.sq2.train import BackwardOOM  # noqa: E402
from rl_core import (ExecReward, HeadMemo, decision_count, distill_loss, group_advantages, grpo_loss,  # noqa: E402
                     list_loss, noisy_gold, passk_advantages, plan_logprobs, plan_targets, sample_draws,
                     soft_targets, trace_conflicts, trace_decisions, trace_logprobs)
import rl_data  # noqa: E402

PHASES = ("t_forward", "t_sample", "t_kbest", "t_exec", "t_logp", "t_list", "t_backward")


# ---- runtime helpers --------------------------------------------------------------------------------------

def _sync(device: str) -> None:
    """Wait for queued GPU work, so a phase's wall time is its own (CUDA kernels run asynchronously)."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def _empty_cache(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or \
        (isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def run_meta(a, extra: dict) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(R), *args], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    status = git("status", "--porcelain")
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
             for f in (HERE / "rl_train.py", HERE / "rl_core.py", HERE / "rl_data.py") if f.exists()}
    return {"args": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v)
                     for k, v in vars(a).items()},
            "seed": a.seed, "git_rev": git("rev-parse", "HEAD"),
            "git_dirty_files": len(status.splitlines()) if status is not None else None, "file_sha256": files,
            "argv": sys.argv, "torch": torch.__version__, "python": platform.python_version(),
            "host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}


def policy_mode(m: LuthorSQ2, grad_ckpt: bool) -> None:
    """eval() (no dropout: sampling and log-probs describe the same policy). With --grad-ckpt: train() with every
    dropout at 0 instead, because HF layers and LuthorSQ2's prefix cache checkpoint only in train mode."""
    if not grad_ckpt:
        m.eval()
        return
    m.train()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    assert getattr(m, "mention_drop", 0.0) == 0.0


# ---- model ------------------------------------------------------------------------------------------------

def tiny_model(seed: int, lora_r: int = 8, grad_ckpt: bool = False) -> LuthorSQ2:
    """A 2-layer random Qwen2 with the OmniSQL tokenizer and LoRA: the real pipeline at toy scale."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
    tok = AutoTokenizer.from_pretrained("seeklhy/OmniSQL-7B")
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32768)
    torch.manual_seed(seed)
    body = Qwen2Model(cfg)
    if grad_ckpt:
        body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    body = get_peft_model(body, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                                           target_modules="all-linear"))
    for n, p in body.named_parameters():  # tiny model: train everything (no pretrained weights to protect)
        p.requires_grad = True
    return LuthorSQ2(Backbone(body, tok), d_proj=32, col_mentions=True, value_side=True)


class _DequantLinear(torch.autograd.Function):
    """y = x W^T with W = int8 q * scale, re-dequantised in backward: plain F.linear would save the bf16 W of all
    196 linears for backward (~15 GB; the first Mac 7B step ran out of MPS memory that way)."""

    @staticmethod
    def forward(ctx, x, q, s, bias):
        ctx.save_for_backward(q, s)
        return torch.nn.functional.linear(x, q.to(x.dtype) * s.to(x.dtype), bias)

    @staticmethod
    def backward(ctx, g):
        q, s = ctx.saved_tensors
        return g @ (q.to(g.dtype) * s.to(g.dtype)), None, None, None


def _qlinear_forward(self, x):
    if torch.is_grad_enabled():
        return _DequantLinear.apply(x, self.q, self.s, self.bias)
    return torch.nn.functional.linear(x, self.q.to(x.dtype) * self.s.to(x.dtype), self.bias)


def real_model(run: Path, device: str, quant: str, grad_ckpt: bool = False) -> tuple[LuthorSQ2, dict]:
    from eval_sq2 import build_model, run_args
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer
    ra = run_args(run)
    tok = AutoTokenizer.from_pretrained(ra["base"])
    body = AutoModel.from_pretrained(ra["base"], dtype=torch.bfloat16, device_map="cpu" if quant == "int8" else device)
    if grad_ckpt:  # before LoRA and before LuthorSQ2 (which detects it and keeps the schema prefix cache)
        body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    body = get_peft_model(body, LoraConfig(r=int(ra["lora_r"]), lora_alpha=64, lora_dropout=0.05,
                                           target_modules="all-linear"))
    m = build_model(Backbone(body, tok), ra)
    set_peft_model_state_dict(body, load_file(str(run / "best_lora/adapter_model.safetensors")))
    m.load_state_dict(torch.load(run / "best_heads.pt", map_location="cpu"), strict=False)
    body.to(torch.bfloat16)
    if quant == "int8":  # Mac only: weight-only int8 base (reports/fanout_r2/zero_gpu/mac_gen.py)
        sys.path.insert(0, str(R / "reports/fanout_r2/zero_gpu"))
        import mac_gen
        mac_gen.QLinear.forward = _qlinear_forward  # autograd keeps int8, not a bf16 copy of every weight
        mac_gen.quantize_int8(body, device)
    return m.to(device), ra


def add_reference(m: LuthorSQ2, build, tiny: bool) -> LuthorSQ2:
    """Frozen copy of the starting policy for the KL term. Real model: a second LoRA adapter "ref" on the SAME base
    weights (no second 7B copy) plus a second set of heads sharing the backbone. Tiny model (whole backbone trains):
    a full deep copy."""
    if tiny:
        import copy
        ref = copy.deepcopy(m)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref._own_body = True
        return ref.eval()
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    body = m.backbone.model
    body.add_adapter("ref", body.peft_config["default"])
    set_peft_model_state_dict(body, get_peft_model_state_dict(body, adapter_name="default"), adapter_name="ref")
    body.set_adapter("default")
    ref = build(m.backbone)
    ref.load_state_dict(m.head_state(), strict=False)
    for n, p in ref.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad_(False)
    ref._own_body = False
    return ref.to(m.device)


# ---- data -------------------------------------------------------------------------------------------------

def load_questions(a):
    cards, exs, gold = train_sq2.load_sets(a.plans, workers=4, strict=False, all_rows=True)
    excl = set(json.loads(a.exclude_dbs.read_text())) if a.exclude_dbs else set()
    skip = set(a.skip_dbs or [])
    ddbs = train_sq2.load_drop_dbs(a.drop_dbs)
    drop = rl_data.load_drop_ids(a.drop_ids) if a.drop_ids else set()
    keep, st = [], collections.Counter()
    for e in exs:
        r = gold[e.id]
        if e.db_id in excl or (a.only_db and e.db_id != a.only_db):
            continue
        if not a.twins and (r.get("twin_of") or "_twin" in e.id):  # generated question variants: off by default
            continue
        if e.db_id in skip:
            st["skip_dbs"] += 1
            continue
        if e.db_id in ddbs:
            st["drop_dbs"] += 1
            st[f"drop_dbs:{e.db_id}"] += 1
            continue
        if e.id in drop:
            st["drop_ids"] += 1
            continue
        keep.append(e)
    if a.drop_ids:
        print(f"--drop-ids {a.drop_ids}: {len(drop)} ids, {st['drop_ids']} training questions dropped", flush=True)
    if skip:
        print(f"--skip-dbs {sorted(skip)}: {st['skip_dbs']} questions dropped", flush=True)
    if ddbs:
        print(f"--drop-dbs {sorted(ddbs)}: {st['drop_dbs']} training questions dropped "
              f"({ {k[9:]: v for k, v in st.items() if k.startswith('drop_dbs:')} })", flush=True)
    return cards, keep, gold, dict(st, drop_ids_listed=len(drop))


def _ckey(cands):
    return [(c.kind, repr(c.value), c.col) for c in cands]


def serve_cands(exs, cards, gold, paths, order):
    """Serving-time literal candidates (card values + live DB lookup), as eval / rerank build them. When that list
    differs from the training-time one, the gold targets are rebuilt against it (value targets point at candidate
    positions; a gold literal serving does not offer is dropped, as make_example does). Returns counts."""
    by = collections.defaultdict(list)
    st = collections.Counter()
    for e in exs:
        by[e.db_id].append(e)
    for db, es in by.items():
        con = sqlite3.connect(f"file:{paths[db]}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "replace")
        vi = ValueIndex.build(con, cards[db])
        for e in es:
            b = tr.question_body(gold[e.id])
            full = find_candidates(b, cards[db], extra=vi.match(b))
            new, m = tr.select_cands(full, order=order, body=b)
            if e.t is not None and _ckey(new) != _ckey(e.cands):
                st["cands_changed"] += 1
                try:
                    e.t = tr._remap(build_targets(Plan.from_dict(gold[e.id]["plan"]), full), m)
                except Exception:
                    e.t = None
                    st["targets_lost"] += 1
            e.cands = new
        con.close()
    return st


# ---- one database step ------------------------------------------------------------------------------------

def _buffer_plans(e, g, buf: dict, cache: dict) -> list:
    """The question's off-policy buffer SQL, parsed to grammar plans the heads can score:
    [(compiled sql, targets, ok_verified)]. Cached per question (candidates are fixed after serve_cands)."""
    if e.id not in cache:
        from luthor.sq.convert import convert
        out, seen = [], set()
        rec = buf.get(e.id)
        for sql, okv in zip(rec["sqls"], rec["ok_verified"]) if rec else ():
            try:
                c = convert(sql, g)
                if c.plan is None:
                    continue
                cs = compile_plan(c.plan, g)
                t = plan_targets(c.plan, e.cands)
            except Exception:
                continue
            if t is not None and cs not in seen:
                seen.add(cs)
                out.append((cs, t, okv))
        cache[e.id] = out
    return cache[e.id]


def rl_db_forward(a, m, ref, card, group, gold, rewarder, rnd, step_seed, log, buf=None, buf_cache=None,
                  qoff: int = 0, scale: float = 1.0):
    """Rollouts + loss for the questions of one database (one schema prefix): forward only. `qoff` keeps each
    question's sampling seed when a group is split after an out-of-memory error. Returns the loss tensor, already
    divided by len(group) and multiplied by `scale`, or None."""
    dev = a.device
    t = time.time()
    tabs = None
    if tr.serve_tables(card, "") is not None:  # schema too long: the training-time table trim
        tabs = tr.train_tables(card, set().union(*(e.tables for e in group)), a.budget_chars, rnd)
    st, oix, enc = tr.encode_db(m, card, group, tabs)
    _sync(dev)
    log["t_forward"] += time.time() - t
    g = card.graph()

    def comp(p):
        try:
            return compile_plan(p, g)
        except Exception:
            return None

    rolls = []
    for qi, (e, x) in enumerate(zip(group, enc)):
        t = time.time()
        draws, sqls = [], []
        if a.rl_weight > 0:
            with torch.no_grad():  # exactly G draws, failed decodes included (reward 0): nothing is dropped
                draws = sample_draws(TorchScores(m, x, st.option_ids, e.cands), a.G, a.temp, step_seed + qoff + qi)
            sqls = [comp(d.plan) if d.plan is not None else None for d in draws]
            log["draw_failed"] += sum(d.plan is None for d in draws)
            log["trace_conflicts"] += sum(trace_conflicts(d.trace) for d in draws)
        log["t_sample"] += time.time() - t
        t = time.time()
        kb, kb_sql = [], []
        if a.list_weight > 0:
            with torch.no_grad():
                try:
                    kb = tr_kbest(TorchScores(m, x, st.option_ids, e.cands), a.list_k, comp)
                except Exception:
                    kb = []
            kb_sql = [comp(p) for p in kb]
        bp = _buffer_plans(e, g, buf, buf_cache) if buf is not None else []
        log["t_kbest"] += time.time() - t
        rolls.append((e, x, draws, sqls, kb, kb_sql, bp))

    t = time.time()  # one execution batch per database: samples, k-best list, buffer plans
    items = []
    for e, _, _, sqls, _, kb_sql, bp in rolls:
        gs = gold[e.id]["gold_sql"]
        items += [(e.id, e.db_id, gs, s) for s in sqls] + [(e.id, e.db_id, gs, s) for s in kb_sql]
        items += [(e.id, e.db_id, gs, s) for s, _, _ in bp]
        items.append((e.id, e.db_id, gs, None))  # the gold alone: its result is cached either way (--skip-noisy-gold)
    rew = rewarder.rewards(items)
    log["t_exec"] += time.time() - t

    losses, i = [], 0
    ref_enc = None
    for qi, (e, x, draws, sqls, kb, kb_sql, bp) in enumerate(rolls):
        rs = rew[i:i + len(sqls)]
        i += len(sqls)
        rk = rew[i:i + len(kb_sql)]
        i += len(kb_sql)
        rb = [ok for _, _, ok in rew[i:i + len(bp)]] if a.buffer_label == "original" else [okv for _, _, okv in bp]
        i += len(bp) + 1
        log["q"] += 1
        if a.skip_noisy_gold:
            why = noisy_gold(rewarder.gold_rows(e.db_id, gold[e.id]["gold_sql"]))
            if why is not None:
                log["gold_noisy"] += 1
                log[f"gold_noisy_{why}"] += 1
                continue
        log["plans"] += len(draws)
        log["reward_sum"] += sum(r for r, _, _ in rs)
        log["correct_sum"] += sum(ok for _, _, ok in rs)
        log["distinct_sql"] += len({s for s in sqls if s})
        memo = HeadMemo(m)
        t = time.time()
        if a.sft_weight > 0 and e.t is not None:
            l_sft, _ = tr.example_loss(m, x, e.t, oix)
            losses.append(a.sft_weight * l_sft)
            log["sft_q"] += 1
        adv = None
        if draws:
            rewards = [r for r, _, _ in rs]
            adv = passk_advantages([ok for _, _, ok in rs], a.passk) if a.adv == "passk" else group_advantages(rewards)
        if adv is not None:
            # every draw is scored by the decisions decode actually made (the old path re-derived slot targets from
            # the finished plan and dropped the ~25% it could not, biasing the gradient)
            log["plans_scored"] += len(draws)
            log["groups_with_signal"] += 1
            logp = trace_logprobs(m, x, draws, a.temp, memo)
            nd = torch.tensor([max(trace_decisions(d.trace), 1) for d in draws], device=logp.device, dtype=logp.dtype)
            if a.decision_norm:
                logp = logp / nd
            ref_logp = None
            if ref is not None and a.beta > 0:
                with torch.no_grad():
                    if ref_enc is None:
                        if ref._own_body:
                            _, _, ref_enc = tr.encode_db(ref, card, group, tabs)
                        else:
                            body = m.backbone.model
                            body.set_adapter("ref")
                            try:
                                _, _, ref_enc = tr.encode_db(ref, card, group, tabs)
                            finally:
                                body.set_adapter("default")  # also restores requires_grad on the policy adapter only
                    ref_logp = trace_logprobs(ref, ref_enc[qi], draws, a.temp, HeadMemo(ref))
                    if a.decision_norm:
                        ref_logp = ref_logp / nd
            l_rl, parts = grpo_loss(logp, adv, ref_logp=ref_logp, beta=a.beta)
            log["pg"] += parts["pg"]
            log["kl"] += parts["kl_ref"]
            losses.append(a.rl_weight * l_rl)
        log["t_logp"] += time.time() - t
        t = time.time()
        right_buf = [tb for (_, tb, _), ok in zip(bp, rb) if ok]
        log["buffer_plans"] += len(bp)
        log["buffer_right"] += len(right_buf)
        log["buffer_q_right"] += bool(right_buf)
        if a.list_weight > 0:
            ts, right, seen = [], [], set()
            for p, s, (_, _, ok) in zip(kb, kb_sql, rk):
                tt = plan_targets(p, e.cands) if s is not None else None
                if tt is not None:
                    ts.append(tt)
                    right.append(bool(ok))
                    seen.add(s)
            log["list_oracle"] += any(ok for _, _, ok in rk)
            log["list_first_right"] += bool(rk and rk[0][2])
            if a.list_gold and e.t is not None and gold[e.id].get("compiled_sql") not in seen:
                ts.append(e.t)  # the verified gold plan, right by construction
                right.append(True)
            for (s, tb, _), ok in zip(bp, rb):
                if ok and s not in seen:
                    seen.add(s)
                    ts.append(tb)
                    right.append(True)
            if len(ts) >= 2 and any(right) and not all(right):
                lp = plan_logprobs(m, x, ts, oix, e.cands, memo)
                if a.decision_norm:  # the toy's norm: one constant per list (direction unchanged)
                    lp = lp / max(sum(decision_count(tt) for tt in ts) / len(ts), 1.0)
                ll = list_loss(lp, right)
                if ll is not None:
                    losses.append(a.list_weight * ll)
                    log["list_groups"] += 1
                    log["list_size"] += len(ts)
                    log["list_loss"] += float(ll.detach())
        if a.distill_weight > 0 and right_buf:
            stt = soft_targets(right_buf, [1.0] * len(right_buf))
            if stt is not None:
                ld, info = distill_loss(m, x, stt, oix)
                if info["n"]:
                    losses.append(a.distill_weight * ld)
                    log["distill_q"] += 1
                    log["distill_kl"] += info["kl"]
        log["t_list"] += time.time() - t
    _sync(dev)
    return torch.stack(losses).sum() / max(len(group), 1) * scale if losses else None


def tr_kbest(sc, k, comp):
    from luthor.sq2.topk import kbest
    return kbest(sc, k, comp)


def run_rl_db(a, m, ref, card, db, group, gold, rewarder, rnd, step_seed, log, ledger, scale: float,
              buf=None, buf_cache=None, qoff: int = 0) -> int:
    """Forward + backward for one database's questions. An out-of-memory error in the forward pass leaves the
    gradients untouched, so the questions are retried in halves (each weighted by its share, so the summed gradient
    is the one the whole group would give), down to single questions; a question that still does not fit is skipped
    and counted in `ledger`. An OOM in backward raises BackwardOOM (the caller drops the step).
    Returns the number of questions trained."""
    loc = collections.Counter()
    oom = False
    try:
        loss = rl_db_forward(a, m, ref, card, group, gold, rewarder, rnd, step_seed, loc, buf, buf_cache, qoff, scale)
        if loss is not None:
            if not torch.isfinite(loss):  # never back-propagate a non-finite loss
                loc["nonfinite_loss"] += 1
                print(f"non-finite loss on {db}: skipped", flush=True)
            else:
                t = time.time()
                try:
                    loss.backward()
                except Exception as e:
                    if is_oom(e):
                        raise BackwardOOM(db) from e
                    raise
                _sync(a.device)
                loc["t_backward"] += time.time() - t
        del loss
    except BackwardOOM:
        raise
    except Exception as e:
        if not is_oom(e):
            raise
        oom = True  # recover outside the except block, so the failed graph (held by the traceback) is freed
    if not oom:
        log.update(loc)
        return len(group)
    loc = None
    _empty_cache(a.device)
    if len(group) == 1:
        print(f"OOM on {db} (1 question, forward): skipped", flush=True)
        ledger.skip(db, 1)
        log["oom_skipped"] += 1
        return 0
    ledger.retries += 1
    log["oom_retries"] += 1
    h = len(group) // 2
    n = 0
    for part, off in ((group[:h], 0), (group[h:], h)):
        n += run_rl_db(a, m, ref, card, db, part, gold, rewarder, rnd, step_seed, log, ledger,
                       scale * len(part) / len(group), buf, buf_cache, qoff + off)
    return n


def group_stream(exs, q_per_db: int, seed: int):
    """(db, questions) groups forever; a fresh shuffle each epoch (seeded)."""
    ep = 0
    while True:
        for x in tr.db_groups(exs, q_per_db, seed * 1000 + ep):
            yield x
        ep += 1


# ---- main -------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=R / "runs/r1r2/r2", help="SFT checkpoint (ignored with --tiny)")
    ap.add_argument("--tiny", action="store_true", help="tiny random backbone (smoke test)")
    ap.add_argument("--tiny-sft-steps", type=int, default=40, help="tiny: SFT warm-up so rewards are mixed")
    ap.add_argument("--plans", nargs="+", type=Path, required=True)
    ap.add_argument("--db-roots", nargs="+", required=True)
    ap.add_argument("--exclude-dbs", type=Path, help="held-out training DBs (never trained on)")
    ap.add_argument("--skip-dbs", nargs="*", default=[], help="databases left out of training (e.g. works_cycles: "
                    "487 OOM skips in R2)")
    ap.add_argument("--drop-dbs", default=None, metavar="DB,DB|FILE.json",
                    help="databases removed from training and from the --buffer (train_sq2.py --drop-dbs; e.g. "
                         "data/sys1/contaminated_dbs.json)")
    grammar_cli.add_argument(ap)
    ap.add_argument("--drop-ids", type=Path, help="question ids to leave out (e.g. data/sys1/gold_noise_drop.json)")
    ap.add_argument("--skip-noisy-gold", action="store_true",
                    help="no loss on a question whose ORIGINAL gold errors, returns nothing, or returns a single 0")
    ap.add_argument("--only-db")
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--twins", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quant", default="bf16", choices=["bf16", "int8"])
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing on the backbone")
    ap.add_argument("--epochs", type=float, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--q-per-step", type=int, default=32)
    ap.add_argument("--q-per-db", type=int, default=8)
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--adv", choices=["grpo", "passk"], default="grpo")
    ap.add_argument("--passk", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--rl-weight", type=float, default=1.0)
    ap.add_argument("--sft-weight", type=float, default=0.0)
    ap.add_argument("--list-weight", type=float, default=0.0, help="k-best list anchor (rl_core.list_loss)")
    ap.add_argument("--list-k", type=int, default=8)
    ap.add_argument("--list-gold", action=argparse.BooleanOptionalAction, default=True,
                    help="add the gold plan to the list (right) when the k-best list lacks it")
    ap.add_argument("--distill-weight", type=float, default=0.0, help="CE to right buffer plans' marginals (--buffer)")
    ap.add_argument("--buffer", nargs="*", type=Path, default=None,
                    help="off-policy SQL buffer (04-01); no paths = data/eval/sel_cands/sel/{train,val}_cands.jsonl")
    ap.add_argument("--buffer-label", choices=["original", "verified"], default="original",
                    help="right = same rows as the ORIGINAL gold (re-executed) or the file's BIRD-Verified flag")
    ap.add_argument("--decision-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--budget-chars", type=int, default=9000)
    ap.add_argument("--exec-seconds", type=float, default=5.0, help="deadline per SQL query")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="SQLite worker processes")
    ap.add_argument("--max-oom-skip", type=float, default=0.002)
    ap.add_argument("--oom-min-seen", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timing-smoke", type=int, default=0, metavar="N",
                    help="run N steps, no checkpoint; print projected GPU-h for 1 and 2 epochs")
    ap.add_argument("--project-questions", type=int, default=8000)
    ap.add_argument("--heldout-eval", action="store_true", help="after training: heldout_eval.sh on OUT (7B only)")
    a = ap.parse_args(argv)
    if a.buffer is not None and not a.buffer:
        a.buffer = list(rl_data.BUFFER_FILES)
    if a.distill_weight > 0 and a.buffer is None:
        ap.error("--distill-weight needs --buffer")
    if a.buffer is not None and a.list_weight <= 0 and a.distill_weight <= 0:
        ap.error("--buffer is used by the list anchor (--list-weight) or --distill-weight; set one")
    return a


def timing_report(hist: list[dict], a, startup_s: float) -> dict:
    use = hist[1:] if len(hist) > 1 else hist  # step 0 pays one-off costs (pool start, allocator warm-up)
    q = sum(r["q"] for r in use)
    s = sum(r["step_s"] for r in use)
    spq = s / max(q, 1)
    phases = {k[2:]: round(sum(r["s_per_q"][k[2:]] * r["q"] for r in use) / max(q, 1), 4) for k in PHASES}
    out = {"steps_measured": len(use), "questions_measured": q, "s_per_question": round(spq, 4),
           "s_per_question_by_phase": phases, "startup_s": round(startup_s, 1), "G": a.G, "q_per_step": a.q_per_step,
           "list_weight": a.list_weight, "beta": a.beta, "grad_ckpt": a.grad_ckpt, "workers": a.workers,
           "project_questions": a.project_questions,
           "gpu_h": {f"{ep}_epoch": round((spq * a.project_questions * ep + startup_s) / 3600, 3) for ep in (1, 2)},
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None}
    return out


def main(argv=None):
    a = parse_args(argv)
    # the SFT run's grammar (args.json, missing = v3) must be this process's; --tiny uses R2's v3 vocabulary
    a.grammar = grammar_cli.resolve(a.grammar) if a.tiny else grammar_cli.check_run(a.run, a.grammar)
    t_start = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    rnd = random.Random(a.seed)
    if str(a.device).startswith("cuda"):  # train_sq2: cuDNN's fused attention backward gave non-finite gradients
        torch.backends.cuda.enable_cudnn_sdp(False)

    from eval_sq2 import run_args, use_cand_order
    ra = {} if a.tiny else run_args(a.run)
    v = json.loads(((R / "runs/r1r2/r2") if a.tiny else a.run).joinpath("vocab.json").read_text())  # grammar-v3 vocab
    vocab = (v["templates"], v["nums"])
    train_sq2.VOCAB_ARGS = vocab
    train_sq2._init_worker(vocab)
    order = use_cand_order(ra if ra else {"cand_order": "v2"})

    cards, exs, gold, fstats = load_questions(a)
    paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, str(p))
    exs = [e for e in exs if e.db_id in paths and e.db_id in cards]
    rnd.shuffle(exs)
    if a.max_q:
        exs = exs[:a.max_q]
    t = time.time()
    cstats = serve_cands(exs, cards, gold, paths, order)
    print(f"{len(exs)} questions on {len({e.db_id for e in exs})} DBs, covered {sum(e.t is not None for e in exs)}; "
          f"serve candidates in {time.time() - t:.0f}s ({dict(cstats)})", flush=True)

    buf, bstats = None, {}
    if a.buffer is not None:
        held = rl_data.heldout_dbs()
        if a.exclude_dbs:
            held |= set(json.loads(a.exclude_dbs.read_text()))
        held |= train_sq2.load_drop_dbs(a.drop_dbs)
        buf, bstats = rl_data.load_buffer(a.buffer, held)
        ids = {e.id for e in exs}
        assert not any(e.db_id in held for e in exs if e.id in buf)
        bstats["training_questions_in_buffer"] = len(ids & set(buf))
        print(f"buffer: {bstats}", flush=True)
    meta = run_meta(a, {"questions": len(exs), "dbs": len({e.db_id for e in exs}),
                        "covered": sum(e.t is not None for e in exs), "filters": fstats, "serve_cands": dict(cstats),
                        "buffer": bstats})
    (a.out / "rl_args.json").write_text(json.dumps(meta, indent=1))

    if a.tiny:
        m = tiny_model(a.seed, grad_ckpt=a.grad_ckpt).to(a.device)
        build = lambda bb: LuthorSQ2(bb, d_proj=32, col_mentions=True, value_side=True)
    else:
        from eval_sq2 import build_model
        m, ra = real_model(a.run, a.device, a.quant, a.grad_ckpt)
        build = lambda bb: build_model(bb, ra)
    m.eval()

    if a.tiny and a.tiny_sft_steps:  # make the toy policy good enough to get mixed rewards
        opt0 = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3)
        cov = [e for e in exs if e.t is not None]
        by = collections.defaultdict(list)
        for e in cov:
            by[e.db_id].append(e)
        for s in range(a.tiny_sft_steps):
            db = rnd.choice(sorted(by))
            tr.train_db(m, cards[db], rnd.sample(by[db], min(8, len(by[db]))), a.budget_chars, rnd)
            opt0.step()
            opt0.zero_grad()
        print(f"tiny SFT warm-up: {a.tiny_sft_steps} steps", flush=True)

    ref = add_reference(m, build, a.tiny) if a.beta > 0 else None
    policy_mode(m, a.grad_ckpt)
    lora = [p for n, p in m.backbone.model.named_parameters() if p.requires_grad and ".default." in n] if not a.tiny \
        else [p for p in m.backbone.parameters() if p.requires_grad]
    heads = [p for n, p in m.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": heads, "lr": a.head_lr}], weight_decay=0.0)
    n_lora, n_heads = sum(p.numel() for p in lora), sum(p.numel() for p in heads)
    print(f"trainable: lora/backbone {n_lora:,} heads {n_heads:,}; ref {'on' if ref is not None else 'off'}; "
          f"grad-ckpt {'on' if getattr(m, 'ckpt_cache', False) else 'off'}", flush=True)

    rewarder = ExecReward(paths, a.exec_seconds, a.workers)
    ledger = train_sq2.OOMLedger(a.max_oom_skip, min_seen=a.oom_min_seen)
    stream = group_stream(exs, a.q_per_db, a.seed)
    n_groups = len(tr.db_groups(exs, a.q_per_db, a.seed))
    per_step = max(1, a.q_per_step // a.q_per_db)
    steps = a.timing_smoke or a.steps or max(1, math.ceil(a.epochs * n_groups / per_step))
    print(f"{steps} steps x {per_step} database groups of <= {a.q_per_db} questions", flush=True)
    startup_s = time.time() - t_start
    hist, buf_cache = [], {}
    for step in range(steps):
        log = collections.Counter()
        t0 = time.time()
        if str(a.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        ex0 = dict(rewarder.stats)
        chunk = [next(stream) for _ in range(per_step)]
        ledger.seen += sum(len(gq) for _, gq in chunk)
        dropped = False
        for gi, (db, group) in enumerate(chunk):
            try:
                run_rl_db(a, m, ref, cards[db], db, group, gold, rewarder, rnd,
                          a.seed * 100003 + (step * per_step + gi) * 1009, log, ledger, 1.0 / per_step, buf, buf_cache)
            except BackwardOOM as e:  # partial gradients from a half-finished backward: the whole step is unusable
                print(f"OOM in backward on {e}: step {step} dropped ({', '.join(d for d, _ in chunk)})", flush=True)
                opt.zero_grad(set_to_none=True)
                _empty_cache(a.device)
                ledger.dropped_steps += 1
                for d, x in chunk:
                    ledger.skip(d, len(x))
                dropped = True
                break
            if a.device == "mps":
                torch.mps.empty_cache()
        gn_l = gn_h = 0.0
        moved = 0.0
        if not dropped:
            gn_l = torch.nn.utils.clip_grad_norm_(lora, 1.0).item() if lora else 0.0
            gn_h = torch.nn.utils.clip_grad_norm_(heads, 1.0).item() if heads else 0.0
            if math.isfinite(gn_l) and math.isfinite(gn_h):
                before = [p.detach().clone() for p in heads[:3]]
                opt.step()
                moved = sum(float((p.detach() - b).abs().sum()) for p, b in zip(heads[:3], before))
            else:
                print(f"non-finite gradient at step {step}: update skipped", flush=True)
                log["nonfinite_grad"] += 1
        opt.zero_grad(set_to_none=True)
        q = max(log["q"], 1)
        nl = max(log["list_groups"], 1)
        rec = {"step": step, "q": log["q"], "mean_reward": round(log["reward_sum"] / max(log["plans"], 1), 4),
               "sample_acc": round(log["correct_sum"] / max(log["plans"], 1), 4),
               "groups_with_signal": log["groups_with_signal"], "plans_scored_frac": round(log["plans_scored"] / max(log["plans"], 1), 3),
               "distinct_sql_per_q": round(log["distinct_sql"] / q, 2),
               "draw_failed_frac": round(log["draw_failed"] / max(log["plans"], 1), 3),
               "trace_conflicts": log["trace_conflicts"],
               "pg": round(log["pg"], 4), "kl": round(log["kl"], 6),
               "list_groups": log["list_groups"], "list_loss": round(log["list_loss"] / nl, 4),
               "list_size": round(log["list_size"] / nl, 2), "kbest_oracle": round(log["list_oracle"] / q, 3),
               "kbest_first_right": round(log["list_first_right"] / q, 3),
               "sft_q": log["sft_q"], "distill_q": log["distill_q"],
               "buffer_q_right": log["buffer_q_right"], "gold_noisy": log["gold_noisy"],
               "oom_retries": log["oom_retries"], "oom_skipped": log["oom_skipped"], "dropped": dropped,
               "nonfinite": log["nonfinite_loss"] + log["nonfinite_grad"],
               "grad_norm_lora": round(gn_l, 4), "grad_norm_heads": round(gn_h, 4), "param_delta_heads3": moved,
               "s_per_q": {k[2:]: round(log[k] / q, 4) for k in PHASES},
               "exec_queries": int(rewarder.stats["queries"] - ex0.get("queries", 0)),
               "step_s": round(time.time() - t0, 2)}
        if str(a.device).startswith("cuda"):
            rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        (a.out / "rl_log.json").write_text(json.dumps(hist, indent=1))
    rewarder.close()
    (a.out / "oom.json").write_text(json.dumps(ledger.summary(), indent=1))
    if a.timing_smoke:
        rep = timing_report(hist, a, startup_s)
        (a.out / "timing.json").write_text(json.dumps(rep, indent=1))
        print(f"TIMING {json.dumps(rep)}", flush=True)
        print(f"PROJECTION {rep['s_per_question']:.3f} s/question -> {rep['gpu_h']['1_epoch']:.2f} GPU-h for 1 epoch, "
              f"{rep['gpu_h']['2_epoch']:.2f} GPU-h for 2 epochs over {a.project_questions} questions "
              f"(G={a.G}, startup {startup_s:.0f}s included)", flush=True)
        return hist
    if not a.tiny:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        (a.out / "best_lora").mkdir(exist_ok=True)
        save_file({k: v.contiguous() for k, v in get_peft_model_state_dict(m.backbone.model, adapter_name="default").items()},
                  str(a.out / "best_lora/adapter_model.safetensors"))
        torch.save(m.head_state(), a.out / "best_heads.pt")
        for f in ("args.json", "vocab.json"):
            (a.out / f).write_text((a.run / f).read_text())
        if a.heldout_eval:
            subprocess.run(["bash", str(HERE / "heldout_eval.sh"), str(a.out)], cwd=R, check=False)
    return hist


if __name__ == "__main__":
    main()
