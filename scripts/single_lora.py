"""Single-LoRA System 1 (docs/paper_notes/single_lora_2026-10-16.md): ONE LoRA on OmniSQL-7B both writes the k=32
candidate plans (slot heads) and scores them (rerank head), for BIRD's Single Trained Model track (BIRD asked for one
adapter trained on a mixture of generation and selection data instead of R1's slot LoRA + VALHARD's rerank LoRA).

Starts from R1 (its adapter and slot heads, --init). Every optimizer step adds, on the same adapter:
  - VALHARD's validator loss: rerank_sq2.py train v2's db_step_v2 (group loss, label smoothing, gradient cache) over
    the hard-negative lists (--train-file), the same lists and batch shape as VALHARD;
  - R1's generator loss (replay): train_sq2.py's step_grads over R1's own training mix (R1's args.json: plans,
    bird_train x3, held-out dev DBs excluded), --gen-weight times.
The rerank head is new (zeros, as VALHARD's). Cosine schedule with warmup over VALHARD's 2 epochs. No checkpoint
selection: the last step is the model (val pick accuracy and the generator's held-out greedy EX are logged per epoch
for diagnosis only).

The out dir is both a generator run (best_lora/, best_heads.pt, args.json, vocab.json -> rerank_sq2.py gen --run)
and a validator dir (rerank_head.pt, settings.json with shared_adapter -> rerank_sq2.py score --rr).

  python scripts/single_lora.py --init /root/d/r1 --train-file /root/d/valhard/r1/train_lists.jsonl \\
      --val-file /root/d/valhard/r1/val_lists.jsonl --cards /root/d/sys1/bird_train --out /workspace/single/run
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_sq2  # noqa: E402
import rerank_sq2 as rh  # noqa: E402
import train_sq2 as ts  # noqa: E402
from luthor.sq.model import Backbone  # noqa: E402
from luthor.sq2.rerank import Reranker  # noqa: E402
from luthor.sq2.train import db_groups, group_seed  # noqa: E402


def r1_plans(ra: dict) -> list[Path]:
    """args.json stores --plans as str(list of PosixPath)."""
    return [Path(p) for p in re.findall(r"PosixPath\('([^']+)'\)", ra["plans"])]


def gen_data(ra: dict, plans: list[Path], dev_dbs: Path, limit: int = 0, drop: frozenset = frozenset()):
    """R1's training rows exactly as train_sq2.main builds them (no --extra-train / --drop-dbs / --bird-filter in R1):
    held-out dev DBs split off, --repeat applied. Returns (cards, train, dev, gold)."""
    ts.VOCAB_ARGS = ts.load_vocab(Path(ra["vocab"])) if ra.get("vocab", "None") != "None" else None
    ts.CAND_ORDER = ra.get("cand_order", "v1")
    ts._init_worker(ts.VOCAB_ARGS)
    cards, exs, gold = ts.load_sets(plans, limit=limit)
    held = set(json.loads(dev_dbs.read_text()))
    train = [e for e in exs if e.db_id not in held and e.db_id not in drop]
    dev = [e for e in exs if e.db_id in held]
    if drop:
        print(f"replay drops dbs {sorted(drop)}: {sum(e.db_id in drop for e in exs)} rows", flush=True)
    for spec in ast.literal_eval(ra.get("repeat", "[]")):
        name, n = spec.split("=")
        d = next(d for d in plans if d.name == name)
        dbs = {p.stem for p in (d / "cards").glob("*.json")}
        extra = [e for e in train if e.db_id in dbs and gold.get(e.id, {}).get("source") != "bird_mut"]
        train += extra * (int(n) - 1)
        print(f"repeat {name} x{n}: +{len(extra) * (int(n) - 1)} rows", flush=True)
    random.Random(0).shuffle(dev)  # train_sq2's dev order (Random(0)), so eval EX is comparable with R1's log
    print(f"generator replay: train {len(train)} rows / {len({e.db_id for e in train})} dbs, held-out dev {len(dev)}",
          flush=True)
    return cards, train, dev, gold


def gen_chunks(train: list, q_per_db: int, dbs_per_step: int, seed: int):
    """train_sq2's step order, epoch after epoch."""
    ep = 0
    while True:
        groups = db_groups(train, q_per_db, seed=group_seed(seed, ep))
        for i in range(0, len(groups), dbs_per_step):
            yield ep, groups[i:i + dbs_per_step]
        ep += 1


def val_args(a) -> argparse.Namespace:
    """The fields rerank_sq2.db_step_v2 reads, set as VALHARD's run (data/bird_submit/weights/valhard_rr/settings.json)."""
    return argparse.Namespace(loss="group", label_smooth=a.label_smooth, tok_budget=a.tok_budget,
                              budget_chars=a.budget_chars, joint_results=False, seed=a.seed)


def save(out: Path, model, rr, ra: dict, init: Path, settings: dict):
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.model.save_pretrained(out / "best_lora")  # the one adapter ("default")
    torch.save(model.head_state(), out / "best_heads.pt")
    torch.save(rr.head_state(), out / "rerank_head.pt")
    (out / "args.json").write_text(json.dumps({**ra, "out": str(out), "init": str(init)}, indent=1))
    if (init / "vocab.json").exists():
        shutil.copy(init / "vocab.json", out / "vocab.json")
    (out / "settings.json").write_text(json.dumps(settings, indent=1))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init", type=Path, required=True, help="R1 run dir: best_lora/, best_heads.pt, args.json, vocab.json")
    ap.add_argument("--plans", nargs="*", type=Path, default=None, help="generator plan dirs (default: R1's args.json)")
    ap.add_argument("--dev-dbs", type=Path, default=None, help="held-out DBs (default: R1's args.json)")
    ap.add_argument("--db-roots", nargs="*", default=[], help="sqlite roots for the generator's held-out eval")
    ap.add_argument("--train-file", type=Path, required=True, help="VALHARD train lists")
    ap.add_argument("--val-file", type=Path, help="VALHARD val lists (pick accuracy per epoch, diagnosis only)")
    ap.add_argument("--cards", nargs="*", type=Path, default=[], help="validator schema-card plan dirs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=None, help="default: R1's args.json")
    ap.add_argument("--epochs", type=float, default=2.0, help="over the validator lists (VALHARD: 2)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4, help="the shared LoRA")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="the new rerank head")
    ap.add_argument("--gen-head-lr", type=float, default=1e-4, help="R1's slot heads (already trained)")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--gen-weight", type=float, default=1.0, help="generator loss weight per step (0 = no replay)")
    ap.add_argument("--q-per-db", type=int, default=4, help="validator (VALHARD: 4)")
    ap.add_argument("--dbs-per-step", type=int, default=4, help="validator (VALHARD: 4)")
    ap.add_argument("--tok-budget", type=int, default=120000)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--budget-chars", type=int, default=9000)
    ap.add_argument("--gen-eval-limit", type=int, default=1500, help="held-out greedy EX questions (R1 logged 1,455)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=0, help="safety stop (saves first)")
    ap.add_argument("--limit", type=int, default=0, help="plan rows per dir (tests)")
    ap.add_argument("--max-oom-skip", type=float, default=0.002)
    ap.add_argument("--rerank-head", choices=["linear", "mlp"], default="linear",
                    help="scoring head on the shared adapter: one linear layer (runs 1-2) or a 2-layer MLP (option 3)")
    ap.add_argument("--empty-cache", action="store_true", help="torch.cuda.empty_cache() between the generator and "
                    "validator parts of a step (less fragmentation, fewer validator OOM retries)")
    ap.add_argument("--drop-dbs", nargs="*", default=[], help="training DBs left out of the generator replay (e.g. "
                    "works_cycles: its schema does not fit one question per pass; R1 skipped it for memory too)")
    return ap


def main():
    a = parser().parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)  # as train_sq2 (non-finite gradients on long prefixes)
    ra = eval_sq2.run_args(a.init)
    if not ra:
        raise SystemExit(f"{a.init}/args.json missing")
    base = a.base or ra.get("base", "seeklhy/OmniSQL-7B")
    plans = a.plans or r1_plans(ra)
    dev_dbs = a.dev_dbs or Path(ra["dev_dbs"])
    g_q, g_dbs = int(ra.get("q_per_db", 8)), int(ra.get("dbs_per_step", 4))
    budget_chars = int(ra.get("budget_chars", 9000))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "single_args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))

    # ---- data
    cards, gtrain, gdev, gold = gen_data(ra, plans, dev_dbs, a.limit, frozenset(a.drop_dbs))
    db_paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            db_paths.setdefault(p.stem, p)
    card_paths = rh.load_cards(a.cards)
    recs = [json.loads(l) for l in open(a.train_file)]
    recs = [r for r in recs if r["db_id"] in card_paths]
    usable = [r for r in recs if rh.usable_list(r)]
    wbar = sum(float(r.get("weight", 1.0)) for r in usable) / max(len(usable), 1)
    by_db = collections.defaultdict(list)
    for r in usable:
        by_db[r["db_id"]].append(r)
    vl = [json.loads(l) for l in open(a.val_file)] if a.val_file else []
    vl = [r for r in vl if r["db_id"] in card_paths]
    print(f"validator lists: {len(usable)} of {len(recs)} usable, {len(by_db)} dbs; val {len(vl)}", flush=True)

    # ---- model: R1's slot model, then the rerank head on the same adapter
    bb = Backbone.from_pretrained(base, ra.get("subfolder") or None, lora_r=int(ra.get("lora_r", 32)), grad_ckpt=False)
    model = eval_sq2.build_model(bb, ra).cuda()
    model.backbone.model.to(torch.bfloat16)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model.backbone.model, load_file(str(a.init / "best_lora" / "adapter_model.safetensors")))
    model.load_state_dict(torch.load(a.init / "best_heads.pt", map_location="cpu"), strict=False)
    rr = Reranker(model.backbone, adapter="default", head=a.rerank_head).cuda()
    rr.activate(train=True)
    lora = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    gheads = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    score = list(rr.score.parameters())
    print(f"one adapter: {sum(p.numel() for p in lora) / 1e6:.1f}M LoRA params, slot heads "
          f"{sum(p.numel() for p in gheads) / 1e6:.1f}M, rerank head {sum(p.numel() for p in score)}", flush=True)
    if not lora:
        raise SystemExit("no trainable LoRA parameters")
    params = lora + gheads + score
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": gheads, "lr": a.gen_head_lr},
                             {"params": score, "lr": a.head_lr}], weight_decay=0.01)
    per_epoch = math.ceil(sum(math.ceil(len(rs) / a.q_per_db) for rs in by_db.values()) / a.dbs_per_step)
    total = a.max_steps or max(1, int(round(per_epoch * a.epochs)))
    sa = argparse.Namespace(sched="cosine", warmup=a.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: rh.lr_factor(sa, s, total))
    print(f"{per_epoch} validator steps per epoch, {total} steps; generator {g_dbs} dbs x {g_q} q per step, "
          f"weight {a.gen_weight}", flush=True)

    rh.PREVIEW = False
    rh.JOINT.update(on=False, orders=2)
    va = val_args(a)
    hist = []

    def evaluate(tag: str, step: int):
        t = time.time()
        m = {"tag": tag, "step": step}
        if db_paths and gdev:
            m["gen_heldout_ex"] = ts.evaluate(model, gdev, cards, gold, db_paths, limit=a.gen_eval_limit)
        if vl and step > 0:
            rh.score_file(rr, vl, card_paths, a.budget_chars)
            m["val_pick_acc"] = rh.pick_acc(vl)
        rr.activate(train=True)
        model.train()
        rr.train()
        m["secs"] = round(time.time() - t, 1)
        hist.append(m)
        (a.out / "history.json").write_text(json.dumps(hist, indent=1))
        print(f"EVAL {json.dumps(m)}", flush=True)

    settings = {"shared_adapter": True, "trainer": "single_lora", "rerank_head": a.rerank_head, "result_preview": False, "joint_results": False,
                "loss": "group", "label_smooth": a.label_smooth, "init": str(a.init), "train_file": str(a.train_file),
                "lr": a.lr, "head_lr": a.head_lr, "gen_head_lr": a.gen_head_lr, "gen_weight": a.gen_weight,
                "epochs": a.epochs, "steps": total, "seed": a.seed, "base": base, "lora_r": int(ra.get("lora_r", 32))}
    evaluate("init (R1)", 0)
    rnd = random.Random(a.seed)
    grnd = random.Random(f"gen:{a.seed}")
    gen_it = gen_chunks(gtrain, g_q, g_dbs, a.seed)
    ledger = ts.OOMLedger(a.max_oom_skip)
    stats = rh.new_stats()
    card_cache = {}
    step, ep, t0 = 0, 0, time.time()
    acc = collections.defaultdict(float)
    model.train()
    rr.train()
    while step < total:
        chunks = []
        for db, rs in by_db.items():
            rs = rs[:]
            rnd.shuffle(rs)
            chunks += [(db, rs[i:i + a.q_per_db]) for i in range(0, len(rs), a.q_per_db)]
        rnd.shuffle(chunks)
        for i in range(0, len(chunks), a.dbs_per_step):
            batch = chunks[i:i + a.dbs_per_step]
            gl, gparts, gok = 0.0, {}, False
            if a.gen_weight > 0:  # generator first: an OOM in its backward zeroes only its own (=all so far) grads
                gep, gchunk = next(gen_it)
                ledger.seen += sum(len(es) for _, es in gchunk)
                dropped, gl, gparts = ts.step_grads(model, gchunk, gchunk, cards, budget_chars, grnd, ledger)
                gok = not dropped
                if a.empty_cache:  # the generator's cached blocks back to the allocator before the validator's lists
                    torch.cuda.empty_cache()
                if gok and a.gen_weight != 1.0:
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(a.gen_weight)
            nq = sum(len(rs) for _, rs in batch)
            vloss, vn = 0.0, 0
            for db, rs in batch:
                l, n = rh.db_step_v2(rr, rh.card_of(card_cache, card_paths, db), rs, va, 1.0 / nq, wbar, stats)
                vloss += l
                vn += n
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gn):
                print(f"non-finite gradient at step {step}: update skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            acc["val"] += vloss / max(vn, 1)
            acc["gen"] += gl if gok else 0.0
            acc["gen_n"] += float(gok)
            if step % 25 == 0:
                el = time.time() - t0
                print(f"step {step}/{total} ep {ep} val_loss {acc['val'] / 25:.3f} gen_loss "
                      f"{acc['gen'] / max(acc['gen_n'], 1):.3f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{el / step:.2f}s/step gradcache {stats['gradcache']} oom {stats['oom']} "
                      f"skipped {sum(stats['skipped'].values())} gen_oom_skipped {ledger.skipped}/{ledger.seen} "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.0f}GB", flush=True)
                acc.clear()
            if a.hours and time.time() - t0 > a.hours * 3600:
                print(f"time limit at step {step}/{total}: saving and stopping", flush=True)
                total = step
            if step >= total:
                break
        ep += 1
        if step < total or ep == math.ceil(a.epochs):
            evaluate(f"epoch {ep}", step)
    if not hist or hist[-1]["step"] != step:
        evaluate("final", step)
    settings["steps_done"] = step
    settings["hours"] = round((time.time() - t0) / 3600, 3)
    save(a.out, model, rr, ra, a.init, settings)
    print(f"trained {step} steps in {(time.time() - t0) / 3600:.2f} h; validator one pass {stats['direct']}, "
          f"gradient cache {stats['gradcache']}, oom retries {stats['oom']}", flush=True)
    print(f"SKIPPED_FOR_MEMORY validator {sum(stats['skipped'].values())} {json.dumps(dict(stats['skipped']))}; "
          f"generator {ledger.skipped}/{ledger.seen}", flush=True)
    print(f"SINGLE_LORA_SAVED {a.out}", flush=True)


if __name__ == "__main__":
    main()
