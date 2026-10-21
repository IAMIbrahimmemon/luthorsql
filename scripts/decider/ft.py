"""LoRA fine-tune of Perplexity Decider v1.1 27B as the candidate selector (runs on one 80 GB GPU).

  python scripts/decider/ft.py --ckpt /workspace/decider --data /workspace/ddata --out /workspace/dout [--max-hours 3.5]

Uses Perplexity's own model class (autojev.model.DecisionModel, Apache-2.0, shipped in the checkpoint's source/src):
Qwen3.8-27B backbone (noncausal attention, last-token pooling) + a 5120 -> 255 readout over answer-code tokens,
cross-entropy over the options (autojev.loss.decision_loss). We add LoRA (r=16) on every Linear of the language
model and train the readout at a lower rate; the vision tower stays frozen.

Stages (each writes its outputs, so a re-launch resumes):
  1. zero-shot probabilities on val, DECISION, RESERVED  -> <out>/zs_<split>.jsonl
  2. one epoch over train.jsonl (stops early at --max-hours, keeping what it trained)
  3. temperature fitted on val (grid NLL), fine-tuned probabilities on val, DECISION, RESERVED -> <out>/ft_<split>.jsonl
  4. adapter + readout -> <out>/adapter/
RESERVED probabilities are written blind: scored only for a pre-registered claim.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch


def rows(p):
    return [json.loads(line) for line in open(p)]


def predict(model, data, out, temperature=1.0, bs=4):
    if Path(out).exists() and sum(1 for _ in open(out)) == len(data):
        print(f"skip {out} (done)", flush=True)
        return
    model.eval()
    with open(out, "w") as f, torch.inference_mode():
        for s in range(0, len(data), bs):
            chunk = data[s:s + bs]
            try:
                batch = model.prepare([{"state": r["state"], "question": r["question"]} for r in chunk], max_length=8192)
                probs = (model(batch) / temperature).softmax(-1).cpu().tolist()
            except (ValueError, torch.OutOfMemoryError) as e:
                torch.cuda.empty_cache()
                probs = [None] * len(chunk)
                print(f"predict skip {chunk[0]['id']}: {type(e).__name__}", flush=True)
            for r, p, n in zip(chunk, probs, [len(x["question"]["criteria"]) for x in chunk]):
                f.write(json.dumps({"id": r["id"], "order": r.get("order"), "probs": p[:n] if p else None}) + "\n")
    print(f"wrote {out}", flush=True)


def fit_temperature(model, val):
    logits, targets = [], []
    model.eval()
    with torch.inference_mode():
        for r in val:
            b = model.prepare([{"state": r["state"], "question": r["question"]}], max_length=8192)
            n = len(r["target"])
            logits.append(model(b)[0, :n].float().cpu())
            targets.append(torch.tensor(r["target"]) / max(sum(r["target"]), 1e-9))
    best = (1.0, float("inf"))
    for t in [0.5 + 0.05 * i for i in range(71)]:
        nll = sum(-(tg * torch.log_softmax(lg / t, -1)).sum().item() for lg, tg in zip(logits, targets)) / len(val)
        best = min(best, (t, nll), key=lambda x: x[1])
    acc = sum(int(lg.argmax()) in [i for i, v in enumerate(tg) if v > 0] for lg, tg in zip(logits, targets)) / len(val)
    print(f"temperature {best[0]:.2f} (val NLL {best[1]:.4f}), val accuracy {acc:.4f}", flush=True)
    return best[0], acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--readout-lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-hours", type=float, default=3.5)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.ckpt) / "source/src"))
    from autojev.loss import decision_loss
    from autojev.model import MAX_OPTIONS, DecisionModel
    from peft import LoraConfig, get_peft_model

    D, O = Path(a.data), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    t0 = time.time()
    model = DecisionModel(checkpoint=a.ckpt, train=False, device="cuda")
    print(f"loaded in {time.time() - t0:.0f}s; temperature {model.temperature}; "
          f"mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    val, dec, res = rows(D / "val.jsonl"), rows(D / "eval_decision.jsonl"), rows(D / "eval_reserved.jsonl")

    # 1. zero-shot
    predict(model, val, O / "zs_val.jsonl", model.temperature)
    predict(model, dec, O / "zs_decision.jsonl", model.temperature)
    predict(model, res, O / "zs_reserved.jsonl", model.temperature)

    # 2. LoRA on every Linear of the language model (not the vision tower), readout trained at a lower rate
    lm = model.backbone.language_model
    names = sorted({n.split(".")[-1] for n, m in lm.named_modules() if isinstance(m, torch.nn.Linear)})
    print("LoRA targets:", names, flush=True)
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
                     target_modules=r".*language_model.*\.(" + "|".join(names) + r")$")
    model.backbone = get_peft_model(model.backbone, cfg)
    model.backbone.print_trainable_parameters()
    model.readout.requires_grad_(True)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.backbone.enable_input_require_grads()
    lora = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": model.readout.parameters(), "lr": a.readout_lr}],
                            weight_decay=0.0)
    train = rows(D / "train.jsonl")
    random.Random(a.seed).shuffle(train)
    if a.max_train:
        train = train[:a.max_train]
    # drop rows over the token limit before training: prepare() rejects them before any forward pass, and skipping
    # them mid-step used to zero the gradients already accumulated for that step
    fits = []
    for r in train:
        try:
            model.prepare([{"state": r["state"], "question": r["question"]}], max_length=8192)
            fits.append(r)
        except ValueError:
            pass
    print(f"{len(train) - len(fits)} of {len(train)} train rows over 8192 tokens, dropped", flush=True)
    train = fits
    steps = math.ceil(len(train) / a.accum)
    warm = max(1, int(a.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up, then linear decay to 0
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    model.train()
    t1, run, seen, skipped, step = time.time(), 0.0, 0, 0, 0
    log = open(O / "train_log.jsonl", "a")
    for n, r in enumerate(train, 1):
        try:
            b = model.prepare([{"state": r["state"], "question": r["question"]}], max_length=8192)
            tgt = torch.zeros(1, MAX_OPTIONS, device="cuda")
            k = len(r["target"])
            tgt[0, :k] = torch.tensor(r["target"], device="cuda") / max(sum(r["target"]), 1e-9)
            loss = decision_loss(model(b), tgt).total / a.accum
            loss.backward()
            run += loss.item() * a.accum
            seen += 1
        except (ValueError, torch.OutOfMemoryError) as e:
            if isinstance(e, torch.OutOfMemoryError):  # a failed backward may leave partial gradients: drop the step
                opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped += 1
            print(f"train skip {r['id']}: {type(e).__name__}", flush=True)
        if n % a.accum == 0 or n == len(train):
            torch.nn.utils.clip_grad_norm_(lora + list(model.readout.parameters()), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                el = (time.time() - t1) / 3600
                rec = {"step": step, "of": steps, "loss": round(run / max(seen, 1), 4), "seen": seen, "skipped": skipped,
                       "hours": round(el, 3), "eta_h": round(el / step * (steps - step), 2),
                       "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                print("STEP", json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run, seen = 0.0, 0
            if (time.time() - t1) / 3600 > a.max_hours:
                print(f"time limit at step {step}/{steps}", flush=True)
                break
    print(f"TRAINED {step} steps in {(time.time() - t1) / 3600:.2f} h, skipped {skipped}", flush=True)

    # 3. temperature on val, then fine-tuned probabilities
    temp, vacc = fit_temperature(model, val)
    json.dump({"temperature": temp, "val_accuracy": vacc, "steps": step, "of": steps}, open(O / "ft_meta.json", "w"))
    for split, data in (("val", val), ("decision", dec), ("reserved", res)):
        predict(model, data, O / f"ft_{split}.jsonl", temp)

    # 4. adapter + readout
    model.backbone.save_pretrained(str(O / "adapter"))
    torch.save(model.readout.state_dict(), O / "adapter" / "readout.pt")
    print("DECIDER_FT_DONE", flush=True)


if __name__ == "__main__":
    main()
