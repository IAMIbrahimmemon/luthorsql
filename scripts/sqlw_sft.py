"""Supervised LoRA training of the SQL writer (round 4) on data/sqlw/sft.jsonl. Runs on the GPU pod.

  python scripts/sqlw_sft.py --data /root/d/sqlw/sft.jsonl --out runs/sqlw_sft --hours 4

Loss on the SQL tokens only. Batches are length-bucketed up to a token budget. The learning rate follows a
cosine over max(step progress, wall-clock progress), so a time limit still ends at a low learning rate.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch


def rss_gb() -> float:
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS"):
                return int(l.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def encode(tok, rows, max_len):
    out, dropped = [], 0
    for r in rows:
        text = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False)
        p = tok(text, add_special_tokens=False)["input_ids"]
        c = tok(r["sql"].strip() + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p) + len(c) > max_len:
            dropped += 1
            continue
        out.append((list(p), c))
    return out, dropped


def batches(items, tokens_per_batch, rnd):
    idx = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    out, cur, longest = [], [], 0
    for i in idx:
        n = len(items[i][0]) + len(items[i][1])
        if cur and max(longest, n) * (len(cur) + 1) > tokens_per_batch:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        out.append(cur)
    rnd.shuffle(out)
    return out


def collate(items, ids, pad):
    T = max(len(items[i][0]) + len(items[i][1]) for i in ids)
    x = torch.full((len(ids), T), pad, dtype=torch.long)
    y = torch.full((len(ids), T), -100, dtype=torch.long)
    m = torch.zeros((len(ids), T), dtype=torch.long)
    for j, i in enumerate(ids):
        p, c = items[i]
        s = p + c
        x[j, :len(s)] = torch.tensor(s)
        y[j, len(p):len(s)] = torch.tensor(c)
        m[j, :len(s)] = 1
    return x, y, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--tokens-per-batch", type=int, default=24576)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rnd = random.Random(0)
    torch.manual_seed(0)
    rows = [json.loads(l) for l in open(a.data)]
    if a.limit:
        rows = rows[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.base)
    items, dropped = encode(tok, rows, a.max_len)
    ntok = sum(len(p) + len(c) for p, c in items)
    print(f"rows {len(items)} (dropped {dropped} over {a.max_len} tokens), {ntok / 1e6:.1f}M tokens", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    per_epoch = len(batches(items, a.tokens_per_batch, random.Random(0)))
    total = a.max_steps or max(1, int(per_epoch * a.epochs / a.accum))
    print(f"{per_epoch} micro-batches per epoch, {total} optimizer steps planned, {a.hours} h limit", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0, step, micro, ep, done = time.time(), 0, 0, 0, False
    loss_acc, n_acc, tok_acc, oom = 0.0, 0, 0, 0
    model.train()
    last_save = time.time()
    while not done:
        for ids in batches(items, a.tokens_per_batch, rnd):
            x, y, m = collate(items, ids, pad)
            try:
                out = model(input_ids=x.cuda(), attention_mask=m.cuda(), labels=y.cuda())
                (out.loss / a.accum).backward()
                loss_acc += float(out.loss)
                n_acc += 1
                tok_acc += int(m.sum())
            except torch.OutOfMemoryError:
                oom += 1
                print(f"OOM on a batch of {len(ids)} x {x.shape[1]}: skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            micro += 1
            if micro % a.accum:
                continue
            prog = max(step / total, (time.time() - t0) / (a.hours * 3600))
            lr = a.lr * min(1.0, (step + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                el = time.time() - t0
                print(f"step {step}/{total} ep {ep} loss {loss_acc / max(n_acc, 1):.4f} lr {lr:.2e} "
                      f"{tok_acc / el:.0f} tok/s elapsed {el / 3600:.2f}h oom {oom} "
                      f"live {torch.cuda.memory_allocated() / 2**30:.1f}GB peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB "
                      f"rss {rss_gb():.1f}GB", flush=True)
                loss_acc, n_acc = 0.0, 0
            if time.time() - last_save > 1800:
                model.save_pretrained(a.out / "lora")
                last_save = time.time()
            if step >= total or time.time() - t0 > a.hours * 3600:
                done = True
                break
        ep += 1
        if ep >= math.ceil(a.epochs):
            done = True
    model.save_pretrained(a.out / "lora")
    tok.save_pretrained(a.out / "lora")
    print(f"DONE_SFT {step} steps in {(time.time() - t0) / 3600:.2f} h, oom {oom}", flush=True)


if __name__ == "__main__":
    main()
