"""Train LuthorSQ (run on the GPU pod, after prepare_plans.py has built the plan sets).

  python scripts/train_sq.py --plans data/plans/bird_train data/plans/spider_train data/plans/synsql_train \
      --out runs/luthor_sq --epochs 1 --real-oversample 2

Model selection uses a dev slice of TRAINING databases (--dev-frac of the db_ids, never BIRD
dev / Spider dev, which stay sealed). Metric: execution accuracy of compiled plans on those
dev databases (abstaining on an in-grammar question counts as wrong).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luthor.sq.data import batches, load_examples, make_db_batch  # noqa: E402
from luthor.sq.model import Backbone, LuthorSQ  # noqa: E402
from luthor.sq.plan import compile_plan  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq.train import TextCache, decode, forward_db, train_step  # noqa: E402


def _load_part(args):
    plans_path, lines, card_dir = args
    import tempfile
    cards = {}
    need = {json.loads(l)["db_id"] for l in lines}
    for db in need:
        p = Path(card_dir) / f"{db}.json"
        if p.exists():
            cards[db] = SchemaCard.load(p)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.writelines(lines)
    return load_examples(Path(f.name), cards)


def load_sets(dirs: list[Path], workers: int = 14):
    """Cards, examples (candidates rebuilt in parallel) and gold rows; cached as a pickle
    next to the plans so later stages load in seconds."""
    import pickle
    from multiprocessing import Pool

    cache = dirs[0] / "examples.cache.pkl"
    if len(dirs) == 1 and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    cards, exs, gold = {}, [], {}
    for d in dirs:
        for p in (d / "cards").glob("*.json"):
            c = SchemaCard.load(p)
            cards[c.db_id] = c
        lines = open(d / "plans.jsonl").readlines()
        for line in lines:
            r = json.loads(line)
            gold[r["id"]] = r
        by_db: dict[str, list[str]] = {}
        for line in lines:  # keep a database's rows (and twins) in one chunk
            by_db.setdefault(json.loads(line)["db_id"], []).append(line)
        groups, cur = [], []
        for db_lines in by_db.values():
            cur += db_lines
            if len(cur) > 4000:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        with Pool(workers) as pool:
            for part in pool.imap(_load_part, [(d / "plans.jsonl", g, d / "cards") for g in groups]):
                exs += part
    if len(dirs) == 1:
        with open(cache, "wb") as f:
            pickle.dump((cards, exs, gold), f)
    return cards, exs, gold


@torch.no_grad()
def evaluate(model, exs, cards, gold, db_paths: dict[str, Path], limit: int = 600, curve: list | None = None) -> dict:
    """Execution accuracy on held-out TRAINING databases. With `curve`, also records
    (gold_unsupported, p_unsupported, correct_if_answered) per question for calibration."""
    model.eval()
    exs = [e for e in exs if not e.twin_of][:limit]
    by_db: dict[str, list] = {}
    for e in exs:
        by_db.setdefault(e.db_id, []).append(e)
    right = total = abstain_ok = n_unsup = 0
    for db, group in by_db.items():
        con = sqlite3.connect(f"file:{db_paths[db]}?mode=ro", uri=True) if db in db_paths else None
        if con is not None:  # a bad plan (e.g. an accidental cross join) must not hang evaluation
            _deadline = [0.0]
            con.set_progress_handler(lambda: 1 if time.time() > _deadline[0] else 0, 10_000)
            _exec = con.execute
            class _TimedCon:
                def execute(self, sql):
                    _deadline[0] = time.time() + 5.0
                    return _exec(sql)
            con = _TimedCon()
        O = None
        for i in range(0, len(group), 8):
            b = make_db_batch(cards[db], group[i:i + 8], all_options=True)
            if O is None:
                O = model.encode_texts(b.option_texts)
            outs, _ = forward_db(model, b, option_vecs=O)
            for out, e in outs:
                p_unsup = float(torch.softmax(out.choice["outcome"], -1)[1])
                plan = decode(model, out, b.option_ids, e.cands)
                forced = decode(model, out, b.option_ids, e.cands, unsupported_threshold=1.01) if curve is not None else None
                row = gold[e.id]
                if curve is not None:
                    ok = False
                    if row["outcome"] == "plan" and forced is not None and con is not None:
                        try:
                            got = con.execute(compile_plan(forced, cards[db].graph())).fetchmany(100_000)
                            ref = con.execute(row["compiled_sql"]).fetchmany(100_000)
                            ok = set(map(tuple, got)) == set(map(tuple, ref))
                        except Exception:
                            pass
                    curve.append((row["outcome"] == "unsupported", p_unsup, ok))
                if row["outcome"] == "unsupported":
                    n_unsup += 1
                    abstain_ok += plan is None
                    continue
                total += 1
                if plan is None or con is None:
                    continue
                try:
                    got = con.execute(compile_plan(plan, cards[db].graph())).fetchmany(100_000)
                    ref = con.execute(row["compiled_sql"]).fetchmany(100_000)
                    right += set(map(tuple, got)) == set(map(tuple, ref))
                except Exception:
                    pass
    model.train()
    return {"ex": right / max(total, 1), "n": total, "abstain_acc": abstain_ok / max(n_unsup, 1)}


def calibrate(curve: list, min_abstain: float = 0.85) -> dict:
    """Sweep the abstain threshold on held-out training databases. Objective: overall accuracy
    (answered right on plan questions + abstained on unsupported ones), subject to abstention
    accuracy >= min_abstain. Returns the chosen threshold and the full curve."""
    pts = []
    for t in [i / 100 for i in range(5, 100, 5)]:
        right = sum((not u and p < t and ok) or (u and p >= t) for u, p, ok in curve)
        n_u = sum(u for u, _, _ in curve)
        abst = sum(u and p >= t for u, p, _ in curve) / max(n_u, 1)
        answered = sum(not u and p < t for u, p, _ in curve)
        prec = sum(not u and p < t and ok for u, p, ok in curve) / max(answered, 1)
        pts.append({"threshold": t, "accuracy": right / max(len(curve), 1), "abstain_acc": abst, "answer_precision": prec})
    ok = [p for p in pts if p["abstain_acc"] >= min_abstain] or pts
    best = max(ok, key=lambda p: p["accuracy"])
    return {"unsupported_threshold": best["threshold"], "chosen": best, "curve": pts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", type=Path, required=True)
    ap.add_argument("--db-roots", nargs="*", default=[], help="db_id=path.sqlite files, or folders searched recursively")
    ap.add_argument("--base", default="AlexWortega/openjev")
    ap.add_argument("--subfolder", default="qwen3.5-4b-nli-v5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-opts", type=int, default=48)
    ap.add_argument("--opt-max-len", type=int, default=64)
    ap.add_argument("--option-grad-frac", type=float, default=0.6,
                    help="share of steps that encode options/values WITH gradient; later steps use a cached no-grad encoding")
    ap.add_argument("--refresh-every", type=int, default=400, help="cache refresh interval (steps) in the no-grad phase")
    ap.add_argument("--dev-dbs", type=Path, help="json list of held-out TRAINING db_ids (their SQLite files must be in --db-roots)")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="faster, uses more GPU memory")
    ap.add_argument("--fused", action="store_true", help="one encode + one backward per batch (needs the memory)")
    ap.add_argument("--partial", action="store_true",
                    help="grad phase: only gold options + hard negatives get gradient; values always cached")
    ap.add_argument("--speed-test", type=int, default=0, help="time N steps with option gradient and N without, then exit")
    ap.add_argument("--calibrate-only", type=Path, help="load a run's best_lora + best_heads.pt and only calibrate")
    ap.add_argument("--limit-train", type=int, default=0, help="use only N training rows (the 2B scaling check)")
    ap.add_argument("--dbs-per-batch", type=int, default=4)
    ap.add_argument("--q-per-db", type=int, default=4)
    ap.add_argument("--dev-frac", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke runs")
    ap.add_argument("--unsup-keep", type=float, default=0.3, help="share of unsupported rows kept")
    a = ap.parse_args()
    torch.manual_seed(0)
    rnd = random.Random(0)

    cards, exs, gold = load_sets(a.plans)
    exs = [e for e in exs if e.target.choice["outcome"] == 0 or rnd.random() < a.unsup_keep]
    dbs = sorted({e.db_id for e in exs})
    if a.dev_dbs:
        dev_dbs = set(json.loads(a.dev_dbs.read_text()))
    else:
        rnd.shuffle(dbs)
        dev_dbs = set(dbs[: max(1, int(len(dbs) * a.dev_frac))])
    train = [e for e in exs if e.db_id not in dev_dbs]
    if a.limit_train:
        rnd.shuffle(train)
        train = train[: a.limit_train]
    dev = [e for e in exs if e.db_id in dev_dbs]
    db_paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            db_paths[p.stem] = p
    print(f"train {len(train)} examples / {len(dbs) - len(dev_dbs)} dbs, dev {len(dev)} / {len(dev_dbs)} dbs", flush=True)

    backbone = Backbone.from_pretrained(a.base, a.subfolder, lora_r=a.lora_r, grad_ckpt=not a.no_grad_ckpt)
    model = LuthorSQ(backbone).cuda()
    model.opt_max_len = a.opt_max_len
    model.backbone.model.to(torch.bfloat16)
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    lora = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": heads, "lr": a.head_lr}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / (a.dbs_per_batch * a.q_per_db))
    total = a.max_steps or int(steps_per_epoch * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    if a.calibrate_only:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model.backbone.model, load_file(str(a.calibrate_only / "best_lora" / "adapter_model.safetensors")))
        model.load_state_dict(torch.load(a.calibrate_only / "best_heads.pt"), strict=False)
        curve: list = []
        m = evaluate(model, dev, cards, gold, db_paths, limit=2000, curve=curve)
        cal = calibrate(curve)
        (a.calibrate_only / "serving.json").write_text(json.dumps({k: v for k, v in cal.items() if k != "curve"}, indent=1))
        (a.calibrate_only / "calibration.json").write_text(json.dumps({**cal, "dev_at_default": m}, indent=1))
        print("calibration:", cal["chosen"], "dev:", m, flush=True)
        return
    cache = TextCache(a.refresh_every)
    model.train()
    if a.speed_test:
        it = batches(train, cards, a.dbs_per_batch, a.q_per_db, a.max_opts, seed=0)
        for mode in (("partial", "cached") if a.partial else ("grad", "cached")):
            torch.cuda.synchronize(); ts = []
            for k in range(a.speed_test):
                b = next(it); t1 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = train_step(model, b, cache=cache if mode in ("cached", "partial") else None, step=10_000 + k,
                                         backward=True, fused=a.fused, partial=mode == "partial")
                opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
                ts.append(time.time() - t1)
            ts = ts[3:] or ts
            from luthor.sq import train as _tr
            if _tr.PROFILE is not None:
                tot = sum(_tr.PROFILE.values()) or 1
                print("PROFILE " + mode + ": " + ", ".join(f"{k} {v / len(ts + [0, 0, 0]):.2f}s ({100 * v / tot:.0f}%)"
                                                           for k, v in sorted(_tr.PROFILE.items(), key=lambda kv: -kv[1])), flush=True)
                _tr.PROFILE.clear()
            print(f"SPEED {mode}: mean {sum(ts) / len(ts):.2f} s/step, min {min(ts):.2f}, max {max(ts):.2f}, "
                  f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        return
    grad_steps = int(total * a.option_grad_frac)
    model.train()  # HF loads in eval mode; gradient checkpointing only runs in train mode
    step, best, t0, epoch = 0, -1.0, time.time(), 0
    best_state: dict = {}
    while step < total:
        for batch in batches(train, cards, a.dbs_per_batch, a.q_per_db, a.max_opts, seed=epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if a.partial:
                    loss, parts = train_step(model, batch, cache=cache, step=step, backward=True,
                                             partial=step < grad_steps)
                else:
                    loss, parts = train_step(model, batch, cache=None if step < grad_steps else cache, step=step,
                                             backward=True, fused=a.fused)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{total} loss {float(loss):.3f} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                      + f" {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % a.eval_every == 0 or step == total:
                m = evaluate(model, dev, cards, gold, db_paths)
                print(f"eval step {step}: {m}", flush=True)
                if m["ex"] > best:
                    best = m["ex"]
                    model.backbone.model.save_pretrained(a.out / "best_lora")
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                                  if "lora_" in k or not k.startswith("backbone.")}
                    torch.save(model.head_state(), a.out / "best_heads.pt")
                    (a.out / "best.json").write_text(json.dumps({"step": step, **m}))
            if step >= total:
                break
        epoch += 1
    print(f"done in {(time.time() - t0) / 3600:.2f} h, best dev EX {best:.3f}")
    # calibrate the abstain threshold on the held-out TRAINING databases (never the sealed sets)
    if best_state:
        model.load_state_dict(best_state, strict=False)  # calibrate the checkpoint we keep
    curve: list = []
    evaluate(model, dev, cards, gold, db_paths, limit=2000, curve=curve)
    cal = calibrate(curve)
    (a.out / "serving.json").write_text(json.dumps({k: v for k, v in cal.items() if k != "curve"}, indent=1))
    (a.out / "calibration.json").write_text(json.dumps(cal, indent=1))
    print("calibration:", cal["chosen"])


if __name__ == "__main__":
    main()
