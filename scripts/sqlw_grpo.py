"""Execution-reward learning (GRPO) for the SQL writer, after supervised training (round 4).
Runs on the GPU pod in /workspace/venv2 (TRL + vLLM colocated on one GPU).

  python scripts/sqlw_grpo.py --model /workspace/m/sqlw_sft --data /root/d/sqlw/rl.jsonl \
      --db-roots /root/d/db BT --out runs/sqlw_rl --hours 4

Reward per sampled SQL (Arctic-Text2SQL-R1 style): 1.0 if its result set equals the gold result, 0.1 if it runs
but differs, 0 if it fails. Prompts are Spider / BIRD TRAIN questions only (never dev or sealed databases).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.execute import connect  # noqa: E402
from luthor.sqlw.prompt import clean_sql  # noqa: E402

GOLD: dict = {}


def _gold(path: str, sql: str):
    k = (path, sql)
    if k not in GOLD:
        try:
            GOLD[k] = set(map(tuple, connect(path, 10.0)(sql)))
        except Exception:
            GOLD[k] = None
    return GOLD[k]


def _one(args) -> float:
    text, gold_sql, path = args
    ref = _gold(path, gold_sql)
    if ref is None:
        return 0.0
    try:
        got = connect(path, 3.0)(clean_sql(text))
    except Exception:
        return 0.0
    return 1.0 if set(map(tuple, got)) == ref else 0.1


POOL = ThreadPoolExecutor(16)
STATS = {"n": 0, "right": 0, "runs": 0}


def exec_reward(prompts, completions, gold_sql, db_path, **kw) -> list[float]:
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    r = list(POOL.map(_one, zip(texts, gold_sql, db_path)))
    STATS["n"] += len(r)
    STATS["right"] += sum(x == 1.0 for x in r)
    STATS["runs"] += sum(x > 0 for x in r)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--db-roots", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8, help="completions per forward/backward")
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.0, help="KL penalty toward the starting model (round 5: 0.02-0.04)")
    ap.add_argument("--temp", type=float, default=1.0)
    a = ap.parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, str(p))
    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if r["db_id"] in paths]
    if a.limit:
        rows = rows[:a.limit]
    # drop questions whose gold SQL fails or takes too long here (no reward signal)
    keep = [r for r in rows if _gold(paths[r["db_id"]], r["gold_sql"]) is not None]
    print(f"rl prompts: {len(keep)} of {len(rows)} (gold runs)", flush=True)
    ds = Dataset.from_list([{"prompt": r["messages"], "gold_sql": r["gold_sql"], "db_path": paths[r["db_id"]]}
                            for r in keep]).shuffle(seed=0)

    t0 = time.time()

    class Stop(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if STATS["n"]:
                print(f"RL step {state.global_step} right {STATS['right'] / STATS['n']:.3f} runs "
                      f"{STATS['runs'] / STATS['n']:.3f} elapsed {(time.time() - t0) / 3600:.2f}h", flush=True)
                STATS.update(n=0, right=0, runs=0)

        def on_step_end(self, args, state, control, **kw):
            if time.time() - t0 > a.hours * 3600:
                control.should_training_stop = True
                control.should_save = True

    per_step = a.prompts_per_step * a.gens
    cfg = GRPOConfig(
        output_dir=str(a.out), learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
        per_device_train_batch_size=a.micro, gradient_accumulation_steps=per_step // a.micro,
        num_generations=a.gens, max_completion_length=256, temperature=a.temp, beta=a.beta,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem, vllm_max_model_length=4352,
        mask_truncated_completions=True,
        bf16=True, gradient_checkpointing=True, logging_steps=5, save_steps=50, save_total_limit=2,
        max_steps=a.max_steps, report_to=[], seed=0)
    tr = GRPOTrainer(model=a.model, reward_funcs=exec_reward, args=cfg, train_dataset=ds,
                     peft_config=LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, target_modules="all-linear",
                                            task_type="CAUSAL_LM"),
                     callbacks=[Stop()])
    tr.train()
    tr.save_model(str(a.out / "lora"))
    print(f"DONE_RL in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
