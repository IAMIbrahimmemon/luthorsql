# scripts/hpc: Slurm jobs for a university cluster (written for UCI HPC3, kept generic)

These are templates. We don't have an allocation yet, so none of them has run on a real cluster. The training code they
call has been tested on CPU only (see "What is verified" below).

## Rule: public data only

Only public benchmark data and code go to the cluster: the `data/sys1` plan sets built from BIRD, Spider and SynSQL,
the BIRD dev databases and `dev.json`, and the public OmniSQL-7B checkpoint. The same allowlist as the GPU boxes
applies (`scripts/vast/leakcheck.sh`). Never copy anything from work or any employer data, `.env` files or
secrets, `docs/`, or `reports/` (the RL code now lives in `scripts/rl/`). Never copy sealed sets either
(`data/sys1/sealed`, `data/eval/sealed_*`, Spider-DK, Spider-Realistic, KaggleDBQA). Nothing here trains or tunes on
BIRD dev. It is only generated and scored, exactly as in the R1/R2 chain.

## One-time setup

1. Copy the code to the cluster: `src/`, `scripts/` and `pyproject.toml`, the same allowlist `push6.sh` uses.
   For example, after `bash scripts/vast/leakcheck.sh src scripts pyproject.toml`:
   `rsync -a --exclude __pycache__ src scripts pyproject.toml <hpc>:luthor/`.
   For RL, `scripts/rl/{rl_core,rl_data,rl_train}.py` (already under scripts/) are needed.
   `leakcheck.sh` does not yet have `reports/` in its allowlist, so check those three files by hand or add them to the
   allowlist first.
2. Copy the data to `$LUTHOR_DATA` on the cluster's scratch or pub space. Use the same layout as the boxes' `/root/d`:
   - `sys1/{bird_train,spider_train,synsql_train,dbs,rrdbs,bird_dev,templates.json,templates_nums.json,dev_dbs_bird.json,bird_dev_dbs.json}`
   - `dev_databases/` and `dev.json` from `data/bench/dev_20240627`
   - optionally `rare1/` (rare-skeleton plans) or teacher rows for `--extra-train`
3. Run `cp scripts/hpc/site.env.example scripts/hpc/site.env` and fill in every `CHANGE_ME`:

   | Setting | What it is |
   |---|---|
   | partition | `sinfo` |
   | GPU account | `sbalance` or `sacctmgr show assoc user=$USER` |
   | GPU type | `sinfo -o "%P %G"` |
   | data root and HF cache | big-disk paths |

   Then set `HPC_TORCH_BACKEND` to match the GPU nodes' driver (`nvidia-smi` on a GPU node: CUDA 12.4+ means `cu126`
   is fine). `site.env` is git-ignored.
4. On the login node, run `bash scripts/hpc/setup.sh`. It builds the venv, prefetches OmniSQL-7B into `$HF_HOME`
   (jobs run with `HF_HUB_OFFLINE=1`) and runs the CPU tests of the training code.

## Jobs

Always submit through `submit.sh`. It adds the partition, account, `--gres=gpu:<type>:<GPUS>`, wall time, CPUs and
memory from `site.env`. The job runs from the repo root and logs to `logs/hpc/<job>-<id>.out`. Outputs go to
`runs/hpc/$RUN`.

```bash
# SFT, R1 recipe (6,714 steps), one GPU
RUN=sft_r1 bash scripts/hpc/submit.sh sft.sbatch
# the same run on 4 GPUs of one node (torchrun data parallel), another seed, extra teacher rows at weight 2
RUN=sft_r1_s1 GPUS=4 SEED=1 EXTRA_TRAIN="$LUTHOR_DATA/teacher/rows.jsonl:2" bash scripts/hpc/submit.sh sft.sbatch
# R2 (2N steps) with the rare-skeleton plans
RUN=sft_r2 STEPS=13428 RARE=$LUTHOR_DATA/rare1 bash scripts/hpc/submit.sh sft.sbatch
# K-best candidates + validator + scored BIRD dev for a finished SFT run (one GPU)
RUN=sft_r1 bash scripts/hpc/submit.sh rerank.sbatch --dependency=afterok:<sft job id>
# GRPO on top of an SFT run (one GPU; rl_train.py has no resume: size --steps to one job)
SFT=runs/hpc/sft_r1 RUN=rl_r1 RL_ARGS="--steps 400 --G 16 --q-per-step 32" bash scripts/hpc/submit.sh rl.sbatch
```

The variables each job reads are listed in the header of its `.sbatch` file. The main ones:

- **`sft.sbatch`** runs the chain's training recipe (`scripts/chains/r1r2_chain.sh` stage 2), plus `--auto-resume`.
  - `TRAIN_EXTRA` adds flags. `TRAIN_ARGS` replaces the recipe.
  - `EXTRA_TRAIN="PATH[:W] ..."` becomes one `--extra-train` per entry.
- **`rerank.sbatch`** runs the chain's stage 3.
  - `STEPS_TO_RUN` picks a subset. Finished steps are skipped, so resubmitting continues.
- **`rl.sbatch`** runs `scripts/rl/rl_train.py`.
  - `RL_ARGS` and `RL_EXTRA` set its flags.

## Queue time limits: checkpoint and resume

A training run longer than one job's wall time continues across jobs on its own:

- `train_sq2.py --auto-resume` keeps `runs/hpc/$RUN/resume/` up to date: LoRA, heads, AdamW state, LR schedule, step,
  epoch and position in the epoch's database order, best dev EX, the OOM ledger, and every process's random state.
  It is written at every eval and every `--save-every` steps (`SAVE_EVERY`, default 250).
  - The directory is replaced as a whole, so a job killed mid-save keeps the previous state.
- `sft.sbatch` reads the time left in the job (`squeue %L`) and passes `--job-hours` (time left minus `JOB_MARGIN_H`,
  default 0.5 h). When that runs out, training saves its state, writes `runs/hpc/$RUN/job_stopped` and exits 0. The
  script then submits itself again, up to `MAX_RESUBMITS` times.
  - The next job runs the same command. It resumes at the same step, with the same LR schedule and the same data order.
  - On CPU, a stopped-and-resumed run ends with bit-identical weights to an uninterrupted one
    (`tests/test_train_sq2_main.py`).
- A resume with different training settings (plans, seed, batch shape, steps, backbone, heads) is refused.
- Changing `GPUS` between jobs is allowed. Training continues, but the per-process random streams restart.
- `--hours` still ends the run itself (final eval, then the gate). `--job-hours` only ends the job.

## Multi-GPU (data parallel)

`GPUS>1` runs `torchrun --standalone --nproc_per_node=$GPUS scripts/train_sq2.py ...` on one node:

- Each step's `--dbs-per-step` databases (the global batch) are split across the GPUs. Each database keeps weight
  1/len(chunk), and gradients are summed with one fp32 all-reduce. A step's update is therefore the single-GPU update
  of the same chunk, with the same step count and LR schedule.
  - The speed-up is at most min(GPUS, `--dbs-per-step`), and less when database sizes are unbalanced.
  - With 8 GPUs, raise `--dbs-per-step` (and expect to retune the LR).
- An OOM is always agreed across processes, so it can never deadlock:
  - An OOM in backward on any GPU drops the step on every GPU.
  - Forward-pass OOM retries stay local.
  - One process passing `--max-oom-skip` stops them all.
- `--hours`, `--fit-hours` and `--job-hours` decisions are also agreed across processes.
- Rank 0 alone evaluates and saves. The others wait (`--dist-timeout-min`, default 180). Weights are re-broadcast
  from rank 0 after every eval.
- Every process loads the whole plan set, so host RAM scales with `GPUS` (`HPC_MEM_PER_GPU_GB`).

## What is verified (Mac, CPU, tiny Qwen2) vs not

Verified on CPU:

- `--seed 0` gives HEAD's weights bit for bit (a one-off comparison against the previous `train_sq2.py`), and
  other seeds differ.
- Extra-train loading and counts work.
- 2-process gloo `torchrun` gives the same summed gradient as one process, with identical weights across ranks.
- OOM drop and stop agreement work without a deadlock.
- Resume is bit-exact.

Not verified:

- Anything on a GPU: NCCL, bf16 all-reduce numerics at 7B, real speed-up, memory headroom of the fp32 gradient buffer
  (about 4 bytes per LoRA parameter, about 0.3 GB for OmniSQL-7B with r=32).
- Slurm itself: `sbatch`/`squeue` on HPC3, module names, partition and GPU names.
- `rl_train.py` and `rerank_sq2.py` under Slurm. These scripts only invoke them.
