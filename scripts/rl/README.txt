fanout_r3 / rl : execution-reward RL on the System-1 slot model (R2 checkpoint runs/r1r2/r2). No GPU used.
2026-10-08: the RL code (rl_train.py, rl_core.py, rl_data.py, heldout_eval.sh, this README) moved from the gitignored
reports/fanout_r3/rl/ to scripts/rl/ (version control, leakcheck allowlist). Data, logs and the sampling/metrics
scripts (sample_k.py, metrics_k.py, k8_existing.py, ho200/dev200.jsonl, dev_strat200.json, smoke*/) stay in
reports/fanout_r3/rl/. Steps 1-2 below run there; step 3 runs from the repo root.
P=<repo>/.venv/bin/python ; cd reports/fanout_r3/rl for steps 1-2

1. k<=8 from existing GPU candidate files (seconds):      $P -I k8_existing.py            -> k8_existing.json
2. k up to 64 on the Mac (MPS, int8 base, ~90 min total, 7B loaded once; forward ~8-14 s/q under memory pressure):
      $P -I sample_k.py --split dev heldout --out dev200.jsonl ho200.jsonl   -> sample_k.log, *.timing.json
      $P -I metrics_k.py dev200.jsonl ho200.jsonl                           -> metrics_k.out (seconds)
   dev = dev_strat200.json[:200] (stratified by DB, seed 0; ORIGINAL dev SQL as gold)
   heldout = 200 random original BIRD-train questions on the 9 held-out train DBs (original train SQL as gold)
3. RL loop: rl_train.py (+ rl_core.py, vendored from .claude/worktrees/wf_cfb20c27-79e-46/src/luthor/sq2/rl.py)
   tiny CPU smoke (~40 s):  cd repo; $P -I scripts/rl/rl_train.py --tiny --plans data/sys1/bird_train \
        --db-roots data/sys1/rrdbs --only-db trains --tiny-sft-steps 40 --steps 4 --G 8 --q-per-step 8 --beta 0.01 \
        --device cpu --workers 2 --out reports/fanout_r3/rl/smoke                       -> smoke/rl_log.json
   7B MPS step (real R2 policy, small DB 'trains'):  same with --run runs/r1r2/r2 --device mps --quant int8 --G 16 \
        --steps 2 (no --tiny)                                                             -> smoke7b/log.txt

2026-10-07: rl_train.py now samples with rl_core.sample_draws and scores each draw by the decisions decode made
(rl_core.trace_logprob). The old plan_targets path dropped ~25% of sampled plans; nothing is dropped now.
Step log: draw_failed_frac (draws decode could not finish, reward 0), trace_conflicts (expected 0).
Checked by tests/test_rl_trace.py.

2026-10-08 (Phase 0, pilot readiness for ladder rung 1b; nothing run on a GPU yet):
rl_train.py gained, all CPU-tested on the tiny backbone by tests/test_rl_pilot.py (~2 min):
  anchors      --list-weight W (rl_core.list_loss on the serving decoder's distinct k-best list, --list-k 8, + the gold
               plan (--list-gold) + right buffer plans; labels = execution vs ORIGINAL gold), alongside --sft-weight;
               --distill-weight (rl_core.soft_targets/distill_loss to right buffer plans; needs --buffer)
  memory       forward OOM -> the database's questions retried in halves, weighted so the gradient is the same (test
               checks it equal to the unsplit gradient); backward OOM -> step dropped; stop past --max-oom-skip
               (train_sq2.OOMLedger, -> OUT/oom.json); --skip-dbs works_cycles; --grad-ckpt (train mode, dropout 0)
  repro        --seed seeds random/numpy/torch and the question order; OUT/rl_args.json = args, seed, git rev, dirty
               file count, sha256 of rl_train/rl_core/rl_data, data and filter counts
  buffer       --buffer [PATHS] (04-01; default off; no paths = data/eval/sel_cands/sel/{train,val}_cands.jsonl):
               rl_data.load_buffer drops the 337 questions on held-out DBs (dev_dbs_bird.json, + --exclude-dbs) and
               asserts none remain -> 2,121 questions; ids bv_<split>_<n> -> bird_train_train_<n>; SQL parsed to plans
               (luthor.sq.convert); records gold_source "bird_verified" (the file's ok flags are vs CORRECTED gold);
               --buffer-label original (default: re-executed vs ORIGINAL gold) | verified
  gold noise   --drop-ids PATH (list / {"ids": [...]} / {id: reason} / JSONL; e.g. data/sys1/gold_noise_drop.json from
               scripts/analysis/gold_noise.py) and --skip-noisy-gold (run time: gold errors / empty / single 0)
  timing       per step s/question for forward, sample, kbest, exec, logp, list, backward (GPU synchronised);
               --timing-smoke N: N steps, no checkpoint, prints TIMING / PROJECTION (GPU-h for 1 and 2 epochs over
               --project-questions 8000, startup included, step 0 excluded) -> OUT/timing.json
  exec         rl_core.ExecReward splits each database's distinct queries over --workers processes (default
               cpu_count - 2; it used to be one process per database, i.e. one busy worker per RL call), per-query
               deadline --exec-seconds, pool killed and queries scored 0 if it does not answer
  held-out     heldout_eval.sh RUN (or --heldout-eval): rerank_sq2.py gen --k 8 on the 1,000 held-out questions
               (first pick, oracle@8) -> rerank_sq2.py score with the frozen R2 validator -> rr/heldout_compare.json
               against runs/r1r2/r2/rr/dev_metrics.json (greedy 0.606, oracle 0.712, rerank 0.674)

Phase 1 RL timing test (cheap GPU box, repo root):
  python -I scripts/rl/rl_train.py --run runs/r1r2/r2 --plans data/sys1/bird_train \
    --db-roots data/sys1/dbs data/sys1/rrdbs --exclude-dbs data/sys1/dev_dbs_bird.json --skip-dbs works_cycles \
    --G 32 --q-per-step 32 --sft-weight 0.5 --list-weight 0.5 --beta 0.01 --timing-smoke 20 --out runs/rl_timing
  (repeat with --G 64 and with --grad-ckpt if peak_mem_gb is close to the card)
Pilot arms (rung 1b, from R2):
  A: ... --epochs 1 --G 32 --sft-weight 0.5 --list-weight 0.5 --beta 0.01 --seed 0 --out runs/rl1/grpo --heldout-eval
  B: the same + --adv passk --passk 8 --out runs/rl1/passk8
