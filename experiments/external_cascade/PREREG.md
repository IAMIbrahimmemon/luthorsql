# The t = 3.0 cascade on two benchmarks no LuthorSQL component has seen (pre-registered 2026-10-21)

Owner, 2026-10-21: "step 2 is doable. lets do it." The goal is a second held-out test of the System 1 → Opus cascade,
on benchmarks outside BIRD, with many more databases than RESERVED's 8.

## Fixed before any run (do not change after the System 1 run starts)

**Questions.** Two sets, sealed since 2026-10-04 (`docs/EVAL_LEDGER.md`):
- **KaggleDBQA test:** 185 questions, 8 real Kaggle databases, file `data/eval/sealed_kaggledbqa_test.jsonl`.
- **Spider-Realistic:** 508 questions, 19 Spider dev databases, file `data/eval/sealed_spider_realistic.jsonl`.

Both are scored on their official gold SQL: executed read-only, 30 s limit, set equality of result rows (BIRD-style
EX, not Spider's test-suite metric).

**Prior exposure, disclosed in full.**
- No LuthorSQL component, threshold, prompt or agent setting was chosen on either set.
- On 2026-10-05, an abandoned selector experiment (Arctic-Text2SQL candidates with an OmniSQL-7B cross-encoder that
  failed to generalize) scored both sets once: Arctic greedy 55.1 (KaggleDBQA) and 87.0 (Spider-Realistic). Nothing
  from that experiment was kept.
- Earlier systems, not LuthorSQL, were scored on the Spider dev databases. Spider dev is not in OmniSQL's or our
  training data.
- KaggleDBQA has two partly overlapping databases: WhatCDHipHop has the same schema as BIRD-train `music_tracker`,
  and TheHistoryofBaseball overlaps Spider-train `baseball_1`. It is reported on all 185 and on the clean 130.
- Frontier agents have likely seen Spider; that would favour the agent arms, not System 1.

**System 1.** The frozen two-adapter LuthorSQL package, unchanged (the same build as the RESERVED test). It runs
gold-free on all 693 questions, including the ones its grammar cannot express, on one rented GPU. The margin and the
top-8 result groups are computed exactly as for RESERVED (`experiments/cost_confirmation/s1_from_package.py` logic).
t = 3.0, not re-tuned.

**System 2.** Claude Opus 5.5 as Claude Code subagents, set up exactly as in the RESERVED test:
- the same prompts, word for word (they say "BIRD" and mention evidence; these sets have no evidence, so the field is
  empty);
- chunks of at most 25 questions from one database;
- a sandbox that holds only these databases and the read-only SQL tool;
- at most 8 executions per question (an instruction);
- interleaved waves.

**Arms.**
- **B0:** Opus alone, all questions.
- **B1:** Opus with System 1's candidates, all questions.
- **C:** the cascade at t = 3.0, run as its own separate subagent runs, exactly as deployed.

No prompt-only arm: the RESERVED replicates cover that control.

**Cost and time.** As in the RESERVED test:
- Agent cost: logged usage at Opus 5.5 list prices ($4/M input, $5/M cache write, $0.20/M cache read, $20/M
  output; output = max(logged, written characters / 4)), totalled per database.
- System 1 cost: measured GPU seconds × the hourly price, per question.
- Time: summed agent time plus System 1's batched GPU time.

**Primary endpoints (pooled over both sets, 693 questions, 27 database clusters).**
1. Cost saving vs B1 = 1 − cost(C)/cost(B1), with a database-bootstrap 95% interval (10,000 resamples, seed 20261021).
2. Accuracy C − B1: exact database sign-flip test or, with more than 20 clusters, 100,000 random sign flips (seed
   20261021), with its inverted 95% interval and McNemar counts.

**Secondary endpoints (all reported).**
- Each set separately, and KaggleDBQA's clean 130.
- Everything against B0.
- Agent share, written down before the cascade arm runs.
- Time savings vs B1 and B0.
- Routing baselines at the same share, scored with B1's answers: vote agreement, question length, random.
- Price sensitivity: no caching, and Sonnet 5.5 prices.

**Pass bars (same as RESERVED).**
- Cost: saving vs B1 ≥ 30%, with the bootstrap lower end ≥ 20%.
- Accuracy: the lower end of the C − B1 interval ≥ −2.5 points.
- Time: saving vs B1 ≥ 25%, with the lower end ≥ 15%.
- Verdict:
  - **CLAIM:** the cost and accuracy bars both pass.
  - **PARTIAL:** cost passes, but accuracy is not shown.
  - **FAIL:** cost fails.

**Predictions.**
- Agent share: KaggleDBQA 55–80% (grammar coverage 74%, a new domain); Spider-Realistic 25–50%.
- Pooled cost saving vs B1: 35–60%.
- C − B1 pooled: between −3 and +1 points.
- B0 on Spider-Realistic is at least 85% (likely memorisation); B0 on KaggleDBQA is 50–70%.
- The margin router beats vote agreement by at least 2 points at the same share.

**Rules.**
- Each set is scored once per arm, and every arm is reported. A chunk is rerun only if it produced no output or the
  tool crashed.
- Transcripts are scanned for access outside each chunk's own files before scoring.
- GPU: one or two rented H100s (racing providers), with a spend cap of $10.

## Recorded before the cascade arm ran (2026-10-21)
- **System 1 run.** One H100 PCIe (Massed Compute, $2.73/h): 607 s for 693 questions including model load, so
  0.876 s and $0.00066 per question. Output in `runs/phase2/external_s1/external/`.
  - A first instance was destroyed before it ran ($0.05): the leak gate refused the data path. The inputs were moved
    under the allowlisted `data/sys1/external_blind`.
  - A Scaleway instance that refused root login was deleted unused.
- **Agent share at t = 3.0:**
  - KaggleDBQA: 116 of 185 (62.70%), 9 with no margin.
  - Spider-Realistic: 244 of 508 (48.03%), 20 with no margin.
  - Pooled: 360 of 693 (51.95%).
- **Chunks:** B0 42, B1 42, C 29.
- **Input detail fixed here.** The sandbox and System 1 both get KaggleDBQA's official column descriptions, written
  as BIRD-style `database_description` CSVs. Spider has none.
- **Deviation.** With 27 clusters the accuracy test uses `luthor.stats.cluster_signflip`'s 200,000 random sign flips
  (seed 0), not the registered 100,000 (seed 20261021).

## VERDICT (scored 2026-10-21; `result.json`, `secondary.json`, `leak_check.jsonl`)

**CLAIM by the pre-registered rule.** Pooled over 693 questions and 27 databases:
- The cost saving vs B1 is **31.67%** (bootstrap 95% CI [24.56, 38.21]), clearing both the 30% bar and the 20%
  lower-end bar.
- C − B1 is **+1.30** points (interval [−1.96, +4.91], p = 0.60; McNemar 28 / 19). The lower end, −1.96, is above
  −2.5, so non-inferiority is shown.
- The time bar fails.

All 113 chunks completed, with no missing answers. The leak check is 112 of 113 clean. The one flag is the word "gold"
in B0 `flight_2_2`'s final report; it touched no other file.

| | n | share | B0 | B1 | C | S1 | $/q B0 / B1 / C | saving vs B1 | vs B0 |
|---|---|---|---|---|---|---|---|---|---|
| Pooled | 693 | 51.9% | 88.31 | 86.87 | 88.17 | 70.27 | .0180 / .0221 / .0151 | 31.7% [24.6, 38.2] | 16.2% [7.5, 25.0] |
| KaggleDBQA | 185 | 62.7% | 76.22 | 80.00 | 81.62 | 58.38 | .0225 / .0258 / .0193 | 25.3% [9.6, 36.5] | 14.3% [−4.9, 27.5] |
| Spider-Realistic | 508 | 48.0% | 92.72 | 89.37 | 90.55 | 74.61 | .0164 / .0207 / .0136 | 34.6% [26.6, 42.1] | 17.1% [6.9, 28.6] |
| KaggleDBQA clean | 130 | 57.7% | 78.46 | 80.77 | 81.54 | 59.23 | .0219 / .0248 / .0204 | 18.0% [3.6, 32.9] | 7.1% [−15.5, 25.3] |

**Accuracy against Opus alone (C − B0).**
- Pooled: −0.14 points ([−3.36, +3.16], 24 / 25).
- KaggleDBQA: +5.41 ([−2.32, +16.17], p = 0.34).
- Spider-Realistic: −2.17 ([−5.26, 0.00], p = 0.14; 8 / 19), a borderline loss.
  - On Spider-Realistic, candidates hurt the agent: B1 − B0 = −3.35 ([−10.55, +0.69]).
  - Opus alone scores 92.7% there, consistent with prior exposure to Spider.

**Time** (agent time + System 1 time, seconds per question; B0 / B1 / C = 1.68 / 2.42 / 2.01):
- Saving vs B1: 17.1% [12.5, 21.4].
- Saving vs B0: −19.9% [−34.6, −5.1]. **The cascade is slower than Opus alone.**

**Routing baselines** at the same share, scored with B1's answers (pooled):
- margin 85.57
- vote agreement 81.82
- question length 80.81
- random 78.89
- KaggleDBQA alone: margin 78.92, vote agreement 70.27.

**Price sensitivity** (saving vs B1 / vs B0):
- Opus list: 31.7 / 16.2
- Opus without caching: 29.5 / 17.1
- Sonnet list: 30.7 / 12.5
- Sonnet without caching: 29.2 / 16.1

**Predictions**
- Hit:
  - agent share for KaggleDBQA (62.7) and Spider-Realistic (48.0);
  - B0 on Spider-Realistic ≥ 85 (92.7);
  - margin beats vote agreement by ≥ 2 (+3.75).
- Missed:
  - pooled saving 35–60% (31.7);
  - C − B1 in [−3, +1] (+1.30, better than predicted);
  - B0 on KaggleDBQA 50–70 (76.2).

**What the paper may say.** On 693 questions from two benchmarks that no LuthorSQL component had seen, the same
t = 3.0 cascade sent 52% of questions to Opus and cut measured cost by 31.7% against Opus with candidates, with
non-inferior accuracy (pre-registered CLAIM). Against Opus alone it saved 16.2% at equal accuracy pooled. The paper
must also report:
- the borderline loss on Spider-Realistic against Opus alone (−2.2, interval reaching 0);
- that the cascade was slower than Opus alone.
