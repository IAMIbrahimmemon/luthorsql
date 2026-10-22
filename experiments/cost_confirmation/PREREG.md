# Measured cost saving of the System 1 → Opus cascade on held-out questions (pre-registered 2026-10-20)

Owner (2026-10-20): "we need to really show the 44% cost savings per question ... and we need to write it in a way
we can stand behind it." The 44% in the current draft is modelled (one average cost per agent call), tuned on the
same 796 DECISION questions it is reported on, uses gold SQL to route out-of-grammar questions, and is measured only
against the agent *with* candidates. This experiment replaces it with a measured, held-out, gold-free number against
the baselines a reader would actually deploy.

## Fixed before any run (do not change after the first agent run)

**Questions.** RESERVED: all 348 questions (8 held-out BIRD-train databases, `data/sys1/heldout_v2/reserved.json`,
ids_sha256 `ede63aab…`). Scored on the original BIRD-train gold. RESERVED was never used to choose the threshold,
the prompts or the agent. (It was scored by System 1 alone several times before; no agent has ever run on it.)

**System 1.** The frozen two-adapter LuthorSQL: R1, k = 32 distinct plans, VALHARD, highest-scoring live candidate.
It is run on **all 348 questions**, including the 58 whose gold query the grammar cannot express. No gold
information is used for routing, candidates or margins. The margin is the VALHARD score gap between the top two result
groups. Questions with fewer than two groups have no margin and go to the agent.

**System 2.** Claude Opus 5.5 run as Claude Code subagents (amended 2026-10-20, before any agent run: the owner is
keeping the API key for the BIRD test submission). This is the same setup as every earlier Opus number in the paper:
- each subagent gets one chunk of at most 25 questions, all from **one database**, so cost can be totalled per database;
- prompts `PROMPT_alone.md` (B0) and `PROMPT_s1.md` (B1, C), exactly as in the paper's appendix;
- the read-only SQL tool in a sandbox that holds only the 8 databases (no gold, no repository);
- at most 8 executions per question, given as an instruction;
- chunks of every arm are run in interleaved waves.

Token usage comes from each subagent's logged `usage` (input, cache writes, cache reads, output). Output is also
estimated as written characters / 4, because logged output counts were incomplete in earlier runs; both are reported,
and the larger is used for cost.

**Arms** (all on the same 348 questions):
- **B0, agent alone:** Opus without candidates, on every question. This is what one would deploy without System 1.
- **B1, agent with candidates:** Opus with System 1's top-8 result groups, on every question.
- **C, cascade, t = 3.0:** System 1 answers alone when its margin is ≥ 3.0; every other question gets the B1 prompt.
  C is run as its own separate subagent runs, exactly as deployed. It does not reuse B1's answers.

**Cost accounting.**
- **Agent cost:** each arm's logged token usage, priced at Anthropic's published list prices for Opus 5.5:
  $4/M input, $5/M 5-minute cache write, $0.20/M cache read and $20/M output (the pricing page, fetched 2026-10-20,
  saved with the results). Each arm is totalled per database; cost per question = arm total / 348.
- **System 1 cost:** measured GPU wall time for the batched run on the rented GPU × its hourly price, divided by 348.
  It is charged to every question in C (and reported separately).
- **Not counted:** human time and the agent tool's own CPU time. Both are the same in every arm and negligible.

**Primary endpoints.**
1. **Cost saving vs B1** = 1 − cost(C)/cost(B1), per question, with a 95% interval by database-cluster bootstrap
   (10,000 resamples of the 8 databases; costs are summed per database because chunks never mix databases).
2. **Accuracy C − B1** in points, using the exact database sign-flip test and its inverted 95% interval, plus McNemar
   wins and losses.

**Secondary endpoints (all reported, whatever they show).**
- The cost saving and accuracy difference vs **B0** (agent alone).
- **Agent call share:** the fraction of questions with margin < 3.0 (label-free; computed before the cascade arm runs and
  written down here once known).
- **Latency:** System 1 seconds per question, and agent wall time per routed question.
- **Routing baselines at the same agent share,** computed from the B1 answers:
  - random routing (expected value);
  - routing by System 1 vote agreement (the share of its 32 candidates in the top result group);
  - routing by question length.
- **Price sensitivity:** the cost saving with caching off, and at Sonnet 5.5 prices ($2 / $2.50 / $0.10 / $10).

**Pass bars (written before the run).**
- **Cost:** saving vs B1 ≥ 30%, with the lower end of the database-bootstrap interval ≥ 20%.
- **Accuracy (non-inferiority):** the lower end of the sign-flip interval for C − B1 ≥ −2.5 points.
- **CLAIM** = both pass. **PARTIAL** = the cost bar passes but non-inferiority is not shown. **FAIL** = the cost bar
  fails.

**Predictions (falsifiable).**
- The agent share is 50–62%.
- Cost saving vs B1 is 35–50%.
- C − B1 is between −1.5 and +0.5 points.
- B1 − B0 is between −2 and +3 points.
- The vote-agreement router at the same share is at most 1 point better than the margin router.

## Also re-run (secondary, DECISION, gold-free)

Run System 1 on the 126 out-of-grammar DECISION questions. Re-route all 796 by margin alone (t = 3.0), run the B1
prompt (subagents) on any newly routed question that has no handoff answer with candidates, and report gold-free DECISION numbers
next to the old oracle-routed ones. This is a robustness check only: DECISION was used to choose t.

## How the result will be written (fixed now, numbers filled in after)

> On 348 held-out questions that played no part in choosing the threshold, routing by System 1's confidence sent
> X% of questions to Claude Opus 5.5 and cut the measured agent cost per question by S% (95% CI [a, b]) relative to
> calling Opus with System 1's candidates on every question, and by S0% relative to Opus alone, with an accuracy
> change of D points (95% CI [c, d]). Costs are metered token usage at Anthropic's list prices on <date>, plus System
> 1's measured GPU time, with Opus run as Claude Code subagents.

If the result is PARTIAL or FAIL, the paper reports that instead, and the 44% DECISION figure is described only as an
in-sample estimate.

## Budget and stop rules

- **Agent runs** use the owner's subscription (no API key); their list-price value is what the paper reports.
- **GPU:** one H100 for about 30–60 minutes (System 1 on 58 + 126 questions, plus a timed batched run on all 348),
  about $5.
- **No BIRD dev look is used.** RESERVED is scored once per arm, and every arm is reported.
- A chunk is rerun only if it produced no output or the tool crashed. Errors are logged.

## Amendment 1 (2026-10-21, before any agent run; owner request)

**Arm B2 (prompt-only control).** This is the handoff prompt `PROMPT_s1.md` with every candidate list empty
("Some questions have no candidates"), on all 348 questions, run like B0 and B1. B1 − B2 isolates the effect of the
candidates from the effect of the prompt. B2 − B0 isolates the prompt.

**End-to-end time (secondary endpoint).**
- Per arm: summed agent time (each subagent's logged duration) plus System 1's measured GPU time for the questions
  that arm runs it on.
- **Time saving vs B1** = 1 − time(C)/time(B1). **Time saving vs B0** = 1 − time(C)/time(B0).
- Both get a database-bootstrap 95% interval (agent time is summed per database because chunks never mix databases).
- System 1 time is the batched run's GPU seconds per question from the held-out package run, plus its per-question
  median latency at batch size 1 from the paper's latency measurement (0.84 s on one H100). Both are reported.
- Pass bar for a time claim: saving vs B1 ≥ 25% with the lower interval end ≥ 15%.
- Prediction: saving vs B1 is 30–50%.

**Candidate effect by whether a right candidate exists (secondary).**
- On B1's questions, split by whether any shown candidate is correct (computed after scoring).
- Report B1 − B0 and B1 − B2 accuracy in each split, with McNemar counts.
- Prediction: candidates help when a right one exists (B1 > B2 there) and hurt when none does (B1 < B2 there).

**Agents weaker than System 1 on confident questions (reporting only).** No new runs; this reuses the cross-agent
results on DECISION and is described as in-sample.

## Recorded before the cascade arm ran (2026-10-21)
- Held-out package run (gold-free, all 1,144 DECISION + RESERVED questions): one H100 PCIe (Scaleway, $3.30/h),
  1,060 s for 1,144 questions including model load, so 0.927 s per question and $0.00085 per question.
  `runs/phase1/heldout_s1/heldout/`.
- RESERVED agent share at t = 3.0: **177 of 348 (50.86%)**, of which 18 have no margin. Chunks: B0, B1 and B2 have
  18 each, C has 11.

## VERDICT (scored 2026-10-21; `result.json`, `secondary.json`, `leak_check.jsonl`)

**FAIL by the pre-registered rule.** The cost saving vs B1 is 29.86%, under the 30% bar (its lower bound, 22.84%,
clears the 20% bar). Non-inferiority is not shown either: the lower end for C − B1 is −2.60, below −2.5. The time bar
fails. Every arm is reported below, and nothing was rerun. All 65 chunks completed with no missing answers. The leak
check is 60 of 65 clean; the 5 flags are each chunk's own answer paths, with no access across chunks.

| | B0 alone | B1 + candidates | B2 prompt only | C cascade | System 1 |
|---|---|---|---|---|---|
| EX (348) | 80.17 | 80.46 | 81.32 | 81.32 | 63.22 |
| $/question (list price, incl. System 1 for B1/C) | 0.02146 | 0.02422 | 0.02267 | 0.01699 | 0.00085 |
| seconds/question (agent + System 1) | 2.94 | 3.43 | 3.44 | 3.01 | 0.93 |

**Primary endpoints**
- Cost saving vs B1: **29.86%** (DB bootstrap 95% CI [22.84, 38.18]).
- C − B1 accuracy: **+0.86** points (sign-flip p = 0.69, CI [−2.60, +3.65]; McNemar 13 wins / 10 losses).

**Secondary endpoints**
- Against B0: cost saving **20.85%** [13.63, 29.07]; C − B0 = **+1.15** points (p = 0.50, CI [0.00, +3.00]; 13 / 9).
- Agent share: 177/348 = 50.86%.
  - Routed questions (177): System 1 alone 36.72; C 72.32, B1 68.93, B0 68.93.
  - Kept by System 1 (171): System 1 90.64, against B1 92.40 and B0 91.81 on the same questions.
- Time, end to end:
  - saving vs B1 **12.28%** [6.35, 18.25];
  - saving vs B0 **−2.42%** [−8.96, +6.27], so the cascade is not faster than Opus alone.
  - The agent spends longer per question on routed questions (about 4.1 s each) than its average (2.9 s).
- Routing baselines at the same share, using B1's answers on routed questions: the margin router is the best by
  more than 5 points.
  - margin 79.60
  - vote agreement 74.43
  - question length 73.85
  - random (expected) 71.99
  - all to the agent 80.46
  - System 1 only 63.22
- Price sensitivity (saving vs B1 / vs B0):
  - Opus list: 29.9 / 20.9
  - Opus, no caching: 20.2 / 18.8
  - Sonnet 5.5 list: 28.9 / 16.9
  - Sonnet, no caching: 20.1 / 17.9
- System 1's GPU time is 5.0% of the cascade's cost.
- Split by whether a right candidate was shown:
  - right one shown (253): B0 91.30, B1 92.09, B2 91.70 (B1 vs B2: 9 / 8);
  - none (95): B0 50.53, B1 49.47, B2 53.68 (B1 vs B2: 5 / 9).
- B1 − B0 = +0.29 (p = 1.0), B1 − B2 = −0.86 (p = 0.97), B2 − B0 = +1.15 (p = 0.66).

**Predictions**
- Hit:
  - agent share in 50–62% (50.86);
  - B1 − B0 in [−2, +3] (+0.29);
  - vote agreement not more than 1 point better than margin (it is 5.2 worse);
  - candidates help when a right one exists and hurt when none does (direction only; the counts are small).
- Missed:
  - cost saving 35–50% (29.86);
  - C − B1 in [−1.5, +0.5] (+0.86, better than predicted);
  - time saving vs B1 30–50% (12.28).

**What the paper may say.** By the rule above, the 44% DECISION figure is an in-sample estimate only. The held-out
statement is:

> On 348 held-out questions that played no part in choosing the threshold, routing by System 1's confidence sent
> 50.9% of questions to Claude Opus 5.5 and cut the measured cost per question by 30% (95% CI [23, 38]) relative to
> calling Opus with System 1's candidates on every question, and by 21% ([14, 29]) relative to Opus alone, with an
> accuracy change of +0.9 points ([−2.6, +3.7]) and +1.1 points ([0.0, +3.0]) respectively.

The paper must also say that this narrowly missed the pre-registered 30% bar, and that non-inferiority against B1 was
not shown at the −2.5 margin. No time saving may be claimed against Opus alone.

## Amendment 2 (2026-10-21, after the verdict, before any replicate ran; owner request): replicates

The verdict above is final. Replicates are for reporting only and cannot change it.

**Runs.**
- Two more complete runs of all four arms (B0, B1, B2, C) on RESERVED, with the same chunk files, prompts, sandbox
  contents and rules.
- Each run gets a fresh sandbox (`s2res_r2`, `s2res_r3`) with empty answer folders, and the arms are interleaved in
  waves.
- System 1's answers and margins are deterministic and reused.

**Reported.**
- Per arm: accuracy and cost per question in each of the 3 runs, and their mean and range.
- Per run: the cost saving vs B1 and vs B0, and C − B1 and C − B0.
- The 3-run mean of each saving, with a database bootstrap that resamples databases and averages runs.
- Agreement between runs: the share of questions that the same arm got right in all 3 runs, in none, or in some.

**How the paper uses it.** The run-1 numbers stay primary. The paper adds the 3-run mean and range, and says whether
the run-1 saving sits inside the range.

**Prediction.** Per-arm accuracy varies by at most 2 points between runs, and the cost saving vs B1 stays in 22–38%.

### Amendment 2 results (scored 2026-10-21; `result_r2.json`, `result_r3.json`, `replicates.json`, `leak_check_r2/r3.jsonl`, `INCIDENTS.md`)

All 130 replicate chunks completed, with no missing answers.

**Leak check.**
- r2: 61 of 65 clean. r3: 57 of 65 clean.
- Every flag except one is an agent listing or reading its own answer file, or a `/tmp` helper file.
- The exception is r2 B0 `cs_semester_2`, which ran B2's helper script (`INCIDENTS.md`).
- No other transcript read another chunk's files.

**Per run.** Accuracy is EX on 348; savings are against B1 and B0.

| | B0 | B1 | B2 | C | saving vs B1 | saving vs B0 | C − B1 | C − B0 |
|---|---|---|---|---|---|---|---|---|
| r1 | 80.17 | 80.46 | 81.32 | 81.32 | 29.86% | 20.85% | +0.86 | +1.15 |
| r2 | 79.60 | 81.32 | 79.60 | 80.46 | 36.72% | 29.03% | −0.86 | +0.86 |
| r3 | 78.74 | 80.46 | 79.02 | 79.60 | 36.17% | 28.58% | −0.86 | +0.86 |

**Pooled over the 3 runs** (database bootstrap that resamples databases and sums all runs; accuracy is each question's
mean over runs, tested with a database sign-flip):
- Saving vs B1: **34.26%**, 95% CI [28.11, 41.63].
- Saving vs B0: **26.18%**, [18.73, 35.31].
- C − B1: −0.29 points, [−2.79, +1.52] (p = 0.94).
- C − B0: +0.96, [−0.61, +2.94] (p = 0.34).
- B1 − B2: +0.77, [−2.15, +3.89] (p = 0.59).
- Mean cost per question: B0 $0.0217, B1 $0.0244, B2 $0.0222, C $0.0160.

**Run-to-run agreement** (right in all 3 runs / none / some):

| Arm | all 3 | none | some |
|---|---|---|---|
| B0 | 266 | 61 | 21 |
| B1 | 268 | 55 | 25 |
| B2 | 268 | 61 | 19 |
| C | 267 | 56 | 25 |

**Incident sensitivity.** Without r2's B0 `cs_semester_2` questions (n = 331), r2 C − B0 is +0.60 instead of +0.86.

**Time (not registered for the replicates; reported because run 1 reported it).**
- Saving vs B1: 12.3% / 23.6% / 21.7% across the three runs.
- Saving vs B0: −2.4% / +28.6% / +24.7%.
- B0 seconds per question: 2.94 / 3.91 / 3.82.
- Agent wall time varies with the harness's concurrent load, so time savings are not reliable in this setup.

**Predictions.**
- The cost saving vs B1 stayed in 22–38%: hit (29.9–36.7%).
- Per-arm accuracy varies by at most 2 points: missed for B2 (range 2.30). B0 1.43, B1 0.86, C 1.72.

**Reading.**
- Run 1 had the lowest saving of the three runs, so its FAIL verdict stands.
- The 3-run estimate of the saving vs B1 is about 34%, with a lower bound of 28%.
- Across runs, the cascade is not distinguishable in accuracy from either baseline.

**Scorer note.** `luthor.stats.cluster_signflip` casts scores to bool, so it is wrong for per-question means over runs.
`replicates.py` uses a version for fractional scores (`signflip_frac`). Earlier uses passed booleans and are
unaffected.
