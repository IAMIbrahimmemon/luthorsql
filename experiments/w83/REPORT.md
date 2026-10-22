> **Run record (added by the orchestrator, 2026-10-22).** Workflow run wf_b81f7f70-0e5: 55 agents in total, 0 errors
> (8 idea generators, 1 selector, designers, blind answerers and scorers for 12 ideas, 1 GPU broker, 2 GPU scorers,
> 2 skeptics, 1 synthesiser). It was an exploratory, brief-driven search, not a pre-registration with decision rules.
> Only L2b got independent skeptic verification (2 skeptics, not refuted). The other results were checked by re-running
> the designers' scripts. L7b was abandoned at $0, and L2a's agent EX was not run.

# W83 report: can the cascade project to BIRD test EX ≥ 83 while saving ≥ 30% vs B1?

The file /Users/ibrahimmemon/Documents/GitHub/luthor-general/experiments/w83/REPORT.md was not written. The harness blocked report-file writes from this subagent, so the full report is below. The caller should save it to that path.

Date: 2026-10-21. All work used the DECISION split only: 796 questions over 8 DBs, 402 of them routed at t = 3. No RESERVED or fresh-set data was used, no Anthropic API was called, and GPUs were used only through the broker.

## Answer

**No.** No verified idea, and no combination of them, projects to ≥ 83 with a saving of ≥ 30%.

The best point estimate is L5b at 82.64. That rests on 3 questions in a 44-question sample from 2 DBs, and its cost figure is unreliable. At the corrected top-3 cost ratio its saving is about 28%, below the 30% bar.

There is also a hard ceiling. On DECISION, sending every question to the existing agent (A1) scores 78.89, against 77.64 for C. That is +1.25 points, and 83 needs +1.4. So no routing or gating change can reach 83 with this agent; that covers L1a, L4c, L8a, L2b and L5b. Getting there would need a more accurate routed agent. All four prompt ideas that tried this (L4b, L3b, L4a, L7a) measured 0 or negative.

**Best measured operating point that meets the 30% bar:** plain cascade C at t = 3, projected 81.6, saving 34–35%.

The only variant that measurably beats C while keeping 30% at today's cost is L2b M@0.51. It gains +0.13 (projected 81.73, saving 30.5%). That is 1 net question and not significant. Both C and M@0.51 are below the top-3 cut-off of 82.28.

## Correction that affects several ideas

Several ideas assumed the top-3 (V3) prompt cuts agent cost to 0.80x. The correction in experiments/top3_at_30/PREREG.md (dated 2026-10-22, read and confirmed) says that figure was mostly a costing bug:
- the corrected development-half ratio is 0.886;
- on the held-out confirmation half, V3 cost 14% more than V0 and lost 5.9 points.

Every saving that assumes 0.80 is therefore optimistic. Below, savings are shown at 0.80, 0.886 and 1.0 where available.

## Results by idea

Gain is the paired change in whole-set DECISION EX against C at t = 3 (77.64). Projected EX = 81.6 + gain.

| Id | What | n | Whole-set gain (pts) | Projected test EX | Saving vs B1 | Measured? | Significance | Verdict |
|---|---|---|---|---|---|---|---|---|
| L1a | Raise threshold to t = 3.75 | 796 | +0.50 (+4 net q); +0.13 when t is chosen leave-one-DB-out | 82.1 (81.7) | 34.3% @0.80, 27.6% @0.886, 18.8% @1.0 | Gain yes (old A1 answers); cost no | sign-flip p = 0.50; bootstrap CI [−0.30, +1.21] | Fail |
| L4c | Route kept questions where a lower group outvotes A (r = 1.5, chosen leave-one-DB-out) | 796 | +0.50 at r = 1.5; +0.38 at gate a | 82.10 | 40.8% @0.80, 27.0% @1.0 | Gain yes (old A1 answers); cost no | p ≥ 0.5; 6–7 discordant questions in 2–3 DBs | Fail |
| L8a | Learned expected-gain router, routed share ≤ 0.505 | 796 | +0.13 | 81.73 | 35.4% | Yes (labels from old A1 answers) | p = 1.0; CI [−0.60, +0.69] | Not adopted |
| L8a, best exploratory variant | Picked after seeing results; routed share 0.525, above the cap | 796 | +1.01 | 82.61 | 33.0% | Chosen post hoc | p = 0.156; CI lower bound 0.00 | Not valid |
| L2a | Bigger System 1 pool on routed questions (+64 samples) | 402 | at most +0.38 (upper bound, not EX) | ≤ 81.98 | 41% @0.80, 28.5% @1.0 | Oracle measured; agent EX not run | none | Ceiling below 83 |
| L2b | Agreement between two System 1 runs as a routing feature | 796 | +0.63 (G@0.6); +0.13 (M@0.51) | 82.23; 81.73 | 20.0% (G@0.6); 30.5% (M@0.51) | Yes | G@0.6: p = 0.125, CI [+0.26, +0.96], 9 wins / 4 losses | Fail (pre-registered rule) |
| L4b | Precomputed execution and SQL diff in the routed prompt | 50 (2 DBs) | 0.0 (39 vs 39) | 81.6 | ~34% assumed; cost not measured | EX yes; cost no | McNemar p = 1.0 | Fail |
| L3b | BIRD-convention card plus lint | 50 (2 DBs) | −3.03 (37 vs 34) | 78.6 | ~34% assumed; cost not measured | EX yes; cost no | 0 wins / 3 losses, p = 0.25 | Fail (harmful direction) |
| L4a | Reliability tiers (no candidates on LOW tier) | 50 (2 DBs) | −1.23 (reweighted to the routed tier mix) | 80.37 | ~34% assumed; cost not measured | EX yes; cost no | 1 win / 2 losses, p = 1.0 | Fail |
| L7a | Anti-anchoring (shuffled labels, vote counts hidden) | 50 (2 DBs) | −2.02 (31 vs 29) | 79.58 | ~34% assumed; cost not measured | EX yes; cost no | 1 win / 3 losses, p = 0.625 | Fail |
| L5b | Verify-lite agent on the kept band 3 ≤ m < 4 | 44 (2 DBs) | +1.04 on the sample; +0.63 on the full band using old A1 answers | 82.64 (82.23) | 33.4% @0.80, ~27.8% @0.886, 20.6% @1.0 | EX yes; cost estimated from transcripts | 3 wins / 0 losses; p = 0.5, the smallest possible with 2 DBs | Fail (below 83; saving < 30% at corrected ratio) |
| L2c | VALHARD veto on the agent's SQL | 402 | 0.00 (leave-one-DB-out); +0.63 if the veto were perfect | 81.6 (≤ 82.23) | unchanged | Yes, on the 40 questions scorable without a GPU | leave-one-DB-out picks "never override" | Killed, $0 GPU |
| L7b | Pairwise adjudicator when the agent rubber-stamps a low-vote A | 102 eligible | +0.38 if the judge were perfect | ≤ 81.98 | not measured | Ceiling only | none | Abandoned, $0 |

Notes on the table:
- **"Old A1 answers":** routed outcomes come from earlier agent runs, not fresh ones. Those runs saw 8 candidates from an older candidate file. Only 135 of 796 candidate lists match the current s1_decision.jsonl, and 126 of those A1 prompts had empty candidate lists.
- **Agent samples are small:** each arm ran once on 2 of the 8 DBs. Run-to-run noise alone flipped 4 of 50 picks (L4b), so a gain measured at n = 50 carries roughly ±3 whole-set points of uncertainty.
- **Agent cost:** it was not measured for L4b, L3b, L4a or L7a. Their "34%" is the C baseline saving, assuming cost is unchanged.
- **L5b cost:** it is estimated from 4 transcripts at assumed Opus prices, and output tokens were largely uncounted.
- **L5b saving at 0.886:** 1 − (0.00085 + 0.509 × 0.0278 × 0.886 + 0.152 × 0.0173) / 0.0222 = 27.8%.

## Ceilings: why 83 is out of reach with this agent

| Bound | Value |
|---|---|
| Send every question to A1, vs C | +1.25 → 82.85, at roughly 0% saving |
| Needed for 83 | +1.4 |
| L2a: agent-wrong questions where the new pool newly has the right answer | 3/796 → +0.38 |
| L2c: perfect asymmetric veto | 5/796 → +0.63 |
| L7b: perfect pairwise adjudicator | 3/796 → +0.38 |
| L5b: A1 replacing System 1 on the full band [3, 4) | +0.63 |

Combining ideas does not get past these limits:
- The routing ideas (L1a, L4c, L8a, L2b, L5b) all draw on the same +1.25 pool, so their gains overlap. The overlap between L1a and L4c was not measured, but both rest on the same few hockey and student_loan questions.
- The candidate-side ideas (L2a, L2c, L7b) could add about +0.4–0.6 each only with a perfect judge. None has a measured gain.

## Recommendation for a RESERVED confirmation

**Do not spend a RESERVED confirmation on the 83 target.** Nothing projects to 83 even before noise, and RESERVED is a limited held-out resource.

If you still want a confirmation aimed at the 82.28 top-3 cut-off rather than 83, I recommend one candidate only, and it must first be completed on DECISION.

**1. L5b verify-lite on the kept band (the only candidate)**

Config:
- Cascade C at t = 3.0 with the routed agent unchanged: PROMPT_s1, top-8 groups, ≤ 8 executions. Do not use the top-3 prompt, given the correction above.
- Questions with 3.0 ≤ margin < 4.0 go to the L5b verify-lite Opus agent: top-1 candidate only, trimmed schema lookup, ≤ 2 SQL executions, default ACCEPT. Freeze the prompt exactly as used in scratchpad/w83/L5b.
- Questions with margin ≥ 4.0 keep System 1's pick.

Do these on DECISION first:
1. Run the remaining 77 band questions on the other 6 DBs.
2. Measure cost from usage logs, including output tokens.
3. Proceed only if the full-band gain is ≥ +3 band points vs System 1 and the saving is ≥ 30% at the routed agent's real cost (ratio 1.0).

Pre-registered RESERVED rule:
- Paired against C run 1 on the same questions.
- Success means a whole-set gain ≥ 0 with a saving ≥ 30% vs B1.
- The aim is a projection ≥ 82.28. Reaching 83 is not expected.

Expected outcome: about +0.6 points (projection around 82.2), with the saving right at the 30% edge.

I recommend no second candidate:
- L2b G@0.6 saves only 20.0% at today's cost, and L4c at r = 1.5 saves only 27.0%.
- L1a fails both targets.
- L8a's only positive variant was chosen after seeing the results.

## Spend

| Item | Amount |
|---|---|
| GPU, L2a (Hyperstack H100, deletion confirmed) | $1.54 |
| GPU, L2b (Massed Compute H100, deletion confirmed) | $1.44 |
| **GPU total** | **$2.98 of the $20.00 cap**. Ledger: /private/tmp/claude-501/-Users-ibrahimmemon-Documents-GitHub/1555fb05-f5b6-4c07-82bb-2c3833dc5ab7/scratchpad/w83/gpu_ledger.jsonl. No instance is running. |
| Anthropic API | $0 (none used) |
| Opus workflow agents | Not measured: no usage logs, apart from the L5b transcript estimate |

## Data handling notes

- **Held-out data:** no RESERVED gold or answers and no fresh-set data were read.
  - The PREREG correction text was read because the brief requires that file. experiments/final_config was not opened.
  - L2b's staging step read heldout_blind/questions.json but dropped every non-DECISION row before writing anything.
  - L2c notes that the stored hockey VALHARD scores may have used table context batched with RESERVED neighbours. This does not affect its bound, which uses only EX labels.
- **L2a scripts:** the GPU broker edited L2a's box.sh and run_l2a.sh to get past a smoke-check bug. The originals are kept as .orig, and the go/no-go result does not depend on the change.
- **L2b replica:** the replica of run 1 picked the same SQL on 724 of 796 questions but matched the margin on only 373, so treat the pooled-margin feature with care.
- **Verification:** only L2b had independent skeptic verification (2 skeptics, not refuted). The other ideas were confirmed by rerunning the designers' own scripts. The exception is L4c, which was also recomputed with an independent script (indep.py).