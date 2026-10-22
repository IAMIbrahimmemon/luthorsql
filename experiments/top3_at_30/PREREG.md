# Top 3 on BIRD test while keeping a ≥ 30% saving (pre-registered 2026-10-22, before any run)

**Goal.** Find a cascade configuration at t = 3.0 that meets both targets:
- **Accuracy:** projected BIRD-test EX ≥ 83.0, with a margin over 3rd place on the overall leaderboard (82.28 on
  2026-10-21).
- **Cost:** saving vs B1 (Opus + candidates on every question) ≥ 30%, with the RESERVED cost accounting of
  `experiments/cost_confirmation/score.py`.

**Cost facts measured before this pre-registration** (RESERVED runs 1–3, all chunks; least-squares fit of cost on
questions per chunk):
- B1 costs about $0.21 fixed per chunk plus $0.011 per question ($0.0222/q overall).
- C's routed arm costs about $0.13 fixed per chunk plus $0.019 per question ($0.0278/q).
- Chunking stays as it is (≤ 25 questions, one database per chunk) for every arm. Batching more questions per session
  would also apply to the baseline and is not a lever.

**Data split.** Seed 20261022 shuffle of RESERVED's 8 databases:
- development: car_retails, law_episode, synthea, university (92 routed questions);
- confirmation: cs_semester, hockey, movie, student_loan (85 routed questions).
Routed-half experiments use the renamed-schema copies (`experiments/renamed_routed/`, recall-free). Their existing run
is V0.

**Variants** (prompts in `prompts/`; the same harness, tool, 8-execution limit and chunk files as V0):
- **V1 answer-first:** `PROMPT_s1_answerfirst.md`, 8 candidate groups.
- **V2 answer-first, top 3:** `PROMPT_s1_answerfirst.md`, only the top 3 candidate groups shown.
- **V3 second attempt, top 3:** `PROMPT_s1_secondattempt.md`, only the top 3 candidate groups shown.
- **G stricter gate** (offline, no agent run). Also route a kept question when System 1's top group returns 0 rows, or
  a first row that is all NULL, or when the top group holds fewer than 8 of the 32 samples. Scored with B1's existing
  answers (runs 1–3 mean) as the agent's answer on newly routed questions, and System 1 otherwise. Development and
  confirmation as above.

**Per variant on development.**
- Routed EX vs V0 on the same items, paired.
- Cost ratio vs V0: agent dollars per question from the transcripts, same pricing.
- Projected saving = 1 − (S1 cost + routed share × RESERVED C routed cost per question × cost ratio) / B1 cost per
  question, all pooled over RESERVED runs 1–3.

**Selection.**
- Among variants whose projected saving is ≥ 30%, pick the one with the highest development routed EX, and only if it
  beats V0 by ≥ 2 points; otherwise none is selected.
- G is added if it raises the projected EX on development and keeps the saving ≥ 30%.

**Confirmation** (the selected configuration only, run once on the confirmation half).
- Projected routed EX = 72.88 (V0, all 177) + (variant − V0 on the confirmation half).
- Projected test EX = kept share × kept EX + routed share × routed EX, with G's effect on kept share and kept EX if G
  is selected.
- **Pass** if projected test EX ≥ 83.0 and projected saving ≥ 30%. Otherwise the targets are not met by these levers
  and we stop tuning on this data.

**Caveats, stated now.**
- Development halves are small (92 and 85 questions; one question is about 1.1 points). Run-to-run variation of the
  routed arm on RESERVED was 68.9–72.3 across three runs.
- The confirmation half is the guard. Any pass is still a projection to BIRD test, not a measurement.

## RESULT: development half (2026-10-22)
Scored with `score_variants.py --half dev` → `dev.json`. Leak scan: 15/15 transcripts clean
(`leak_check_dev.jsonl`). Every variant answered all 92 questions.

| | routed EX (92) | − V0 | own SQL | $/q | cost vs V0 | projected saving vs B1 | projected test EX |
|---|---|---|---|---|---|---|---|
| V0 (renamed run) | 76.09 | | 42 | 0.0355 | 1.00 | 35.0% | 81.6 |
| V1 answer-first | 76.09 | 0.00 | 36 | 0.0350 | 0.99 | 35.8% | 81.56 |
| V2 answer-first + top 3 | 75.00 | −1.09 | 37 | 0.0296 | 0.83 | 45.1% | 81.01 |
| V3 second attempt + top 3 | 76.09 | 0.00 | 38 | 0.0284 | 0.80 | 47.2% | 81.56 |

- **Selection rule** (≥ 2 points over V0 with saving ≥ 30%): **no variant qualifies.** As pre-registered, nothing goes
  to confirmation; these prompt levers do not reach 83.
- **G (stricter gate)** is not yet evaluated. It needs the BIRD-train databases on the external drive, which is
  unmounted.
- **Observed (exploratory, not a selection).** Showing the top 3 candidate groups instead of 8 cut the routed half's
  agent cost by 17–20% with no measurable accuracy change on the development half (V3 76.09 = V0). If it holds on the
  confirmation half, this frees budget: under the ≥ 30% target, routed cost may rise 8.1% above V0, and at V3's cost
  that leaves about 35% headroom for extra Opus work.

## Round 2 (pre-registered 2026-10-22, before any round-2 run)
**Configuration R2.**
- **Agent 1** is variant V3 (second-attempt prompt, top 3 candidates); its development-half answers already exist.
- **Agent 2** is an independent fresh V3 agent, run only on the questions where agent 1's answer returns a different
  result from System 1's top candidate A. This check is computed by execution on the renamed database, without gold.
- **Final answer.** The result at least two of {agent 1, agent 2, A} agree on; if all three differ, agent 1.
- **Cost.** Agent 1's dollars plus agent 2's dollars, per routed question.

**Development and selection.**
- Same rule as round 1: at least +2 points over V0 on the development half, with projected saving ≥ 30% by the
  pre-registered formula.
- **If selected:** agent 1 and agent 2 are run the same way on the confirmation half. It passes if projected test EX
  ≥ 83.0 and saving ≥ 30%.
- **If not selected:** stop. Report the two operating points (accuracy at the current saving, and the best
  configuration found).

G is evaluated when the BIRD-train drive is mounted and combined as pre-registered.

## RESULT: round 2, development half (2026-10-22)
Scored with `score_r2.py` → `dev_r2.json`. Leak scan: 5/5 clean (`leak_check_r2.jsonl`).
- Agent 1 (V3) disagreed with System 1's top candidate on 49/92 questions. Agent 2 ran on those 49, and 45 were
  decided by a 2-of-3 majority.
- **Routed EX 72.83, −3.26 vs V0.** Cost 1.54× V0, so the projected saving is **1.9%**; projected test EX 79.9.
- **Not selected. Round 2 fails both targets.**
- **Why, descriptively.** On routed questions System 1's top candidate is usually wrong (it is right on about 37% of
  routed questions on the fresh set). Counting it as one of three votes pulls the final answer toward it whenever the
  two agents split.

**Conclusion under this pre-registration.**
- None of the tested levers (answer-first, top-3 candidates, in-agent second attempt, an independent second agent with
  a majority vote) raised the routed half's accuracy on the development half.
- The projection stays about **81.6 at a 35% saving**.
- Showing the top 3 candidates instead of 8 cut the routed half's cost by about 20% at equal development accuracy.
  This is exploratory and unconfirmed; if it holds it would mean about **81.6 at about 45% saving**.
- G (stricter gate) is still pending the external drive.

## RESULT: G (stricter gate), offline (2026-10-22, drive mounted)
Scored with `gate_eval.py` → `gate.json`.
- G newly routes 57 of the 171 kept questions.
- On those questions System 1 was already right on 47 (23 development, 24 confirmation); B1's answers get 48.7 on
  average.
- **EX change:** +0.39 points (development) and +0.57 (confirmation).
- **Cost:** routed share rises from 0.53/0.49 to 0.69/0.66, so the saving at V0 cost falls to 13% (development) and
  17% (confirmation).
- **Not selected:** the saving falls below 30% for under 1 point. The rule's "fewer than 8 of 32" condition mostly
  catches questions System 1 already gets right.

## FINAL (this pre-registration)
- No tested lever reaches projected test EX ≥ 83 while keeping the saving ≥ 30%. We stop tuning on RESERVED.
- **Operating point that stands:** projected about 81.6 (range 79.5–84.2) at the measured 34–35% saving vs B1.
- **Exploratory, unconfirmed:** top-3 candidates suggest about the same accuracy at about 45% saving.

## CORRECTION (2026-10-22)
- **Bug.** `score_variants.py` costed V0 over all 177 renamed routed questions but each variant over its half only. The
  development half is cheaper than average, so variant cost ratios were understated. Fixed: V0 is now costed on the same
  half. `dev.json` is rescored.
- **Corrected development half.** V1 ratio 1.093 (saving 29.3%); V2 0.924 (39.6%); V3 0.886 (42.0%). Accuracy
  unchanged.
- **The "top-3 candidates cut cost about 20%" observation was mostly this artifact.** Corrected it is about 11% on the
  development half. On the held-out confirmation half V3 cost 14% *more* than V0 and lost 5.9 points
  (`experiments/final_config/step2_confirm.json`).
- The round-1 and round-2 selection decisions do not change: no variant gained ≥ 2 points.
- Round 2 (`dev_r2.json`) rescored with the fixed scorer on 2026-10-22; the corrected numbers are in that file.
