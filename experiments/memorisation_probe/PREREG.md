# Memorisation probe: Spider-Realistic vs KaggleDBQA vs RESERVED (pre-registered 2026-10-21)

Owner, 2026-10-21: Spider-Realistic is likely memorised by Opus 5.5. This probe measures that directly. The external
test's pre-registered verdict (CLAIM, pooled over both sets) stands whatever this shows. The probe only adds evidence
for how to read the per-set results.

## Fixed before any run
**Model.** Claude Opus 5.5 as Claude Code subagents with no database and no SQL tool. Each agent only reads its chunk
file and writes its output file.

**P1: schema-free SQL.**
- Input per question: the question text and the database name only. No schema, sample rows, evidence or tool.
- Instruction: write the SQLite query that answers the question on that database.
- Scoring: EX against gold on the real database, using the scorers of the cost tests (30 s, set equality).
- Items: all 185 KaggleDBQA, all 508 Spider-Realistic and all 348 RESERVED questions, in chunks of up to 50 from
  one set.
- Reading: success without the schema requires knowing the real table and column names and the intended answer, a
  memorisation signal. Some guessable schemas (e.g. `singer`) inflate it, so the comparison is between sets.

**P2: guided vs general completion (Golchin & Surdeanu style).**
- Items: 100 questions per set, sampled with seed 20261021, at least 8 words each.
- Each question is cut after floor(n_words / 2) words. The agent sees the first half and writes the rest.
- **Guided condition:** "This is a question from the <dataset name> dataset (<split>). Complete it exactly as it
  appears in the dataset."
- **General condition:** "Complete this database question naturally."
- The two conditions run in separate agents, so no agent sees both for the same item.
- Scoring: ROUGE-L F1 (word level, lowercase) and exact match after normalisation, against the true second half.
- Memorisation signal: guided − general. Tested with a paired sign-flip per set, 10,000 flips, seed 20261021.

**Predictions.**
- P1 EX: Spider-Realistic > KaggleDBQA by ≥ 15 points. RESERVED (BIRD train) lies in between.
- P2: Spider-Realistic shows a positive guided − general ROUGE-L gap (p < 0.05). KaggleDBQA shows a gap under 0.05.

**Reporting.** All numbers for all three sets. If Spider-Realistic shows the signals, the paper:
- labels its per-set result "likely contaminated";
- reports the pre-registered pooled result unchanged;
- reports KaggleDBQA alone as the cleaner external estimate.
No result is removed.

## Amendment 1 (2026-10-21, before any probe run and before any per-set look at agent effort)
- **Blinding.** Chunk files have neutral names (`p1_01`, …), so no agent sees a benchmark name in a path. P2-general
  chunks hold only `item_id` and `first_half`. The mapping is in `chunk_map.json`.
- **Secondary, descriptive.** Prompted by the owner's hypothesis that memorisation also explains the external test's
  time result: from the external test's existing B0 transcripts, report per set the tool calls, output tokens and
  seconds per question, Spider-Realistic vs KaggleDBQA.
- **Prediction.** All three are lower on Spider-Realistic.
- **Caveat.** This is confounded with difficulty: Spider databases and questions are simpler. It cannot show
  memorisation alone and is read only alongside P1/P2.

## VERDICT (2026-10-21)
Run and scoring: `score_probe.py` → `result.json`; `effort_ext.py` → `effort_ext.json`.
- **Compliance.** 34/34 transcripts clean: one Read of the agent's own chunk and one Write each.
- **Id mismatch.** One answer line carried the wrong id (`p2g_01`, labelled `kaggledbqa_test_49` for `_64`; the agent
  reported this itself). Answers are aligned to items by line position, applied the same way to every chunk.

**P1: schema-free SQL (no schema, sample rows, evidence or tool).** No query failed to execute (0 errors in 1,041).
"Gold string" means the gold SQL reproduced after whitespace and quote normalisation.

| set | n | EX | gold string | with-DB B0 EX (cost tests) |
|---|---|---|---|---|
| Spider-Realistic | 508 | 94.69 | 469 (92.3%) | 92.72 |
| KaggleDBQA test | 185 | 79.46 | 98 (53.0%) | see external result.json |
| RESERVED (BIRD train) | 348 | 79.31 | 168 (48.3%) | 80.17 (run 1) |

The reproductions include dataset-specific details no schema-free guess would produce: BIRD's `T1/T2/T3` alias style,
backticked column names such as `` `Character Name` ``, and exact join paths.

**P2: guided vs general completion of the question's second half.**

| set | ROUGE-L guided | ROUGE-L general | guided − general | p | exact guided / general |
|---|---|---|---|---|---|
| Spider-Realistic | 0.670 | 0.558 | +0.113 | 0.0001 | 22 / 16 |
| KaggleDBQA test | 0.798 | 0.295 | +0.503 | 0.0001 | 66 / 4 |
| RESERVED (BIRD train) | 0.895 | 0.364 | +0.532 | 0.0001 | 77 / 8 |

**Predictions.**
- P1 Spider-Realistic > KaggleDBQA by ≥ 15: **met** (+15.23).
- RESERVED in between: **not met** (79.31, just below KaggleDBQA).
- P2 Spider-Realistic gap > 0 with p < 0.05: **met**.
- P2 KaggleDBQA gap < 0.05: **not met** (+0.50). KaggleDBQA is not clean.
- Secondary (B0 effort lower on Spider-Realistic): **met** on all three measures (tool calls 0.25 vs 0.38 per question,
  output tokens 48 vs 74, seconds 1.38 vs 2.49), but confounded with difficulty.

**Reading.**
- Opus 5.5 has memorised all three evaluation sets, not only Spider-Realistic. On each one, its schema-free accuracy
  equals its with-database accuracy within about 2 points. The System 2 baselines (B0, B1, B2) on these sets therefore
  measure recall, not text-to-SQL.
- On RESERVED, Spider-Realistic and KaggleDBQA, every accuracy comparison against Opus is against a recalling oracle.
  That biases non-inferiority against the cascade, since System 1 cannot recall. Agent cost and time are also lower
  than they would be on unseen data, because recall needs little database exploration (B0 makes fewer than 0.4 tool
  calls per question).
- The planned KaggleDBQA "cleaner estimate" is withdrawn. Neither external set is clean.
- An unseen evaluation set (planned: databases published after the model cutoff, with non-Claude question and SQL
  authors) is now required for any accuracy claim about System 2. The BIRD hidden test is the other clean check.
- No result is removed. The pre-registered verdicts of both cost tests stand as written.
