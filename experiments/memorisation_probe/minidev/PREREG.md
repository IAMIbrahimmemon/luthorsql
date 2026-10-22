# Memorisation probe on BIRD Mini-Dev (SQLite) (pre-registered 2026-10-21)

**Context.** A third-party leaderboard PR (bird-bench.github.io #251, "schemagate + Claude Opus 5.5") reports Mini-Dev
SQLite EX 81.00 (re-run 80.20) when Opus is given the table DDL and the evidence. Our first probe (`../PREREG.md`)
found that Opus 5.5 recalls gold SQL for BIRD train, KaggleDBQA and Spider-Realistic. This probe asks the same
question of Mini-Dev, which is drawn from BIRD dev, public since 2023.

## Fixed before any run
- **Data.** `mini_dev_sqlite.json` from HF `birdsql/bird_mini_dev` (500 questions, downloaded 2026-10-21, sha256 in
  `data.sha256`). Gold: its `SQL` field. Databases: BIRD dev (`data/bench/dev_20240627/dev_databases`).
- **Model and harness.** Same as the first probe: Opus 5.5 Claude Code subagents, one Read and one Write each, no
  database, no schema, audited from transcripts. Chunk names are neutral (`md1_NN`, `md2g_NN`, `md2n_NN`).
- **P1, schema-free SQL.** Input is the question text and db_id only (no evidence). Scored by EX against the Mini-Dev
  gold (read-only SQLite, 30 s, set equality). We also report how often the gold SQL is reproduced as a string after
  normalisation, and the execution error rate.
- **P2, guided vs general completion.** 100 questions with at least 8 words, sampled with seed 20261021. The guided
  agent is told "This is a question from the BIRD dataset (dev split, also released as BIRD Mini-Dev)"; the general
  agent gets the same instruction text as the first probe. Scored by ROUGE-L and exact match; paired sign-flip test.
- **Prediction.** P1 EX ≥ 70 (within about 11 points of the with-DDL 81.00), and P2 guided − general ROUGE-L > 0.2
  with p < 0.05.
- **Reading.** If P1 EX ≥ 70, most of schemagate's 81 is reachable without the schema, so it cannot be read as a
  measure of schema linking. Below 50, recall explains little of it. In between, partial. We report the number whatever
  it is. Nothing is posted to that PR.

## VERDICT (2026-10-21)
Scored with `score_md.py` → `result.json`. Compliance: 14/14 transcripts clean; no id mismatches.

**P1 (no schema, no evidence).**
- EX **86.20** (simple 92.57, moderate 85.60, challenging 78.43).
- 0 of 500 queries failed to execute.
- 292/500 (58.4%) reproduce the gold SQL as a string.

**P2.**
- Guided ROUGE-L 0.965 vs general 0.568, guided − general +0.398, p = 0.0001.
- Exact completions: 91/100 guided vs 25/100 general.

**Predictions.** P1 ≥ 70: met. P2 gap > 0.2 with p < 0.05: met.

**Cross-tab against schemagate (PR #251), descriptive.** Their prediction files were re-scored with our executor
against the same gold and the BIRD dev databases. Our executor reads about 2.4 points below their official
evaluator: Opus 78.6 here vs 81.0 reported. Two gold queries also failed to execute in our setup; Mini-Dev ships its
own copy of the databases, and we used `dev_20240627`. Every comparison below uses one executor, so they are
internally consistent.

| their run | EX (ours) | their correct | also correct blind | share |
|---|---|---|---|---|
| Opus 5.5 | 78.6 | 393 | 377 | 95.9% |
| Opus 5.5 re-run | 78.2 | 391 | 375 | 95.9% |
| Haiku 5.5 | 60.2 | 301 | 287 | 95.3% |
| Gemini 2.5 Flash | 62.4 | 312 | 299 | 95.8% |

**Reading.**
- With no schema and no evidence, Opus 5.5 scores higher on Mini-Dev SQLite (86.2) than the same model inside
  schemagate with DDL, column descriptions and evidence (78.6 on the same executor).
- 96% of the questions schemagate-Opus answers correctly, blind Opus also answers correctly.
- Mini-Dev SQLite results for Opus 5.5 therefore cannot be read as schema-linking or text-to-SQL ability.
- This applies to any Opus 5.5 entry on public BIRD splits, not only schemagate; the #1 SQLite entry is also reported
  to run on Opus 5.5.
- The same overlap share for Haiku and Gemini (about 95%) is expected: those are mostly easy questions that blind Opus
  also gets. It says nothing about whether Haiku or Gemini memorise.
- Nothing has been posted to the PR.

## Amendment: third-party source not trusted (2026-10-22, owner review)
PR bird-bench.github.io #251 is unmerged and self-reported, and the owner does not consider its author a reliable
source. Therefore:
- **Not to be cited or relied on:**
  - its reported EX (81.00 / 80.20);
  - the per-question cross-tab against its prediction files (the "96% also correct blind" figure and the 78.6 re-score),
    because nothing shows those files came from the described pipeline;
  - the statement, taken from its README, that the #1 Mini-Dev SQLite entry uses Opus 5.5 (unverified).
- **Unaffected and citable:** P1 schema-free EX 86.20 on the official HF `birdsql/bird_mini_dev` SQLite file, the
  gold-string reproduction, and P2 guided completion. These use only official data, our agents and our scorer.
- The paper will not mention PR #251.
