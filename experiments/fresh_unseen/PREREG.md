# Fresh unseen evaluation set (pre-registered 2026-10-21, before any data is collected)

**Why.**
- Opus 5.5 recalls the gold SQL of every public set we tried (`experiments/memorisation_probe/`).
- The renamed-schema test (`experiments/renamed_routed/`) removed verbatim recall without lowering the routed-half
  accuracy, but it cannot rule out recall of query structure.
- This set has questions and gold that cannot be in any model's training data, on databases whose contents postdate
  Opus 5.5's training cutoff (June 2026).

## Fixed now
**Databases.**
- 8–10 SQLite databases built from public data, each with at least 3 related tables, declared foreign keys, and
  description CSVs in BIRD's format.
- Each holds only records dated 2026-07-01 or later, apart from small reference tables such as teams or areas.
- Sources, licences, retrieval dates and filters are recorded in `sources.json`.
- Databases are built by Claude subagents (data plumbing only). No Claude model writes questions or gold.

**Questions and gold.**
- **Author.** `deepseek-v4.1-flash:cloud` (Ollama) writes 40 candidate items per database in BIRD style: a question, an
  evidence note and gold SQLite. The mix aims for 30% simple, 45% moderate and 25% challenging.
- **Automatic filters.** Gold must execute in under 30 s with a non-empty result of at most 1,000 rows; repeated
  execution must return the same result; near-duplicate questions are removed.
- **Independent second check.** `glm-5.3:cloud` sees the schema, question, evidence and gold SQL plus its result
  preview, and answers whether the gold answers the question exactly. Items judged NO are removed.
- **Size.** Up to 30 items per database survive, taken in generation order.
- **Human check.** The owner checks 50 items sampled uniformly (seed 20261021) before any model is run on the set. If
  more than 10% are wrong, the owner's corrections are applied to those 50 and the error rate is reported. If more than
  20% are wrong, generation is revised and re-checked before any run. Every check is logged.

**Arms.** Same harness as `experiments/external_cascade/` (prompts, tool, ≤ 25-question one-database chunks, Opus 5.5
subagents, sandbox with the databases and the tool only).
- **S1:** System 1 (the BIRD-submitted package) alone. GPU run, gold-free, routing margins recorded.
- **B0:** Opus alone (`PROMPT_alone.md`).
- **C:** cascade at t = 3.0. System 1 answers when its margin ≥ 3; otherwise the routed arm (`PROMPT_s1.md`) gets the
  question with System 1's top-8 candidate groups.

**Endpoints.**
1. **Primary.** EX of C's routed half on this set, compared with RESERVED's routed half (72.3, run 1) and the renamed
   test (72.9). DB-bootstrap 95% CI.
2. EX of S1, B0 and C overall. C − B0 with a DB sign-flip test.
3. Projected BIRD-test EX for C at t = 3.0 = kept share × kept accuracy + routed share × routed EX, all measured on
   this set, reported next to the RESERVED-based projection (81.6).
4. Cost saving of C vs B0, with the same accounting as the cost tests (secondary).

**No prediction on the direction.** We report whatever the set shows. Items are never removed after any arm has run.

## Log: database exclusions (before any questions were generated)
- 2026-10-21 `mlb_2026`: built, then excluded. The MLB Stats API terms (gdx.mlb.com/components/copyright.txt) permit
  only individual, non-commercial, non-bulk use and no redistribution, which breaks the builder spec's open-licence rule
  and would keep the set from being released. It was moved to `data/sys1/fresh/excluded/`; no question was ever
  generated on it.
- 2026-10-21 `wc2026`: the builder included the June group stage and early Round of 32 (79 of 104 matches before
  2026-07-01), against this pre-registration's recency rule. Those matches and their goals, cards and lineups were
  deleted before any question was generated, leaving the 25 knockout matches from 2026-07-01 to 07-19. The `teams` and
  `stadiums` reference tables were kept. The full build is preserved in `data/sys1/fresh/excluded/`.

## Log: generation (2026-10-21, before any model run)
- 240 items, 30 per database. `gen_stats.json` has the per-database filter counts.
- Difficulty: simple 82, moderate 92, challenging 66.
- **Judge.** `glm-5.3` rejected none of the items it judged. It acted as a weak filter, so the owner's check carries the
  quality claim.
- **Observed before any run.** The questions often name columns or codes explicitly (e.g. "ppd_category_code 'A'").
  That makes them easier than BIRD's natural phrasing, so absolute EX on this set is expected to run above BIRD.
  Comparisons within the set (routed vs kept half; arms against each other) are the point.
- Owner check sample: `human_check_ids.json` (50 ids, seed 20261021). The review page stores verdicts in the
  artifact's db.

## Amendment 1 (2026-10-21, before any model run on the set)
- **Change.** The owner's human check of 50 items is replaced by a multi-agent check the owner runs: a workflow of
  LLM verifiers, at least 2 independent verdicts per item, over all 240 items. The 50 pre-registered sample ids are
  flagged and checked first. Bundle: `fresh_verify_bundle.zip`, with instructions in its `README_VERIFY.md`.
- **Unchanged.** The thresholds apply to the agents' verdicts on the 50-item sample:
  - ≤ 10% wrong: proceed;
  - over 10%: apply corrections and report the rate;
  - over 20%: revise and re-check.
- **Removal rule.** Items judged wrong across all 240 are removed or corrected before any run, using one rule fixed
  now: an item is removed if a majority of its verdicts are "wrong". Items without a majority either way are kept.
- **Reported as.** "Gold verified by independent LLM agents (≥ 2 per item), no human check." If the verifiers are Claude
  models, the gold was judged by the same model family as the System 2 being evaluated, a possible bias toward
  Claude-style readings. This is stated as a limitation.

## Verification result (2026-10-21, before any model run)
- **The workflow.** The owner's 200-agent workflow returned 497 verdicts (`verification/verdicts.jsonl`; report in
  `verification/workflow_report.md`). Every item was judged by 2 independent blind verifiers, plus adjudicators on
  flagged items. Verifiers committed their own SQL before seeing the gold, and 23/23 planted canaries were caught.
- **Pre-registered 50-item sample.** 46 correct, 4 wrong: **8% ≤ 10%, so the run proceeds.**
- **All 240.** 13 items with a majority "wrong" verdict are removed under Amendment 1's rule
  (`verification/removed_ids.json`). 4 of those 13 were 2–1 decisions. **Final set: 227 items**
  (`data/sys1/fresh/items_final.jsonl`).
- **Owner spot check.** The owner also checked 20 items by hand and found all 20 correct. Item ids were not recorded.

## VERDICT (2026-10-22)
- **Runs.** System 1 GPU run: 227 questions in 313 s on one H100, about $0.39–0.56 spent, instance verified destroyed.
  The cascade routed 125/227 (55.1%) to the agent. 23 Opus subagent transcripts. Scored with `score_fresh.py` →
  `result.json`.
- **Leak scan.** 21/23 clean. Two flags, both benign:
  - `B0/chicago_crime_2026_2` read its own hung background command's 45-byte output file and listed its own answer file;
  - `B0/nws_alerts_2026_1` listed the file *names* in `answers/B0/`, a rule breach but no contents were read.
  No agent read any gold or any other agent's answers.

| | EX | DB-bootstrap 95% CI |
|---|---|---|
| System 1 alone | 63.88 | 58.77–68.58 |
| Opus alone (B0) | 97.36 | 95.09–99.14 |
| Cascade C, t = 3.0 | 94.27 | 90.37–97.83 |

- **Kept half (102).** System 1 96.08; B0 on the same questions 100.0.
- **Routed half (125), primary endpoint.** Cascade agent **92.80** (88.10–97.37), against RESERVED 72.32 (run 1) and
  renamed 72.88. On the same questions System 1 scores 37.60 and B0 95.20.
- **C − B0.** −3.08 points (exact DB sign-flip p = 0.0625, CI −6.02 to 0.00).
- **Cost.** Saving of C vs B0: **20.1%** (DB-bootstrap 11.6–26.6%).

**Reading.**
1. **No memorisation needed.** On questions and data no model has seen, Opus 5.5 with the database tool answers 97%.
   Its strength as System 2 does not depend on recall, consistent with the renamed-schema test.
2. **Confidence routing transfers.** System 1 is right on 96% of what it keeps and 38% of what it routes.
3. **The cascade costs accuracy here.** It trails Opus alone by 3.1 points while saving 20% of cost.
   - About 4 of the 7 net losses are System 1 errors on the kept half (96.1 vs Opus's 100).
   - The rest are on the routed half, where the agent with candidates (92.8) is below Opus without them (95.2).
4. **The pre-registered BIRD projection does not transfer.** The measured formula gives 94.3. As logged before any run,
   this set is easier than BIRD (questions name columns and codes), so absolute EX here cannot be used to project BIRD
   test EX. The set supports the relative findings (1–3), not a BIRD number. RESERVED and the renamed test remain the
   basis for the BIRD projection (~81.6, range 79.5–84.2).
