# Renamed-schema test of the routed half (pre-registered 2026-10-21)

**Why.** The memorisation probes (`experiments/memorisation_probe/`) show that Opus 5.5 recalls the gold SQL of BIRD
train/dev questions. The cascade's routed-half accuracy on RESERVED (72.3%, run 1) therefore cannot be used to project
its BIRD hidden-test score. This test measures Opus on the same routed questions with verbatim recall blocked.

## Fixed before any agent run
**Items.**
- The RESERVED questions the cascade routes at t = 3.0 (margin < 3 or no margin): 177 questions. System 1 rows come
  from `runs/phase2/cost_confirmation/s1_reserved.jsonl`.

**Transformation, built without Claude.** A non-Claude model writes all new text: `deepseek-v4.1-flash:cloud` via
Ollama, on public BIRD-train material only.
- **Database ids.** Each becomes a neutral id `db_NN`.
- **Tables and columns.** Every one gets a meaning-preserving snake_case name that shares no word of 3+ letters with
  the original. A name breaking that rule is replaced by a neutral `t_k` / `c_k`; the count is reported.
- **Database copies.** Built with SQLite `ALTER TABLE … RENAME`, so the data is unchanged. Description CSVs are renamed
  and rewritten with the same mapping; descriptions are kept.
- **SQL.** Gold SQL and System 1's candidate SQL are rewritten with sqlglot, resolving every column to its table.
- **Checks.**
  - A rewritten gold must return exactly the original gold's result on the renamed copy; otherwise the item is excluded
    before any run.
  - A rewritten candidate must return its original result; otherwise that candidate is dropped.
  - Result previews are recomputed on the renamed copy.
- **Question and evidence.** The question is paraphrased and the evidence rewritten with the new names. Every literal
  (quoted strings, numbers) must survive; after 3 failed attempts the original wording is kept and counted.
- **Reporting.** Exclusions and fallbacks are reported in full.

**Agent arm.**
- The cascade's routed arm exactly: prompt `PROMPT_s1.md` verbatim, the same wrapper text, the same tool
  (`TOOLS/sql.py`, read-only, 8 executions per question), chunks of up to 25 questions from one database, Opus 5.5
  Claude Code subagents.
- Sandbox: the renamed DBs and the tool only.

**Endpoints.**
- **Primary.** EX of the renamed arm on the eligible routed items, against the original routed-arm EX on the same items
  (run 1). Paired difference; DB-cluster sign-flip test and 10k DB-cluster bootstrap CI.
- **Secondary.** The same comparison against the mean of runs 1–3. Pick vs own-SQL rates. Tool calls per question.
- **Projection update.** Projected cascade test EX = kept share × kept accuracy + routed share × renamed EX, with kept
  share 0.49 and kept accuracy 90.6 from RESERVED, as a range using the renamed EX's 95% CI.

**Prediction.** Renamed EX below the original. We expect 55–68 if the original is substantially inflated by recall.

**Caveat, stated now.** Renaming removes verbatim recall of names and SQL text. It does not prevent recall of a query's
structure (joins, filters, values) from a reworded question. The result is therefore an upper bound on recall-free
accuracy.

## VERDICT (2026-10-21)
Scored with `score_ren.py` → `result.json`. Leak scan: `leak_check.jsonl`.
- **Leak scan.** 10/11 transcripts clean. `C/db_03_2` was flagged for the pattern "Gold". The match is the agent's own
  tool-call description ("Gold-style q7774") on an ordinary read-only query inside the sandbox; no file outside the
  sandbox was touched. Recorded as an access false positive. The wording suggests the agent was reasoning about
  BIRD's gold-SQL conventions, which is relevant to the caveat below.
- **Answers.** 177/177 answered: 91 picked a candidate, 86 wrote their own SQL.

| | EX on the 177 routed questions |
|---|---|
| Renamed (this test) | **72.88** (DB-bootstrap 95% CI 68.91–78.05) |
| Original, run 1 | 72.32 |
| Original, runs 1/2/3 | 72.32 / 70.62 / 68.93 (mean 70.62) |

- **Renamed − run 1.** +0.56 points (DB sign-flip p = 1.0, CI −7.14 to +6.02, 8 DBs).
- **Renamed − mean of runs.** +2.26 (p = 0.56).

**Prediction** (renamed below original): **not met.** With the database, the tool and System 1's candidates, Opus's
routed-half accuracy does not fall when names, SQL text and question wording are changed.

**Reading.**
- The memorisation probes show that Opus can recall gold SQL from the question alone. This test shows that in the
  cascade's routed arm, where it has the schema and the tool, removing verbatim recall costs nothing measurable.
- The pre-registered caveat stands. Recall of a query's structure from a lightly reworded question is not ruled out, so
  72.9 is an upper bound on recall-free accuracy. The fresh unseen set is still needed to close that gap.
- **Projection** (pre-registered formula; kept share 0.49, kept accuracy 90.6 from RESERVED): cascade test EX ≈ **81.6**,
  range 79.5–84.2 over the renamed EX's 95% CI. This assumes BIRD test behaves like RESERVED, as System 1's own test
  score (68.0 vs 63.2 on RESERVED) suggests it at least does not do worse.
