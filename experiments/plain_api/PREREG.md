# Cascade cost through a plain API agent, no Claude Code harness (pre-registered 2026-10-26)

Reviewer request (2026-10-25): measure the cost comparison with the agent called through the API directly. Every
earlier Opus number in the paper ran as Claude Code subagents, whose ~46k-token per-session context (system prompt,
tool definitions) is part of each arm's cost, so part of the measured saving comes from running fewer sessions. This
experiment removes that harness.

## Fixed before any run (do not change after the first API call of this experiment)

**Questions.** RESERVED, all 348 questions (8 held-out BIRD-train databases, `data/sys1/heldout_v2/reserved.json`),
original BIRD-train gold, the same set as `experiments/cost_confirmation/PREREG.md`.

**System 1.** The frozen two-adapter LuthorSQL outputs already computed for that experiment
(`runs/phase2/cost_confirmation/s1_reserved.jsonl`: pick, margin, top-8 result groups; no gold). Routing at the
registered threshold t = 3.0: a question goes to the agent when its margin is < 3.0 or missing. That is 177 of 348.

**Agent.** Claude Opus 5.5 (`claude-opus-5-5`) through the Anthropic Messages API (`experiments/plain_api/run.py`):
- one conversation per question; the system prompt is the paper's API-agent prompt (`paper/prompts/api_loop.txt`:
  agent alone, or agent with candidates); the user message holds the schema (the same `--schema` output of
  `scripts/system2/sql_tool.py`: CREATE statements, 3 sample rows per table, column descriptions), the question, the
  evidence and, with candidates, the top-8 result groups exactly as in `s1_reserved.jsonl`;
- tools `run_sql` (read-only, the same tool) and `submit`; at most 8 executions enforced in code, at most 14 turns;
- default sampling settings, no extended thinking, `max_tokens` 4096 per turn;
- prompt caching: the system prompt and the schema block are cache breakpoints (so the schema is cached across the
  questions of one database), plus a rolling breakpoint on the latest message;
- questions are run database by database; within a database the first question runs alone (writes the cache), the
  rest with 4 workers; the two arms alternate per database.

**Arms** (Opus answers every question in both):
- **A0, agent alone:** no candidates.
- **A1, agent with candidates:** System 1's top-8 result groups shown.
- **C, cascade, t = 3.0:** System 1's pick on kept questions; A1's answer on the 177 routed questions. Because each
  question is its own conversation with the identical prompt, this is exactly what a deployed cascade would send; C
  is not run separately (the Claude Code experiment ran it separately because chunks shared one session).

**Cost.** Each call's logged `usage`, priced at Opus 5.5 list prices: $4/M input, $5/M 5-minute cache write, $0.20/M
cache read, $20/M output. C's agent cost is A1's cost on the routed questions. System 1: **$0 (owned hardware)** for
the primary numbers; the rented-H100 price used in the registered experiment ($0.00085 per question, on every
question of A1 and C) is reported as a sensitivity.

**Endpoints.**
- Primary: cost saving of C against A0 (agent alone), `1 - cost(C)/cost(A0)`, with a 95% database-bootstrap interval
  (10,000 resamples of the 8 databases, seed 20261026).
- Secondary: saving of C against A1; EX of A0, A1 and C with C − A0 and C − A1 (database sign-flip test,
  `luthor.stats.cluster_signflip`); cost per question of each arm; tokens by type; turns and executions per question.
- Reported alongside: the registered Claude Code numbers on the same questions (C vs Opus alone 24.8% / 32.9% /
  32.5% with System 1 free).

**Claim rule.** The paper may say the saving holds through a plain API agent only if the primary saving's interval
lower bound is > 0. Whatever the number is, it is reported, in the main text if it changes a claim.

**Prediction (written before any call).** Removing the harness removes a fixed per-session cost that the cascade
saved by running fewer sessions, so the saving against A0 should be smaller than the 25–33% measured in Claude Code.
Routed questions are the harder ones (more turns), so the saving should also be below the 49% of questions kept
locally. We expect roughly 10–30%. C's accuracy should be within about 2 points of A0.

**Budget.** One complete run first (both arms, 348 questions each). The runner stops a run when its summed list-price
cost passes $40. Two more complete runs (fresh outputs, same settings) only if the first costs ≤ $25 and the API
budget left after the BIRD cascade reserve allows; all completed runs are reported.

**Smoke test.** Before run 1, one question per database per arm (16 conversations, `--limit-per-db 1`, separate
directory) checks the pipeline and the cost per question. Its answers are never scored or reused; run 1 starts from
empty answer files. Only a crash or a cost above $0.15 per question would change anything (the run settings would be
fixed and this file amended before run 1).

**Not changed after a look:** questions, routing, prompts, tools, limits, model, prices, endpoints, claim rule.
