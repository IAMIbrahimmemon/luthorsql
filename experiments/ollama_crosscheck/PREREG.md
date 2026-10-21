# Is the System 1 → System 2 cascade Opus-specific? GLM 5.3 Flash and DeepSeek 4.1 Flash as System 2 (pre-registered 2026-10-18)

Kept separate from docs/ and the paper notes on purpose (owner, 2026-10-18): "test it with Luthor SQL and see ... if
we're seeing the exact same type of situation ... that same threshold number is the right number ... make sure what
we're doing is not specific to Opus ... Keep this separate from the rest of the docs." Models (owner): GLM 5.3 Flash
and DeepSeek 4.1 Flash, **each tested separately** as System 2 (no combination of the two).

## Setup (frozen before the full run)
- Questions: all 796 DECISION questions (8 held-out BIRD-train DBs, original gold), same as the Opus threshold choice
  (`runs/phase2/decision_rest/`). No BIRD dev question is used (no dev look).
- System 1: the same frozen two-adapter System 1 (candidates `runs/analysis/hybrid/s1_cands_decision.jsonl`, top 8
  result groups, label-free; margin `scripts/analysis/cascade_threshold.py:margin`).
- System 2 runner: `scripts/system2/api_runner.py` (Ollama Cloud `/api/chat`, temperature 0, one question per
  conversation, schema in the first message, `run_sql` with the 8-execution limit enforced in code, `submit` tool,
  14 turns max). Models `glm-5.3-flash` and `deepseek-v4.1-flash`.
- Arms per model: **A0** model alone; **A1** model + System 1 handoff (pick a candidate or write own SQL).
- Differences from the Opus runs (stated, not hidden): API loop instead of Claude Code subagents; one question per
  conversation instead of 25-question chunks; schema given up front instead of fetched with a tool call; limit
  enforced by the tool instead of by instruction.

## Scoring and the questions asked
Strict EX vs original BIRD-train gold, 30 s. Per model, on all 796:
1. A0, A1, System 1 alone; A1 − A0 and A1 − System 1 (exact DB sign-flip, 8 DBs).
2. The cascade curve on the 0.25 grid 2.0–6.0 (System 1 if margin ≥ t, else A1) and the per-band gain of A1 over
   System 1 (same bands as the Opus table).
3. **Same threshold?** The frozen Opus rule (max $1 × EX − cost) applied with Opus's cost ($0.0278 per handoff
   question) so only the accuracy pattern differs → t*(model). Also: the loss at t = 3.0 vs always-handoff, and the
   smallest t within 0.5 points of always-handoff.
4. **Same savings?** Opus share and loss at t = 3.0 for each model vs Opus (54.7%, −0.50).

## Predictions (falsifiable)
- Opus share at t = 3.0 is identical for all models (54.7%: it depends only on System 1).
- For each model, A1 > System 1 by ≥ 10 points and A1 ≥ A0 − 2.
- t*(model) ∈ [2.5, 3.5] and the loss at t = 3.0 is ≤ 1.0 point for each model ("not Opus-specific").
- The per-band gain above margin 3.5 is ≤ 5 points for each model.
A model that fails the last two shows the cascade threshold is model-dependent.

## VERDICT (2026-10-19; `result.json`, `score.py`)
- Run: 796/796 answered in every arm, 0 unsubmitted. A third model, **GLM 5.3** (non-flash), was added by the owner
  on a second key; that key ran out of credit part-way through A1 and the rest ran on the first key (resume skips
  finished ids; one answer per id kept). The first key was held to ≤ 2 concurrent requests (owner's limit), except a
  brief 3-request probe while diagnosing a stall. API models see only `run_sql` on the sandbox DBs (no file system),
  so no transcript leak check applies.
- All 796, System 1 58.92:

  | model | A0 alone | A1 + S1 | A1 − S1 | t = 3.0 cascade | cascade − A1 (DB sign-flip 95% CI) | rule t* |
  |---|---|---|---|---|---|---|
  | DeepSeek 4.1 Flash | 65.95 | 69.22 | +10.30 | 70.23 | +1.01 [+0.36, +2.03] | 2.0 |
  | GLM 5.3 Flash | 65.95 | 68.47 | +9.55 | 68.84 | +0.38 [−0.63, +1.56] | 2.0 |
  | GLM 5.3 | 66.21 | 70.48 | +11.56 | 70.60 | +0.13 [−0.57, +1.19] | 2.0 |
  | Opus (reference) | — | 78.89 | +19.97 | 78.39 | −0.50 | 3.0 |

- Every curve is flat from 2.0 to 4.0 (spread ≤ 0.3 points), then drops slightly as more high-margin questions go to
  the weaker model. The rule picks 2.0 (grid floor) for all three because the curve is flat and fewer calls win; 3.0
  is within 0.26 of 2.0 for each.
- Per band, gains over System 1 come only below margin 2.5 (+3 to +43 points). From 3.5 up the open models are level
  with or worse than System 1 (−3.7 to +1.2): handing confident questions to them hurts, which is why the cascade
  beats always-handoff.
- Predictions: share 54.7% for all **right**; A1 > S1 by ≥ 10 right for DeepSeek and GLM 5.3, **wrong** for GLM 5.3
  Flash (+9.55); A1 ≥ A0 − 2 right (all three); t* ∈ [2.5, 3.5] **wrong** (2.0, flat curve; 3.0 equally good); loss at
  3.0 ≤ 1.0 right (all three gain); band gain above 3.5 ≤ 5 right. **The threshold is not Opus-specific**: with
  every model tested (also Sonnet and Haiku, `../sonnet_crosscheck`, `../haiku_crosscheck`), t = 3.0 costs at most
  0.5 points against always-handoff, and with weaker models it is better than always-handoff.
