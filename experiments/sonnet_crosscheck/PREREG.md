# Sonnet as System 2: same measurements as Opus on all 796 DECISION questions (pre-registered 2026-10-18)

Owner (2026-10-18): "run some parallel sonnet agents right now and measure all the same things we did with opus."
Kept with the other cross-checks, outside docs/.

- Same setup as Opus: Claude Code subagents (model `sonnet`), 25-question chunks, prompts
  `runs/analysis/hybrid/opus_s1_raw/PROMPT_alone.md` (A0) and `PROMPT_s1.md` (A1) unchanged, rebuilt tool
  `scripts/system2/sql_tool.py`, sandbox outside the repo with the 8 DECISION DBs, candidates
  `runs/analysis/hybrid/s1_cands_decision.jsonl`. 32 chunks per arm. Leak check `scripts/system2/leak_check.py`.
- **No BIRD dev run**: scoring Sonnet on dev would spend dev look #3, kept for the final system.
- Measured (all 796, original gold): A0, A1, System 1; A1 − A0, A1 − S1 (exact DB sign-flip); cascade curve on the
  0.25 grid; frozen Opus rule (max $1 × EX − cost, cost = Opus's $0.0278 and Sonnet's own measured cost); loss at
  t = 3.0; per-band gains; cost per question.
- Predictions: t* ∈ [2.5, 3.5]; loss at t = 3.0 ≤ 1.0; Sonnet A1 below Opus A1 (78.89) by 0–5 points.

## VERDICT (2026-10-18; `result.json`, scored with `experiments/ollama_crosscheck/score.py`)
- Run: 64 chunks, 796/796 answered in both arms. One A1 chunk (c10) wrote 23 of 25 lines; the 2 missing questions
  were answered by one extra agent (`c10b`) — a missing answer, not a wrong one, so filled rather than scored wrong.
- Leak check: 65 transcripts, 16 flagged, all read: agents wrote scratch scripts to /tmp (against the instruction to
  use their work folder) or read their own background-command output. No repo, gold, web or other-chunk access; no
  answer file holds another chunk's question or a duplicate. Recorded as a protocol deviation, no rerun.
- All 796: System 1 58.92; **Sonnet alone 71.11; Sonnet + System 1 74.50** (Opus + System 1: 78.89).
  A1 − A0 = +3.39 (sign-flip CI [−0.35, +7.93]).
- **Cascade: the frozen rule with Opus's cost picks t = 3.0 for Sonnet too.** Loss at 3.0 vs always-handoff −0.13
  (sign-flip CI [−0.75, +0.84]) at 54.7% of the calls (Opus: −0.50). Curve flat from 2.25 to 6.0 (73.99–74.62).
  With Sonnet's own cost ($0.0199 per handoff question) the rule picks 4.0, ahead of 3.0 by 2 questions (a tie).
- Per-band gain of Sonnet + S1 over S1 above margin 3.5: −1.2 to +1.3 points (prediction ≤ 5: right).
- Predictions: t* ∈ [2.5, 3.5] right (3.0); loss at 3.0 ≤ 1.0 right; Sonnet A1 below Opus by 0–5 right (−4.39).
- Cost (list price, assumed Sonnet $3/M in, $3.75/M cache write, $0.30/M cache read, $15/M out — check current
  pricing): A0 $0.0143, A1 $0.0199 per question.
