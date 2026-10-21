# Haiku as System 2: same measurements as Opus on all 796 DECISION questions (pre-registered 2026-10-19)

Owner (2026-10-19): "can you test the samething on haiku 5.5 now."
Kept with the other cross-checks, outside docs/.

- Same setup as Opus: Claude Code subagents (model `haiku` (Haiku 5.5)), 25-question chunks, prompts
  `runs/analysis/hybrid/opus_s1_raw/PROMPT_alone.md` (A0) and `PROMPT_s1.md` (A1) unchanged, rebuilt tool
  `scripts/system2/sql_tool.py`, sandbox outside the repo with the 8 DECISION DBs, candidates
  `runs/analysis/hybrid/s1_cands_decision.jsonl`. 32 chunks per arm. Leak check `scripts/system2/leak_check.py`.
- **No BIRD dev run**: scoring Haiku on dev would spend dev look #3, kept for the final system.
- Measured (all 796, original gold): A0, A1, System 1; A1 − A0, A1 − S1 (exact DB sign-flip); cascade curve on the
  0.25 grid; frozen Opus rule (max $1 × EX − cost, cost = Opus's $0.0278 and Haiku's own measured cost); loss at
  t = 3.0; per-band gains; cost per question.
- Predictions: t* ∈ [2.5, 3.5]; loss at t = 3.0 ≤ 1.0; Haiku A1 below Sonnet A1 (74.50); the loss at t = 3.0 may be positive (as for the weaker open models).

## VERDICT (2026-10-19; `result.json`, scored with `experiments/ollama_crosscheck/score.py`)
- Run: 64 chunks, 796/796 answered in both arms, no missing or foreign answers, no reruns.
- Leak check (`leak_check.jsonl`): 64 transcripts, 5 flagged, all read, all false positives: "gold" in an agent's own
  note, `/answers/` (its own output path; one agent put a helper script in `answers/A0/`), its own work folder created
  one level up, and one agent reading the output file of its own auto-backgrounded command (and an empty `/tmp/x`).
  No repo, gold, web or other-chunk access.
- All 796: System 1 58.92; **Haiku alone 68.34; Haiku + System 1 72.61** (Sonnet 74.50, Opus 78.89).
  A1 − A0 = +4.27 (sign-flip CI [+0.99, +7.53]); A1 − S1 = +13.69.
- **Cascade: loss at t = 3.0 vs always-handoff 0.00** (sign-flip CI [−0.57, +0.84]) at 54.7% of the calls. The curve
  is flat from 2.0 to 6.0 (72.24–72.61). The frozen rule with Opus's cost picks 2.25, and with Haiku's own cost
  ($0.0073 per handoff question) also 2.25; 2.25 and 3.0 give the same EX (72.61), the rule just prefers fewer calls.
- Per band, Haiku + S1 gains over S1 only below margin 2.5 (+10 to +47 points); from 2.5 up it is −1.2 to +1.2.
- Predictions: t* ∈ [2.5, 3.5] **wrong** (2.25, on a flat curve: same EX as 3.0); loss at 3.0 ≤ 1.0 right (0.00);
  Haiku A1 below Sonnet A1 right (−1.89); loss at 3.0 positive: no (exactly 0).
- Cost (list price, assumed Haiku $1/M in, $1.25/M cache write, $0.10/M cache read, $5/M out — check current
  pricing; output estimated as characters / 4): A0 $0.0070, A1 $0.0073 per question.
