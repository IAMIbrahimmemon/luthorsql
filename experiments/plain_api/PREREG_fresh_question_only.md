# Question-only cascade through a plain API agent, fresh unseen set (pre-registered 2026-10-26)

**Why.** In the registered plain-API test on RESERVED (`PREREG.md`, run 1), showing the agent System 1's candidates
nearly doubled the cost of each routed question and added no accuracy, so the registered cascade cost 18.3% more than
Opus alone. Forwarding only the question was computed afterwards on the same answers (exploratory: 35.3% cheaper,
-2.0 points). This test fixes that design in advance and measures it on data no plain-API agent has seen.

## Fixed before any run

**Questions.** The fresh unseen set: all 227 items (`data/sys1/fresh/items_final.jsonl`, 8 databases published after
Opus 5.5's training cutoff; `experiments/fresh_unseen/PREREG.md`). Gold as in `experiments/fresh_unseen/score_fresh.py`.

**System 1.** The frozen two-adapter outputs already computed (`runs/phase2/fresh_s1/s1_fresh.jsonl`). Routing at
t = 3.0 (margin < 3.0 or missing): 125 of 227 routed. Not tuned: 3.0 is the registered threshold.

**Agent.** Exactly the RESERVED plain-API agent alone (arm A0 of `experiments/plain_api/run.py`): Claude Opus 5.5
through the Messages API, one conversation per question, the paper's API-agent prompt without candidates, the schema
in the prompt, the read-only SQL tool (databases `data/sys1/fresh/dbs`), at most 8 executions and 14 turns, default
sampling, no extended thinking, prompt caching as in `PREREG.md`. It answers all 227 questions (`--arms A0`).

**Arms.**
- **A0, Opus alone:** the agent's answer on every question.
- **Q, question-only cascade:** System 1's pick on the 102 kept questions; the agent's answer on the 125 routed ones
  (each question is its own conversation with the identical prompt, so this is what a deployed cascade sends).

**Cost.** Logged API usage at Opus 5.5 list prices ($4 / $5 / $0.20 / $20 per M). Q's cost is A0's cost on the
routed questions; System 1 is $0 (owned hardware).

**Endpoints.**
- Primary: cost saving of Q against A0, with a 95% database-bootstrap interval (10,000 resamples, seed 20261026).
- Co-primary: EX of Q minus EX of A0 (database sign-flip test, `luthor.stats.cluster_signflip`).
- Also: EX of A0, Q and System 1; cost per question; tokens; routed and kept accuracy.

**Reporting.** Both endpoints are reported together, whatever they are: the paper states the saving with the accuracy
difference beside it.

**Prediction (before any call).** Saving 30–45% (routed questions are the harder, costlier ones, so below the 55%
kept share); accuracy about 1–4 points below A0 (System 1 was right on 96.1% of the questions it kept on this set in
the Claude Code run).

**Budget.** One run, stopped by the runner if it passes $10 (expected about $2–3). No reruns unless a run crashes.
