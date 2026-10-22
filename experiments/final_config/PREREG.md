# Final cascade configuration C* (pre-registered 2026-10-22, before any run)

**Configuration C*.**
- **Routing.** System 1 (the BIRD-submitted package) answers alone unless one of these holds:
  - its margin is < 3.0 or absent (the current rule), or
  - G: its top group returns 0 rows, a first row that is all NULL, or fewer than 8 of the 32 samples.
- **Routed agent.** Opus 5.5 with `PROMPT_agent.md` (the round-1 V3 prompt: second attempt on disagreement), shown only
  the top 3 candidate groups. The tool, the 8-execution limit and the chunking (≤ 25 questions, one database per chunk)
  are unchanged.

**Step 2: held-out confirmation of top-3 (renamed RESERVED, confirmation half: cs_semester, hockey, movie,
student_loan; 85 routed questions).**
- V3 agents on the confirmation chunks, scored by `experiments/top3_at_30/score_variants.py --half confirm` against V0.
- **Pass** if the cost ratio vs V0 is ≤ 0.90 and EX ≥ V0 − 2.0 points.

**Step 3: full RESERVED end-to-end run of C* (original databases, all 348 questions).**
- Routed set: margin < 3 or G (expected 234 questions).
- Scored like `experiments/cost_confirmation/score.py`: EX against BIRD-train gold; Opus 5.5 list prices; System 1's
  GPU cost on every question.
- **Comparators** (existing runs 1–3, pooled): B1 (Opus + 8 candidates on every question) for cost; C (the current
  cascade) for accuracy.
- **Primary targets:**
  1. saving of C* vs B1 ≥ 30% (point estimate; DB-bootstrap 95% CI reported);
  2. C* EX ≥ mean C EX over runs 1–3 − 2.0 points (non-inferiority; DB sign-flip test of C* vs each run reported).
- **Projection:** BIRD-test EX of C* = 81.6 + (C* EX − mean C EX), with range ± 2.5.

**Step 4: fresh unseen set (227 items).**
- C* routing, using the fresh set's own System 1 margins and candidates, and the C* agent.
- Reported against the fresh set's C (94.27) and B0 (97.36), with saving vs B0 (no B1 run exists there).
- Descriptive; no pass rule.

**Decision.** C* goes forward to the BIRD submission build (step 5) only if steps 2 and 3 pass. Otherwise we report the
numbers and keep the current configuration (about 81.6 at 35%).

## VERDICT (2026-10-22): C* FAILS. Keep the current configuration.
Leak scan: 28/28 transcripts clean (`leak_check.jsonl`).

**Step 2, confirmation half, renamed (`step2_confirm.json`, corrected scorer).**
- V3 + top 3 scored 63.53 vs V0 69.41 (**−5.88**), at a cost ratio of **1.141** vs V0.
- **FAIL** (needed ratio ≤ 0.90 and EX ≥ V0 − 2).

**Step 3, full RESERVED end-to-end (`result_reserved.json`).**
- Routed 234/348.
- C* EX **80.17** vs C runs 81.32 / 80.46 / 79.60 (mean 80.46): **−0.29**, non-inferior (pass).
- Cost $0.0197/q vs B1 $0.0244/q: saving **19.1%** (DB-bootstrap 7.8–31.0%). **FAIL** (needed ≥ 30%).
- Same accounting for the current C: 29.9 / 36.7 / 36.2% (runs 1–3).

**Step 4, fresh set (descriptive).**
- C* EX 92.95, vs C 94.27 and B0 97.36. Routed 72.2% (C: 55.1%); routed EX 91.46.

**Decision (as pre-registered).**
- C* does not go forward. The BIRD submission uses the current cascade: t = 3.0, 8 candidates, `PROMPT_s1.md`.
- That configuration projects to about **81.6** BIRD-test EX (79.5–84.2) at a **30–37%** measured saving vs B1.
