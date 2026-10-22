# Incidents during the RESERVED replicates (Amendment 2)

## r2: the B0 cs_semester_2 agent ran B2's helper script (2026-10-21)
- **What happened** (the agent's own report, then checked on disk):
  - The prompt's helper folder, `answers/../work/<chunk name>/`, does not name the arm. So B0 and B2 working on
    `cs_semester_2` shared `work/cs_semester_2/`.
  - The B0 agent ran B2's `w.py` from there twice by mistake. Each run appended B2's 17 answer lines again to
    `answers/B2/cs_semester_2.jsonl`.
- **Checked:**
  - The file has 51 lines: the same 17-line block three times, byte for byte, with 17 unique question ids.
  - The scorer keeps the first line per id, so B2's score is unchanged. The file is left as it was, as evidence.
- **Risk:** the B0 agent may have seen B2's SQL for these 17 questions.
- **Handling (decided now, before r2 is scored):**
  - r2 is reported as run.
  - A sensitivity analysis drops r2's B0 `cs_semester_2` chunk (17 questions) from the B0 comparisons.
  - Nothing is rerun, per the rerun rule.
- **Protocol note:** the shared-folder risk existed in run 1 too, where B2 `car_retails_2` used
  `answers/work/car_retails_2`. The prompts are fixed by the registration, so they are not changed for r3 or the
  external test. Every transcript is scanned for cross-arm access before scoring.
