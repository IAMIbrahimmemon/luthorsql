# Gold-SQL verification of the fresh unseen text-to-SQL set (240 items)

Task spec: `TASK_README_VERIFY.md`. Deliverable: **`results/verdicts.jsonl`** (497 verdict lines, exact README schema).

## Outcome
- 240 items, every item judged by **2 independent blind verifiers** (480 verdicts) plus 17 adjudicator verdicts on flagged items. All verdicts are kept.
- **227 correct, 13 wrong, 0 unresolved.** Pre-registered sample (50): 46 correct, 4 wrong.
- 4 of the 13 "wrong" and 1 "correct" were decided 2–1 after adjudication (marked "majority" below); treat those as contestable.
- Audit (`results/audit_report.json`): 0 integrity problems, 0 coverage gaps, planted canaries caught **23/23**, no verifier missed a canary.

## Process (200 agents, `workflow.js`)
1. 168 verifier agents, 2–3 tasks each, same database per agent. Pass A and pass B batches differ, so an item's two verifiers are different agents.
2. 32 adjudicator agents for items the ledger flagged. 17 had work; 15 had empty buckets.

## Anti-hallucination checks (`harness/`)
- Agents never write verdicts directly. Everything goes through `harness/v.py`, which runs SQL itself on read-only connections.
- **Blind first:** an agent must `commit` its own executable SQL plus an interpretation before `reveal` shows the gold. Committed SQL is immutable, and every step is timestamped in `ledger/events.log`.
- **own_matches_gold is computed by code**, not reported by agents. A verdict that conflicts with it is refused unless the agent gives a written explanation, and such items are sent to adjudication.
- **"Wrong" requires a `fixed_sql` that executes.** If the fix returns exactly the gold result, the verdict is refused (unless the category is `ambiguous`).
- Agents see **opaque task tokens**, not item ids.
- **23 canaries**: gold SQL deliberately broken with a known defect (`>=`→`>`, off-by-one literal, DESC→ASC, dropped DISTINCT), mixed in blind. They measure verifier reliability. Answer key: `harness/private/tasks.jsonl` (`is_canary`).
- `harness/assemble.py` re-executes every own/fixed/gold query. It also checks that each verdict's SQL matches its commit hash and that the steps ran in commit→reveal→submit order.
- A post-hoc scan of all 2160 agent tool calls found no reads of `items.jsonl`, the private answer key, the ledger, or other agents' work dirs. Reading those was forbidden by instruction but not technically blocked.
- Preflight: all 240 gold queries execute and match `gold_result_preview`. The only differences are two FDA previews truncated mid-string.

## Items not unanimously correct
| id | prereg | status | first verdict's reason (truncated) |
|---|---|---|---|
| fresh_chicago_crashes_2026_14 | yes | wrong (unanimous, 2 verifiers) | Gold returns 98 rows including an extra (NULL, NULL) group built from 159 unit records at 25 mph with NULL make and NULL occupant_cnt (PEDESTRIAN 75,  |
| fresh_chicago_crashes_2026_18 |  | wrong (unanimous incl. adjudicator) | Gold groups by make without excluding NULL, so it returns an extra (NULL, 78.61) row (149 vs 148 rows); that group is 2020 vehicles rows all with vehi |
| fresh_chicago_crashes_2026_27 | yes | wrong (unanimous incl. adjudicator) | Gold's raw float test '> 30' admits 4 crashes with an exact 30-minute gap (e.g. 15afb2a439: 13:00 -> 13:30, julianday gives 30.000000223517) shown as  |
| fresh_chicago_crashes_2026_6 |  | wrong (unanimous incl. adjudicator) | Gold rows are right (CHEVROLET 1823, TOYOTA 1818, FORD 1591; #4 HONDA 1298, so no boundary tie; NULL make 1588 is excluded and below FORD anyway), but |
| fresh_chicago_crashes_2026_8 |  | correct (majority after adjudication: {'correct': 2, 'wrong': 1}) | Gold joins all 28314 crashes to categories (0 unmatched/NULL causes), uses the exact crash_type string, and returns the same 7 categories in the same  |
| fresh_chicago_crashes_2026_9 |  | wrong (unanimous incl. adjudicator) | Gold's 'second-highest make' is the placeholder 'UNKNOWN' (5111 drivers, behind TOYOTA 5975); the question/evidence never say whether 'UNKNOWN' counts |
| fresh_chicago_crime_2026_6 | yes | wrong (majority after adjudication: {'wrong': 2, 'correct': 1}) | police_districts has 23 labels but gold's inner join returns 22 rows: district 31ST has 7 crimes (ALLEY 1, OTHER (SPECIFY) 5, RESIDENCE 1) and 0 at ST |
| fresh_f1_2026_18 | yes | correct (unanimous incl. adjudicator) | Round-9 min fastest lap is 91.777 (ANT); gold returns the 9 drivers with gap <= 1.5 (ANT 0.0 .. PIA 1.14), the next driver HUL at 93.433 (gap 1.656) i |
| fresh_f1_2026_19 |  | correct (unanimous incl. adjudicator) | Gold returns the same 13 round-9 drivers, totals and order as my committed query (LEC 29.0, RUS 23.0, HAM 22.0 ... COL 2.0 before PIA 2.0 by driver_id |
| fresh_f1_2026_9 |  | wrong (majority after adjudication: {'wrong': 2, 'correct': 1}) | Gold's values are right (Ferrari 20.0, Mercedes 9.0, RB 7.0 ... Alpine 1.5, Aston Martin/Cadillac/Haas/Williams 0.0; all 11 teams have 2 rows, 2 disti |
| fresh_fda_recalls_2026_25 |  | wrong (unanimous incl. adjudicator) | firm_id is documented as a firm LOCATION key and gold partitions by it while outputting only firm_name, so 4 names get several conflicting 'latest' ro |
| fresh_nws_alerts_2026_1 |  | correct (unanimous incl. adjudicator) | Gold returns the same 6 distinct events as my independent query (harness multiset_equal_float6=true): Flash Flood Warning, Flood Watch, Severe Thunder |
| fresh_nws_alerts_2026_17 |  | wrong (unanimous incl. adjudicator) | Gold compares the raw julianday float to 60. 13 alerts last exactly 3600 s, and 9 of them (e.g. 2026-10-21T11:45:00Z to 12:45:00Z) compute to 59.99999 |
| fresh_quakes_2026_3 |  | wrong (unanimous incl. adjudicator) | There are 22 regions with region_type='us_state', but gold's inner join with magnitude >= 3.0 in WHERE returns only 18 rows. It drops Illinois (max 2. |
| fresh_uk_house_prices_2026_23 |  | wrong (majority after adjudication: {'wrong': 2, 'correct': 1}) | Gold returns 83 rows including a (NULL, 1) group from transaction 5834E4E8-C4C1-29C7-E063-4804A8C015BC on property 7393 whose postcode and outward_cod |
| fresh_uk_house_prices_2026_4 | yes | correct (unanimous incl. adjudicator) | Gold filters ppd_category_code='A' and averages price per new build status via a lossless 1:1 PK join (61811 N and 15 Y rows all matched), giving Newl |
| fresh_uk_house_prices_2026_6 |  | wrong (unanimous incl. adjudicator) | Gold wraps the evidence formula in ROUND(...,2) and sorts by the rounded value. As a result it lists NORTH SOMERSET (18.0769, 260 tx) before SOUTH GLO |
| fresh_wc2026_23 | yes | wrong (unanimous incl. adjudicator) | Question asks 'for each stage', but gold's WHERE went_to_extra_time = 1 drops stages with zero extra-time matches: Semi-final (2 matches, 0 ET) and Br |

Full per-item detail: `results/item_summary.json`. Raw immutable records: `ledger/{commits,reveals,verdicts}/`.

Reproduce the audit: `python3 harness/assemble.py results/`. This needs the bundle at the path in `harness/private/config.json`; the databases are not committed (103 MB).
