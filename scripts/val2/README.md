# val2: data for the Recipe A+ validator retrain (Mac-only preparation)

Plan: `docs/paper_notes/research/selection_memo_2026-10-10.md` §3 Step 1 and §4. Nothing here trains or loads a
model, uses a GPU or calls a paid API. Public data only (BIRD-train, BIRD-Verified); no sealed, BIRD-dev or RESERVED
data is read. DECISION ids are used only by `attribute_ensemble.py`.

| Script | Output | What |
| --- | --- | --- |
| `attribute_ensemble.py` | `data/val2/attribute_ensemble.out` | Task 1: R1 / R2 / sum_z, own lists vs union, k=8 and k=32, on DECISION |
| `build_lists.py` | `data/val2/{train_lists,val_lists}.jsonl`, `relabel_stats.json`, `build_stats.json`, `build.log` | Tasks 2–3 |
| `lengths.py` | `data/val2/lengths.json` | Task 4: validator input length per list (OmniSQL tokenizer) |
| `uncovered_dbs.py` | `data/val2/uncovered_dbs.json` | Task 5: BIRD-train DBs with no wide-pool question |
| `execq.py`, `mutate.py` | | gen-time execution labelling (copy of `rerank_sq2._exec_db`), near-miss mutants |

Tests: `uv run pytest tests/test_val2_build.py -q`.

## build_lists.py

```
uv run python scripts/val2/build_lists.py [--workers 6] [--seed 0]     # ~4 min on 6 CPU workers, < 0.5 GB RAM
```

Inputs: `runs/phase1/p1w/p1w/{r1,r2}/cands_train.jsonl` (box W, k=32 distinct plans, 4,541 BIRD-train questions on
46 DBs), `data/raw/bird_train/{train.json,train_databases}`, `data/raw/bird_verified/bird-verified-{train,val}.json`.

1. **Filters.** Held-out DBs (`data/sys1/dev_dbs_bird.json`) and contaminated DBs are dropped (none are in the
   pools). `data/sys1/gold_noise_drop.json` ids are dropped unless BIRD-Verified has an entry for them.
2. **Relabel.** For every question with a BIRD-Verified entry, the verified question and evidence replace the
   original ones (the candidates are labelled against the verified gold, which answers the verified question). When
   the verified SQL differs from the original (whitespace-normalised), the question is *corrected*: the verified gold
   and every R1 ∪ R2 candidate are re-executed with the gen-time labelling (`execq.py`: 3 s per candidate as at gen
   time, 10 s for the gold; ok = result-set equality with the gold; sig = gen-time signature). Corrected questions
   graded by BIRD-Verified other than as a `set` (subset / list / multiset: 41 BIRD-Verified entries overall, 18 in the pools) are dropped. Every
   kept question's gold is executed; a gold that errors, times out, returns no rows or only NULLs drops the question.
3. **Lists** (one record per question, same fields as `cands_train.jsonl`, plus `weight` and `gold_corrected`;
   candidates keep their fields plus `src` = r1 / r2 / both, mutants add `mutant: true` and `mutation`):
   - R1 ∪ R2 by exact SQL text (rank/cost = the best over the two runs);
   - dead candidates dropped (`runs` and `nonnull` both required);
   - one representative per result signature `sig`: even question index (`bird_train_train_<n>`) → the best-ranked
     SQL of the class; odd → `random.Random(f"val2:{seed}:{id}:{sig}")` choice among the class;
   - at most 12 classes: every right class (there is at most one: right = same result set as the gold) and the
     best-ranked wrong classes (ordered by the best rank of any member of the class);
   - up to 2 near-miss negatives, appended after the cap (so a list has at most 14 entries): one right SQL (the
     representative, or the best-ranked class member that round-trips) is mutated by one edit (`mutate.py`: toggle
     DISTINCT; add LIMIT 1 / remove LIMIT; remove a `* 100` factor or multiply a numeric projection by 100; append one
     more column of the first projected column's table; `<`↔`<=`, `>`↔`>=`); each mutant is executed and kept only if it
     runs, is non-null, is not right, and its sig differs from every class in the list and from the other mutant.
     Mutants get `ok: false`, `rank` 100+, `cost` 99.0. Only SQL whose sqlglot round trip reproduces the text exactly
     is mutated, so the edit is the only textual difference (no formatting tell);
   - `weight` 3.0 when the rank-0 (greedy) candidate of either run's original list is wrong (or dead), else 1.0;
   - kept only with ≥ 1 right and ≥ 1 wrong entry (a list whose only negatives are mutants is kept and counted in
     `build_stats.json` → `kept_only_mutant_negatives`).
4. **Split.** Questions on citeseer, disney, professional_basketball, trains → `val_lists.jsonl`; the rest →
   `train_lists.jsonl`.

SQL text is never normalised: the validator sees test-time SQL as is (this deliberately departs from the memo's
"normalise SQL formatting with sqlglot").

### Using the lists with the current trainer

`scripts/rerank_sq2.py train --train-cands data/val2/train_lists.jsonl` reads the file as is, but its
`train_items` samples one right + `--train-k − 1` wrong candidates and ignores `weight`; use `--train-k 14` (or
more) so the whole list is used. `weight`, all-positives logsumexp and token-budget chunking are for the trainer
rewrite (owned elsewhere).

## Results of the 2026-10-09 build (seed 0)

- Relabel: 1,204 pool questions have a BIRD-Verified entry, 531 with a changed gold; 334 of those flip at least one
  candidate label (1,480 candidates to right, 4,953 to wrong); 181 lose every right candidate (dropped: no live right
  class), 8 gain one. 3 golds dropped (2 empty, 1 all-NULL). Per DB: `relabel_stats.json`.
- train_lists: 3,971 lists on 42 DBs, 9.26 real classes + 1.39 mutants per list on average, 25.3% weight 3.0, 310 with
  corrected gold. val_lists: 197 lists on 4 DBs, 36.6% weight 3.0. 61 lists have two right classes (int vs float
  results: equal as sets, so right, but different `sig`).
- Mutants: a right SQL could be mutated in 4,136 lists; 90.8% of train lists got ≥ 1 mutant (2,010 two, 1,766 one).
  Kept / tried by edit: x100 1,522 / 1,606, extra_col 3,432 / 3,916, limit 759 / 4,136, cmp 61 / 346, distinct
  12 / 4,136 (DISTINCT almost never changes a result *set*, so it is nearly useless under set-based EX; a
  COUNT(x) ↔ COUNT(DISTINCT x) edit would be the useful variant).
- Lengths (`lengths.json`, OmniSQL tokenizer): schema + padded list per question p50 2,942, p99 7,421, max 9,271
  tokens (works_cycles); the old k=32 lists were p50 4,347, max 17,558.

## Known gaps

- 14 BIRD-train DBs (1,277 gen-eligible questions) have no wide-pool candidates (`uncovered_dbs.json`): Step 1's
  k=32 R1/R2 generation on them is GPU work.
- The in-pool validator scores needed for the gate-signal check (task 6) do not exist for the train lists: box W
  never scored `cands_train.jsonl`. It needs OOF scores from Step 1's fold models.
