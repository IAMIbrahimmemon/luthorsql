# LuthorSQL

Code for *LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent* (Ibrahim Memon and Hanif Jhaveri; paper in preparation).

**Status: private pre-release.** The license is not decided yet (see "License"). Please do not redistribute.

## What is here

| Path | Contents |
|---|---|
| `src/luthor/sq2/` | System 1: plan grammar, slot heads, k-best plan search, compiler (`topk.py`, `targets.py`, `serialize.py`, `rerank.py`) |
| `src/luthor/stats.py` | Paired tests (exact database-level sign-flip, McNemar) used for every comparison in the paper |
| `scripts/bird_submit/` | The packaged System 1 that BIRD ran on its hidden test set (`README.md` there has the run commands); `cascade/`: stage 2 of the cascade submission (Claude Agent SDK runner, the paper's prompt and SQL tool, tool guard; `README_cascade.md`), built with `CASCADE=1 build_package.sh` |
| `scripts/rerank_sq2.py` | Candidate execution and validator scoring |
| `scripts/system2/` | System 2 experiments: SQL tool for the agent, sandboxes, dev scoring, threshold rule, leak check, API loop for open-weight agents |
| `scripts/analysis/` | Analysis scripts behind the paper's numbers (cascade threshold, latency, single-adapter evaluation, etc.) |
| `experiments/*_crosscheck/` | Cross-agent runs (Sonnet, Haiku, GLM 5.3, GLM 5.3 Flash, DeepSeek V4.1 Flash): pre-registrations with verdicts, answers, scored results |
| `experiments/cost_confirmation/` | Pre-registered held-out cascade test on RESERVED (3 runs): PREREG with verdicts, scorer, result files, incident log; `harness_free.py`: cost repriced without the agent harness's per-session context (exploratory) |
| `experiments/external_cascade/` | Pre-registered external test (KaggleDBQA test + Spider-Realistic, 693 questions) |
| `experiments/memorisation_probe/` | Pre-registered memorisation probes of the agent (schema-free SQL, guided vs general completion), incl. BIRD Mini-Dev |
| `experiments/renamed_routed/` | Renamed-schema test of the routed half (renamer, rewritten gold, scorer) |
| `experiments/fresh_unseen/` | Fresh test set on 8 post-cutoff public databases: builders, sources and licences, question generator, verification record, scorer |
| `experiments/top3_at_30/`, `experiments/final_config/` | Pre-registered attempts to raise accuracy at the same cost (none adopted), incl. a corrected scorer bug |
| `experiments/w83/` | Exploratory (not pre-registered) search on the selection split; report and GPU ledger |
| `scripts/chains/*_s1*.sh`, `scripts/shade/` | Gold-free System 1 GPU runs for the held-out, external and fresh tests (one rented H100, auto-destroyed) |
| `splits/` | DECISION (796) and RESERVED (348) question ids from BIRD train, with their hashes; `fresh_unseen/items.jsonl`: the 227 verified fresh-set questions with gold SQL (databases rebuilt by `experiments/fresh_unseen/builders/`) |
| `tests/` | Unit tests |

Many other scripts are from earlier rounds of the project (described in the paper's history appendix).

## Main results (details and every source file in the paper and in `experiments/`)

- **System 1 alone:** 68.03% execution accuracy on BIRD's hidden test set (scored by the BIRD team), 0.84 s per question on one H100.
- **Cascade, threshold 3.0, Claude Opus 5.5 as System 2:**
  - Measured against Opus given System 1's candidates on every question, on held-out data with a threshold fixed in advance:
    - Cost per question: 29.9–36.7% lower (34.3% pooled over three RESERVED runs) and 31.7% lower on 693 external questions.
    - Accuracy: within 1.3 points in every run; no difference was significant.
    - Total time per question: 12–24% lower.
  - The pre-registered RESERVED run 1 narrowly missed its 30% cost bar and did not show non-inferiority. The external test met its cost and accuracy bars.
  - Repriced without Claude Code's per-session context (exploratory), the saving against Opus with candidates stays at 28–35%, but the saving against Opus alone (16–29% as measured) falls to −26% to +11%.
- **Memorisation:** without a schema, Opus reproduces 48–92% of public benchmarks' gold SQL. With its tool, its accuracy on the routed questions changed by +0.6 points (95% interval −7.1 to +6.0) when every name was rewritten.
- **Fresh set:** on 227 questions over post-cutoff databases, the cascade scored 94.3% against 97.4% for Opus alone, at 20% lower measured cost.
- **Not yet measured:** the cascade has not been run on BIRD's hidden test set.

## Re-scoring the experiments

Each new experiment folder has an `artifacts/` directory with everything needed to recompute its numbers. Gold SQL is
not in it; it comes from BIRD train, KaggleDBQA, Spider-Realistic or `splits/fresh_unseen/`.
- `chunks/<arm>/*.jsonl`: the exact inputs each agent saw, with System 1's candidates.
- `answers/<arm>/*.jsonl`: the agents' answers.
- `usage.json`: per agent session, token usage (input, cache write, cache read, output, plus a characters/4 output
  estimate), wall seconds, and dollars at Opus 5.5 list prices. These are the inputs to every cost and time figure.
  Full transcripts are not included.
- `s1_*.jsonl` (cost_confirmation, external_cascade, fresh_unseen): System 1's gold-free outputs per question (pick,
  margin, top-8 candidate groups with result previews).

To re-score, point a scorer's `--box` at the matching `artifacts/` folder. The scorers rebuild each question's SQL from
the answer's pick label and the chunk's candidates, then execute it against the gold. Accuracy needs only the databases.
Cost and time come from `usage.json`. The fresh-set databases are in the private Hugging Face dataset
`IAMIbrahim/luthorsql-fresh-unseen`; BIRD-train, KaggleDBQA and Spider databases come from their official releases.

## Weights

Hugging Face (private for now):
- `IAMIbrahim/luthorsql-bird`: the two-adapter system (generator `r1/`, validator `valhard_rr/`).
- `IAMIbrahim/luthorsql-bird-single`: the single-adapter variant (`single/`).

Both are LoRA adapters plus small heads on [seeklhy/OmniSQL-7B](https://huggingface.co/seeklhy/OmniSQL-7B).

## Quick start (System 1 on BIRD)

```bash
hf download IAMIbrahim/luthorsql-bird --local-dir weights
cd scripts/bird_submit && bash run_bird.sh <dev.json> <dev_tables.json> <dev_databases> predict_dev.json
```

You need one NVIDIA GPU with at least 48 GB (tested on A100 80GB, CUDA 12.2).

## Data

All training data is public: BIRD train, Spider train and a SynSQL-2.5M sample. The VALHARD validator also used
BIRD-Platinum labels (Zhu et al., 2026), for which we found no license. Data files are not in this repository.

## License

Not decided. Because of the BIRD-Platinum labels, the weights will likely be released for research use only.
