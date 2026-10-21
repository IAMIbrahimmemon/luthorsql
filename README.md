# LuthorSQL

Code for *LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent* (paper in preparation).

**Status: private pre-release.** The license is not decided yet (see "License"). Please do not redistribute.

## What is here

| Path | Contents |
|---|---|
| `src/luthor/sq2/` | System 1: plan grammar, slot heads, k-best plan search, compiler (`topk.py`, `targets.py`, `serialize.py`, `rerank.py`) |
| `src/luthor/stats.py` | Paired tests (exact database-level sign-flip, McNemar) used for every comparison in the paper |
| `scripts/bird_submit/` | The packaged System 1 that BIRD ran on its hidden test set (`README.md` there has the run commands) |
| `scripts/rerank_sq2.py` | Candidate execution and validator scoring |
| `scripts/system2/` | System 2 experiments: SQL tool for the agent, sandboxes, dev scoring, threshold rule, leak check, API loop for open-weight agents |
| `scripts/analysis/` | Analysis scripts behind the paper's numbers (cascade threshold, latency, single-adapter evaluation, etc.) |
| `experiments/` | Cross-agent runs (Sonnet, Haiku, GLM 5.3, GLM 5.3 Flash, DeepSeek V4.1 Flash): pre-registrations with verdicts, answers, scored results |
| `splits/` | DECISION (796) and RESERVED (348) question ids from BIRD train, with their hashes |
| `tests/` | Unit tests |

Many other scripts are from earlier rounds of the project (described in the paper's history appendix).

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
