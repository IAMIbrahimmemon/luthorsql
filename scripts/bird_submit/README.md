# LuthorSQL: BIRD test submission

LuthorSQL is a single 7B model system with no API.

1. A grammar-constrained generator writes typed query plans. A deterministic compiler turns each plan into SQLite SQL.
2. The k = 32 distinct best-scoring plans are kept as candidates.
3. Every candidate is executed read-only on the database.
4. A learned validator scores the candidates. The answer is the highest-scoring candidate that runs and returns a
   non-null result.

Both the generator and the validator are LoRA adapters (plus small heads) on
[seeklhy/OmniSQL-7B](https://huggingface.co/seeklhy/OmniSQL-7B).

- Inputs: the question, the evidence, the SQLite database and its `database_description/*.csv` files.
- `column_meaning.json` is not used. No gold SQL is read; the `SQL` field of the question file, if present, is ignored.

## Requirements

- One NVIDIA GPU with at least 48 GB. We tested it on an A100 80GB PCIe with driver 535 (CUDA 12.2). Peak memory is
  about 41 GB.
- Linux, internet access for the first run (Python packages and the base model, about 15 GB).
- No Java and no vLLM.

## Weights

Download the weights into `./weights`, so that you have `weights/r1/` (generator) and `weights/valhard_rr/` (validator):

```bash
hf download <weights repo, sent by email> --local-dir weights --token <token sent by email>
```

The base model, seeklhy/OmniSQL-7B, is downloaded from Hugging Face automatically on first use.

## Run

```bash
bash run_bird.sh <path>/test.json <path>/test_tables.json <path>/test_databases predict_test.json
```

- On the first run, this creates `./venv` (Python 3.12, through `uv`) and installs `requirement.txt`. The torch
  2.11.0 cu128 wheels run on CUDA 12.2 and 12.3 drivers through CUDA minor-version compatibility.
- The output, `predict_test.json`, is in the official BIRD format: `{"0": "SQL\t----- bird -----\tdb_id", ...}`, in
  the order of the question file.
- `predict_test.json.cands.jsonl` lists every candidate with its validator score, for inspection only. A candidate
  that failed to run carries its SQLite error in an `error` field.
- `predict_test.json.log` holds the progress, warnings and full tracebacks. Each run appends to it. Every caught
  exception in the package code is logged. Plans that the candidate search drops because they do not decode or
  compile are expected; they are counted per database in the log.

**Incremental saving and resume.**
- The databases are run one at a time. Each stage of each database (generate, execute, validate) is saved under
  `predict_test.json.work/dbs/` as soon as it finishes.
- `predict_test.json` and its `.cands.jsonl` are rewritten after every database with the questions finished so far.
- If the run stops, rerun the same command. It continues from the last saved stage. Questions are batched exactly as
  in an uninterrupted run, so resuming does not change any prediction.
- If one database fails, the error is logged with its traceback and the other databases continue. The run then ends
  with `PREDICT_INCOMPLETE` and exit code 1, and a rerun retries only the failed databases.
- To start over, delete `predict_test.json.work/`. Changed inputs or settings also start over automatically.

**Time.** We ran the full BIRD dev set (1,534 questions) with this package on an A100 SXM4 80GB with driver 535
(CUDA 12.2) and 8 CPU cores. The run took 54 minutes after installation:

| Step | Time |
|---|---|
| Model loading | about 2 minutes |
| Generation | 16 minutes |
| SQL execution | 9 minutes |
| Validator | 20 minutes |

Expect about 1 hour for 1,789 test questions. SQL execution runs in parallel, so more CPU cores make it faster.

**Installing by hand.** If `uv` is not wanted, use Python 3.11 or 3.12 with pip:
`pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirement.txt`, then
`python luthor/scripts/bird_submit/predict.py --questions ... --tables ... --db-root ... --out ... --run weights/r1 --rr weights/valhard_rr`.

## Check on BIRD dev

```bash
bash run_bird.sh dev.json dev_tables.json dev_databases predict_dev.json
```

This run reproduces our scored dev run (59.39 EX on the original 1,534-question dev) up to bf16 numerics.

- **Our own re-run on an A100:** the top-ranked plan matched on 1,528 of 1,534 questions, and the final pick was the
  same SQL on 89% of questions.
- **The differing picks:** two thirds of them return the same result. They come from near-tied plans at the low end
  of the 32-candidate list.

## Files

- `run_bird.sh`: environment setup and the run.
- `requirement.txt`: the pinned packages.
- `luthor/scripts/bird_submit/predict.py`: the whole pipeline. It builds schema cards, finds literal candidates,
  runs the generator (k = 32), compiles, executes, runs the validator and picks.
- `luthor/src/luthor/`: the plan grammar and compiler (`sq/`) and the model code (`sq2/`).
- `luthor/scripts/`: the generator, validator and data-loading code the pipeline imports.
