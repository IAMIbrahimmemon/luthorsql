# LuthorSQL: BIRD test submission

LuthorSQL is a single 7B model system with no API.

1. A grammar-constrained generator writes typed query plans. A deterministic compiler turns each plan into SQLite SQL.
2. The k = 32 distinct best-scoring plans are kept as candidates.
3. Every candidate is executed read-only on the database.
4. The same model scores the candidates. The answer is the highest-scoring candidate that runs and returns a
   non-null result.

One LoRA adapter on [seeklhy/OmniSQL-7B](https://huggingface.co/seeklhy/OmniSQL-7B) does both jobs: small heads read
it to write the plans, and a linear head reads the same adapter to score each executed candidate. The adapter was
trained on a mixture of generation and selection data. There is no second adapter or model.

- Inputs: the question, the evidence, the SQLite database and its `database_description/*.csv` files.
- `column_meaning.json` is not used. No gold SQL is read; the `SQL` field of the question file, if present, is ignored.

## Requirements

- One NVIDIA GPU with at least 40 GB. We tested it on an A100 SXM4 80GB (about 44 GB in use during the run, 13.5 GB
  for the model) and on an A100 SXM4 40GB with driver 535 (CUDA 12.2). The torch 2.11.0 cu128 wheels run on CUDA 12.2
  and 12.3 drivers through CUDA minor-version compatibility.
- Linux, internet access for the first run (Python packages and the base model, about 15 GB).
- No Java and no vLLM.

## Weights

Download the weights into `./weights`, so that you have `weights/single/` (the one adapter and its heads):

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

**Time.** We ran the 1,534 BIRD dev questions (question text only) with this package on an A100 SXM4 80GB:

| Step | Time |
|---|---|
| Model loading | about 2 minutes on the first run, 10 seconds once cached |
| Generation | 13.5 minutes |
| SQL execution | 5.5 minutes |
| Scoring the candidates | 19 minutes |

Expect about 45 minutes for 1,789 test questions. SQL execution runs in parallel, so more CPU cores make it faster.

**Resume check.** In that run we stopped the job after 3 of the 11 databases were saved (`predict_dev.json` already held
340 predictions) and ran the same command again; it logged `resume: 3 of 11 databases already done (340 questions)`
and finished the rest.

**Installing by hand.** If `uv` is not wanted, use Python 3.11 or 3.12 with pip:
`pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirement.txt`, then
`python luthor/scripts/bird_submit/predict.py --questions ... --tables ... --db-root ... --out ... --run weights/single --rr weights/single`.

## Files

- `run_bird.sh`: environment setup and the run.
- `requirement.txt`: the pinned packages.
- `luthor/scripts/bird_submit/predict.py`: the whole pipeline. It builds schema cards, finds literal candidates,
  runs the generator (k = 32), compiles, executes, scores the candidates with the same adapter and picks.
- `luthor/src/luthor/`: the plan grammar and compiler (`sq/`) and the model code (`sq2/`).
- `luthor/scripts/`: the generator, validator and data-loading code the pipeline imports.
