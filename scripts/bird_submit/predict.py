"""LuthorSQL (frozen System 1: R1 grammar generator k=32 + VALHARD validator) on a BIRD-format question file.

One command, questions + databases in, BIRD prediction file out:

  python scripts/bird_submit/predict.py --questions test/test.json --tables test/test_tables.json \
      --db-root test/test_databases --out predict_test.json \
      --run weights/r1 --rr weights/valhard_rr

Steps (the same code paths as the scored BIRD dev run, runs/phase1/devlook/FROZEN.md variant A):
  1 cards     per database: schema card from <split>_tables.json + the SQLite file + database_description/*.csv
              (scripts/prepare_plans.py's card_from_spider_tables + read_descriptions). column_meaning.json is not read.
  2 questions id / db_id / question / context (= evidence) only.
  3 generate  R1: literal candidates (question + evidence + live DB value lookup), encoder, lazy k-best 32 --distinct
              plans, deterministic compile to SQL (blind_gen.gen = rerank_sq2.py gen).
  4 execute   every candidate read-only, 3 s per query, at most 100k rows (runs / non-null / result fingerprint).
  5 validate  VALHARD scores every candidate (rerank_sq2.score_file).
  6 pick      highest VALHARD score among candidates that run and return a non-null value; if none, among all with
              SQL (valhard_eval.py's rule). A question with no candidate gets 'SELECT 1'.
Output: {"<index>": "<SQL>\\t----- bird -----\\t<db_id>"} in question-file order, the official BIRD format.
Also writes <out>.cands.jsonl (every candidate with its score and, if it failed to run, its SQLite error) and
<out>.log (progress, warnings and tracebacks; appended to on every run).

Incremental saving and resume: the databases are run one at a time, and each stage (generate, execute, validate) of
each database is saved under <out>.work/dbs/ when it finishes. <out> and <out>.cands.jsonl are rewritten after every
database with the questions finished so far. If the run stops, rerun the same command: it starts from the last saved
stage. Questions are batched as in an uninterrupted run, so resuming does not change any prediction. --fresh starts
over; changed inputs or settings start over automatically.

Needs one GPU (~41 GB peak on an A100 80GB) and the inference packages in requirement.txt.
No gold SQL is read: the 'SQL' field of the question file, if present, is ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "scripts" / "devlook", ROOT / "scripts", ROOT / "src"):
    sys.path.insert(0, str(p))
os.environ.setdefault("LUTHOR_GRAMMAR", "v3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __name__ == "__main__":  # LUTHOR_GRAMMAR before the first luthor.sq import (pool workers inherit it)
    from luthor.grammar_cli import preset
    preset()

FILLER = "SELECT 1"
LOG = logging.getLogger("luthorsql")


def build_inputs(a) -> tuple[list[dict], dict[str, str]]:
    """Cards + label-free question rows into a.work/qs/; returns (rows in file order, db_id -> sqlite path)."""
    from prepare_plans import read_descriptions
    from luthor.sq.schema_card import card_from_spider_tables
    qs = json.loads(Path(a.questions).read_text())
    tables = {t["db_id"]: t for t in json.loads(Path(a.tables).read_text())}
    root = Path(a.db_root)
    paths = {}
    for db in sorted({q["db_id"] for q in qs}):
        p = root / db / f"{db}.sqlite"
        if not p.exists():
            raise SystemExit(f"missing database {p}")
        paths[db] = str(p)
    out = Path(a.work) / "qs"
    (out / "cards").mkdir(parents=True, exist_ok=True)
    t = time.time()
    for db, p in paths.items():
        cp = out / "cards" / f"{db}.json"
        if not cp.exists():
            card_from_spider_tables(tables[db], Path(p), read_descriptions(root / db / "database_description")).save(cp)
    LOG.info(f"cards: {len(paths)} databases in {time.time() - t:.0f}s")
    rows = [{"id": f"{a.id_prefix}_{i}", "db_id": q["db_id"], "question": q["question"].strip(),
             "context": (q.get("evidence") or "").strip(), "outcome": "missing"} for i, q in enumerate(qs)]
    if a.limit:
        rows = rows[:a.limit]
    (out / "plans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows, paths


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _save(path: Path, recs: list[dict]) -> None:
    _atomic_write(path, "".join(json.dumps(r) + "\n" for r in recs))


def _load(path: Path, ids: list[str]) -> list[dict] | None:
    """A saved stage of one database, or None if it is missing, unreadable or holds other questions."""
    if not path.exists():
        return None
    try:
        recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except Exception:
        LOG.exception(f"unreadable checkpoint {path}: redoing this stage")
        return None
    if sorted(r["id"] for r in recs) != sorted(ids):
        LOG.warning(f"checkpoint {path} holds other questions: redoing this stage")
        return None
    return recs


def _exec_task(task):
    """(db path, [[sql | None]], seconds) -> per question, per candidate (runs, nonnull, sig, preview, error).
    blind_gen._exec_blind plus the SQLite error message: same connection, time limit, row cap and fingerprints."""
    import hashlib
    import rerank_sq2 as R
    path, qs, seconds = task
    rows = R._connect(path, seconds)
    out = []
    for sqls in qs:
        res, seen = [], {}
        for s in sqls:
            if s is None:
                res.append((False, False, None, None, "no SQL (plan did not compile)"))
            elif s in seen:
                res.append(seen[s])
            else:
                try:
                    got = rows(s)
                    gs = set(map(tuple, got))
                    sig = hashlib.md5(repr(sorted(map(repr, gs))).encode()).hexdigest()[:12]
                    r = (True, any(v is not None for x in got for v in x), sig, R.result_preview(got), None)
                except Exception as ex:
                    r = (False, False, None, None, f"{type(ex).__name__}: {ex}"[:300])
                seen[s] = r
                res.append(r)
        out.append(res)
    return out


def _generate(a, m, card, group, path) -> list[dict]:
    import zlib
    import blind_gen as BG
    from luthor.sq.plan import compile_plan
    from luthor.sq2.model import TorchScores
    from luthor.sq2.topk import candidates
    from luthor.sq2.train import encode_db, serve_chunks
    g = card.graph()
    try:
        BG._value_cands(a, card, group, path)
    except Exception:
        LOG.exception(f"value index failed on {card.db_id}: questions keep their text-only literal candidates")
    recs = []
    for tabs, chunk in serve_chunks(card, group, 8):
        st, oix, enc = encode_db(m, card, chunk, tabs)
        for e, x in zip(chunk, enc):
            sc = TorchScores(m, x, st.option_ids, e.cands)
            try:
                pc = candidates(sc, k=a.k, with_cost=True, wide_singles=0, wide_pairs=0, runner_up2=0,
                                samples=0, temp=1.0, seed=zlib.crc32(e.id.encode()), distinct=True,
                                compile_fn=lambda p: compile_plan(p, g))
            except Exception:
                LOG.exception(f"{e.id}: candidate search failed, no candidates for this question")
                pc = []
            cl = []
            for rank, (p, cost) in enumerate(pc):
                try:
                    sql = compile_plan(p, g)
                except Exception as ex:
                    LOG.warning(f"{e.id}: plan {rank} did not compile ({type(ex).__name__}: {ex})")
                    sql = None
                cl.append({"rank": rank, "cost": round(cost, 4), "sql": sql, "src": "r1"})
            recs.append({"id": e.id, "db_id": card.db_id, "question": e.question, "context": e.context, "cands": cl})
    return recs


def _execute(a, pool, path, recs) -> int:
    n = max(1, -(-len(recs) // a.workers))  # questions per task: the workers share one database
    chunks = [recs[i:i + n] for i in range(0, len(recs), n)]
    tasks = [(path, [[c["sql"] for c in r["cands"]] for r in ch], a.seconds) for ch in chunks]
    n_err = 0
    for ch, res in zip(chunks, pool.imap(_exec_task, tasks)):
        for rec, r in zip(ch, res):
            for c, (runs, nonnull, sig, prev, err) in zip(rec["cands"], r):
                c.update(runs=runs, nonnull=nonnull, sig=sig, res=prev)
                if err:
                    c["error"] = err
                    n_err += 1
    return n_err


def predict(a, rows, paths, on_db_done) -> tuple[dict[str, dict], list[str]]:
    """Runs the databases one at a time: generate -> execute -> validate. Each stage of each database is saved under
    <work>/dbs/ as soon as it finishes, and a rerun starts from the last saved stage (resume). Questions are batched
    exactly as in a single uninterrupted run, so resuming does not change any candidate, score or pick. A database
    that fails is logged with its traceback and skipped; the others continue. Returns (records by id, failed dbs)."""
    import collections
    from multiprocessing import Pool

    import torch
    import blind_gen as BG
    import rerank_sq2 as R
    from luthor.sq2 import topk

    torch.backends.cuda.enable_cudnn_sdp(False)  # as rerank_sq2.main
    torch.manual_seed(0)
    a.plans = [Path(a.work) / "qs"]
    a.bird_desc = ""
    a.grammar = None
    a.base, a.subfolder = "seeklhy/OmniSQL-7B", ""
    R.use_run_settings(a)  # grammar check, vocab.json, candidate order, base model (from the run's args.json)
    cards, exs, _ = BG._load(a)
    card_paths = R.load_cards(a.plans)
    by_db = collections.defaultdict(list)
    for e in exs:
        by_db[e.db_id].append(e)
    state = Path(a.work) / "dbs"
    state.mkdir(parents=True, exist_ok=True)
    done, failed, todo = {}, [], []
    for db, group in by_db.items():
        recs = _load(state / f"{db}.done.jsonl", [e.id for e in group])
        if recs is None:
            todo.append(db)
        else:
            done.update((r["id"], r) for r in recs)
    LOG.info(f"resume: {len(by_db) - len(todo)} of {len(by_db)} databases already done ({len(done)} questions)")
    if not todo:
        return done, failed
    t = time.time()
    m = R.load_slot_model(a.run, a.base, a.subfolder)
    R.validator_settings(a.rr)
    shared = R.shared_adapter(a.rr)  # single-LoRA weights: generator and validator are one adapter on one backbone
    if shared and a.rr.resolve() != a.run.resolve():
        raise SystemExit(f"--rr {a.rr} is a shared-adapter dir: pass the same dir as --run")
    rr = R.shared_validator(m, a.rr) if shared else R.load_validator(a.rr, a.base, a.subfolder, a.lora_r)
    rr.eval()
    LOG.info("one adapter (generator and validator share the LoRA)" if shared else "two adapters (R1 + VALHARD)")
    LOG.info(f"models loaded in {time.time() - t:.0f}s ({torch.cuda.memory_allocated() / 2**30:.1f} GB)")

    with Pool(a.workers) as pool, torch.no_grad():
        for db in todo:
            group, ids, T = by_db[db], [e.id for e in by_db[db]], time.time()
            try:
                recs = _load(state / f"{db}.exec.jsonl", ids)
                if recs is None:
                    recs = _load(state / f"{db}.gen.jsonl", ids)
                    if recs is None:
                        t = time.time()
                        topk.SKIPPED.clear()
                        recs = _generate(a, m, cards[db], group, paths[db])
                        _save(state / f"{db}.gen.jsonl", recs)
                        LOG.info(f"  {db}: generated {len(recs)} questions in {time.time() - t:.0f}s (plans dropped "
                                 f"during the search: {dict(topk.SKIPPED) or 0}; --debug logs each one)")
                    t = time.time()
                    n_err = _execute(a, pool, paths[db], recs)
                    _save(state / f"{db}.exec.jsonl", recs)
                    LOG.info(f"  {db}: executed in {time.time() - t:.0f}s "
                             f"({n_err} candidates failed to run; messages in the 'error' field of .cands.jsonl)")
                t = time.time()
                R.score_file(rr, recs, card_paths, a.budget_chars)
                m.eval()  # score_file ends in train mode; with one shared backbone that would turn LoRA dropout on
                rr.eval()
                _save(state / f"{db}.done.jsonl", recs)
                LOG.info(f"  {db}: validated in {time.time() - t:.0f}s")
            except Exception:
                LOG.exception(f"database {db} failed ({len(group)} questions); rerun the same command to retry it")
                failed.append(db)
                torch.cuda.empty_cache()
                continue
            done.update((r["id"], r) for r in recs)
            on_db_done(done)
            LOG.info(f"done {db} in {time.time() - T:.0f}s: {len(done)}/{len(exs)} questions saved")
    return done, failed


def pick(cands):
    """scripts/analysis/valhard_eval.py's rule (as devlook_score.py and sys1_latency.py)."""
    u = [c for c in cands if c.get("sql")]
    if not u:
        return None
    live = [c for c in u if c.get("runs") and c.get("nonnull")] or u
    return max(live, key=lambda c: c.get("rr", -1e9))


def write_outputs(a, rows, recs, final: bool) -> int:
    """<out> and <out>.cands.jsonl in question-file order. Before the end of the run (final=False) they hold only the
    questions finished so far; at the end every question is present ('SELECT 1' where there is no candidate)."""
    preds, lines, n_fill = {}, [], 0
    for i, r in enumerate(rows):
        rec = recs.get(r["id"])
        if rec is None and not final:
            continue
        p = pick(rec["cands"]) if rec else None
        sql = p["sql"].replace("\t", " ").replace("\n", " ") if p else FILLER  # one line per prediction
        n_fill += p is None
        preds[str(i)] = f"{sql}\t----- bird -----\t{r['db_id']}"
        lines.append(json.dumps({"index": i, **(rec or {"id": r["id"], "db_id": r["db_id"], "cands": []}),
                                 "pick": p and p["sql"]}) + "\n")
    _atomic_write(Path(f"{a.out}.cands.jsonl"), "".join(lines))
    _atomic_write(Path(a.out), json.dumps(preds, indent=1))
    return n_fill


def setup_log(out: str, debug: bool = False) -> None:
    LOG.setLevel(logging.DEBUG if debug else logging.INFO)
    LOG.propagate = False
    for h in list(LOG.handlers):  # main() called again in one process: one set of handlers
        LOG.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(f"{out}.log")):  # the log file is appended to
        h.setFormatter(fmt)
        LOG.addHandler(h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="BIRD-format question file (e.g. test.json)")
    ap.add_argument("--tables", required=True, help="BIRD-format tables file (e.g. test_tables.json)")
    ap.add_argument("--db-root", required=True, help="folder holding <db_id>/<db_id>.sqlite (+ database_description/)")
    ap.add_argument("--out", required=True, help="prediction file (BIRD format)")
    ap.add_argument("--run", type=Path, required=True, help="generator dir (best_lora/, best_heads.pt, args.json, vocab.json)")
    ap.add_argument("--rr", type=Path, required=True, help="validator dir (rerank_head.pt, settings.json, rerank_lora/); "
                    "a single-LoRA dir (settings.json shared_adapter) is passed as both --run and --rr")
    ap.add_argument("--work", default="", help="scratch + checkpoint dir (default: <out>.work)")
    ap.add_argument("--fresh", action="store_true", help="ignore saved checkpoints and start over")
    ap.add_argument("--debug", action="store_true", help="also log each plan the search drops, with its exception")
    ap.add_argument("--id-prefix", default="bird", help="question ids are <prefix>_<index>")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--budget-chars", type=int, default=9000)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first n questions (smoke test)")
    a = ap.parse_args(argv)
    a.work = a.work or f"{a.out}.work"
    setup_log(a.out, a.debug)
    T = time.time()
    try:
        # checkpoints belong to one exact input + settings; anything else starts over
        key = {"questions": hashlib.sha256(Path(a.questions).read_bytes()).hexdigest(),
               "tables": hashlib.sha256(Path(a.tables).read_bytes()).hexdigest(),
               "run": str(Path(a.run).resolve()), "rr": str(Path(a.rr).resolve()),
               **{k: getattr(a, k) for k in ("id_prefix", "k", "seconds", "budget_chars", "lora_r", "limit")}}
        kp = Path(a.work) / "run_key.json"
        if a.fresh or (kp.exists() and json.loads(kp.read_text()) != key):
            LOG.info(f"{'--fresh' if a.fresh else 'inputs or settings changed'}: clearing checkpoints in {a.work}/dbs")
            shutil.rmtree(Path(a.work) / "dbs", ignore_errors=True)
        Path(a.work).mkdir(parents=True, exist_ok=True)
        _atomic_write(kp, json.dumps(key, indent=1))
        rows, paths = build_inputs(a)
        recs, failed = predict(a, rows, paths, lambda done: write_outputs(a, rows, done, final=False))
    except Exception:
        LOG.exception("PREDICT_FAIL: rerun the same command to resume from the last saved stage")
        return 1
    n_fill = write_outputs(a, rows, recs, final=True)
    if failed:
        LOG.error(f"PREDICT_INCOMPLETE: {len(failed)} database(s) failed ({', '.join(failed)}); their questions are "
                  f"'{FILLER}' in {a.out}. Rerun the same command to retry only those databases.")
        return 1
    LOG.info(f"PREDICT_DONE {len(rows)} predictions ({n_fill} without a candidate -> '{FILLER}') -> {a.out} "
             f"in {time.time() - T:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
