"""System-1 latency (frozen: R1 grammar generator + VALHARD validator) on DECISION questions. LATENCY ONLY: no label is
on the box, nothing is scored against gold, no accuracy is computed or printed.

  stage  (Mac, CPU, seconds)  the seeded sample of DECISION questions -> data/latency/
           qs/plans.jsonl     id / db_id / question / context / outcome="missing" only (labels stripped, as
                              scripts/devlook/prep.py questions); the decoder sees what rerank_sq2.py gen feeds it
           qs/cards/*.json    the 8 DECISION databases' schema cards (data/sys1/bird_train/cards)
           sample.json        seed, warm-up ids, measured ids (DECISION only)
           rr_valhard/        copy of the VALHARD adapter + head + settings (runs/phase1/valhard/valhard/rr)
           MANIFEST.sha256    every staged file (the box refuses a partial push)
  run    (box, one GPU, in-process; scripts/chains/latency_box.sh)  -> OUT/latency.json, latency.txt, rows.jsonl
           model load (generator, validator) timed and reported separately; per-DB setup (value index, column list,
           schema tokens) built before timing and reported separately; 3 warm-up questions run through every path and
           dropped. torch.cuda.synchronize() around every timed region.
           1 single query (batch 1, one question at a time), end-to-end wall seconds per question:
               greedy  k=1 (literal lookup + encode + greedy decode + compile; + executing its one SQL), no validator
               k32     lookup + encode + lazy k-best 32 --distinct + compile + SQLite execution + VALHARD + pick
               k64r    the same with k=64, then the repair stage (repair_gen triggers / constrained re-decoding with the
                       encoder output reused, its candidates compiled + executed) merged in, VALHARD over the merged
                       pool + pick
             p50 / p90 / mean / min / max, per-stage breakdown (lookup, generate, compile, execute, repair,
             repair_execute, validate, pick). Execution is serial in-process (one SQLite connection per question).
           2 throughput with the chains' batched code paths on the same questions (scripts/devlook/blind_gen.py gen /
             repair / merge = rerank_sq2.py gen / repair_gen.py gen minus the gold, then rerank_sq2.score_file):
             per database, chunks of 8 questions per encoder pass, execution in a Pool of --workers processes,
             validator 4 questions per batch; questions / second and $ per 1,000 questions at --price-h.
           GPU name, driver, torch / CUDA / vLLM / transformers / peft versions, peak GPU memory.
         Pick = scripts/analysis/valhard_eval.py's rule (highest VALHARD score among running, non-null candidates).
         Differences from the chains (documented in latency.json "notes"): the k=64+repair pool is not topped up with
         a stored k=32 pool (a live system has none; k=32 distinct is a prefix of k=64 distinct in practice).

  python3 scripts/analysis/latency/sys1_latency.py stage            # Mac
  python  scripts/analysis/latency/sys1_latency.py run --run runs/r1r2/r1 --data /root/d/latency \
      --db-roots /root/d/sys1/dbs --out /workspace/latency --price-h 2.50   # box (venv6)

The stage step reads DECISION ids (data/sys1/heldout_v2/decision.json) and BIRD-train question text only; never
reserved.json, BIRD dev, sealed sets or any pool file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "scripts" / "devlook", ROOT / "scripts" / "repair", ROOT / "scripts", ROOT / "src"):
    sys.path.insert(0, str(p))
if __name__ == "__main__":  # LUTHOR_GRAMMAR before the first luthor.sq import (pool workers inherit it)
    from luthor.grammar_cli import preset
    preset()

SEED = "sys1_latency/2026-10-11"
N_MEASURE = 200
N_WARM = 3
DECISION = ROOT / "data/sys1/heldout_v2/decision.json"
PLANS = ROOT / "data/sys1/bird_train/plans.jsonl"
CARDS = ROOT / "data/sys1/bird_train/cards"
VALHARD = ROOT / "runs/phase1/valhard/valhard/rr"
OUT = ROOT / "data/latency"
Q_KEYS = ("id", "db_id", "question", "context")
LABEL_KEYS = {"ok", "gold_sql", "gold_error", "covered", "in_slots", "compiled_sql", "plan", "SQL", "gold", "rr",
              "gold_rows", "gold_sig", "value_labels", "limit_label", "empty_result", "twin_of", "db_values", "outcome_gold"}
RR_FILES = ("rerank_head.pt", "settings.json", "rerank_lora/rerank/adapter_config.json",
            "rerank_lora/rerank/adapter_model.safetensors")
FORBIDDEN = ("bird_dev", "dev_databases", "sealed", "reserved", "gate.json", "dev.json", "spider_dk", "kaggledbqa",
             "realistic")


# ---- sample + stage (Mac, no torch) ------------------------------------------------------------------------------------
def decision() -> dict:
    d = json.loads(DECISION.read_text())
    assert d.get("split") == "decision", "not the DECISION split file"
    return d


def plan_rows(path: Path = PLANS) -> dict:
    """id -> question row (non-twin rows of the BIRD-train plan set). Only Q_KEYS are ever kept."""
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("twin_of"):
            continue
        rows[r["id"]] = {k: r.get(k, "") for k in Q_KEYS}
    return rows


def sample_ids(dec_ids: list[str], have: set[str], seed: str = SEED, n: int = N_MEASURE, n_warm: int = N_WARM):
    """(warm-up ids, measured ids): a seeded draw from the DECISION ids that have a question row, sorted first so the
    draw does not depend on file order. Selection uses no label (not coverage, not correctness)."""
    pool = sorted(set(dec_ids) & set(have))
    if len(pool) < n + n_warm:
        raise SystemExit(f"only {len(pool)} DECISION questions with a row; need {n + n_warm}")
    pick = random.Random(seed).sample(pool, n + n_warm)
    return pick[:n_warm], pick[n_warm:]


def audit_rows(rows) -> None:
    for r in rows:
        bad = set(r) & LABEL_KEYS
        if bad:
            raise SystemExit(f"LATENCY_STAGE_FAIL label field(s) {sorted(bad)} in row {r.get('id')}")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage(a) -> int:
    dec = decision()
    rows = plan_rows(a.plans)
    warm, meas = sample_ids(dec["ids"], set(rows), a.seed, a.n, a.n_warm)
    dbs = sorted(dec["database_counts"])
    out: Path = a.out
    if out.exists():
        shutil.rmtree(out)
    (out / "qs" / "cards").mkdir(parents=True)
    staged = [{**rows[i], "outcome": "missing"} for i in warm + meas]
    audit_rows(staged)
    assert all(r["db_id"] in dbs for r in staged), "a sampled question is off the DECISION databases"
    (out / "qs" / "plans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in staged))
    for db in dbs:
        shutil.copy2(a.cards / f"{db}.json", out / "qs" / "cards" / f"{db}.json")
    (out / "sample.json").write_text(json.dumps({
        "seed": a.seed, "split": "decision", "decision_ids_sha256": dec.get("ids_sha256"),
        "eligible": "DECISION ids with a non-twin row in data/sys1/bird_train/plans.jsonl",
        "n_eligible": len(set(dec["ids"]) & set(rows)), "databases": dbs,
        "warmup": warm, "measured": meas}, indent=1))
    if a.rr:
        for f in RR_FILES:
            src = a.rr / f
            if src.exists():
                (out / "rr_valhard" / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out / "rr_valhard" / f)
            elif f != "settings.json":
                raise SystemExit(f"LATENCY_STAGE_FAIL missing {src}")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("MANIFEST.sha256", ".DS_Store"))
    (out / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    by_db = {db: sum(r["db_id"] == db for r in staged[a.n_warm:]) for db in dbs}
    print(f"STAGED {len(meas)} measured + {len(warm)} warm-up DECISION questions on {len(dbs)} DBs {by_db} "
          f"(seed {a.seed!r}), {len(files)} files -> {out}")
    return 0


# ---- run (box, GPU) ----------------------------------------------------------------------------------------------------
def pct(xs: list[float], q: float) -> float:
    """numpy's default (linear) percentile."""
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summary(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": round(pct(xs, 50), 4), "p90": round(pct(xs, 90), 4), "mean": round(sum(xs) / len(xs), 4),
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


def live(cands):
    L = [c for c in cands if c.get("runs") and c.get("nonnull")]
    return L or cands


def pick(cands):
    """scripts/analysis/valhard_eval.py's rule, verbatim (as devlook_score.py)."""
    u = [c for c in cands if c.get("sql")]
    if not u:
        return None
    return max(live(u), key=lambda c: c.get("rr", -1e9))


def versions() -> dict:
    from importlib import metadata
    out = {}
    for p in ("torch", "vllm", "transformers", "peft", "accelerate"):
        try:
            out[p] = metadata.version(p)
        except Exception:
            out[p] = None
    return out


def run(a) -> int:
    import collections
    import subprocess

    import torch
    import blind_gen as BG
    import repair_gen as RG
    import rerank_sq2 as R
    from luthor.sq.candidates import ValueIndex, find_candidates
    from luthor.sq.plan import compile_plan
    from luthor.sq2.model import TorchScores
    from luthor.sq2.targets import decode
    from luthor.sq2.topk import candidates
    from luthor.sq2.train import encode_db, question_body, select_cands, serve_chunks
    EV = RG.EV

    torch.backends.cuda.enable_cudnn_sdp(False)  # as rerank_sq2.main
    torch.manual_seed(0)
    T0 = time.time()
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    data: Path = a.data
    smp = json.loads((data / "sample.json").read_text())
    warm_ids, meas_ids = smp["warmup"], smp["measured"]
    if a.n:
        meas_ids = meas_ids[:a.n]
    a.plans = [data / "qs"]
    a.rr = data / "rr_valhard"
    a.bird_desc = ""
    a.grammar = getattr(a, "grammar", None)
    a.base, a.subfolder = "seeklhy/OmniSQL-7B", ""
    for f in FORBIDDEN:
        for p in (a.run, data, *a.db_roots):
            assert f not in str(p).lower(), f"refusing a {f} path: {p}"

    def sync():
        torch.cuda.synchronize()

    def now():
        sync()
        return time.perf_counter()

    dev = torch.cuda.get_device_properties(0)
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:
        drv = None
    env = {"gpu": torch.cuda.get_device_name(0), "gpu_memory_total_gb": round(dev.total_memory / 2**30, 2),
           "gpu_count": torch.cuda.device_count(), "driver": drv, "cuda": torch.version.cuda, "python": sys.version.split()[0],
           "cpu_count": os.cpu_count(), "workers": a.workers, **{f"{k}_version": v for k, v in versions().items()}}
    print(f"env {json.dumps(env)}", flush=True)

    # ---- questions (label-free rows), databases ----
    R.use_run_settings(a)  # grammar check, vocab.json, candidate order (as rerank_sq2 gen)
    cards, exs, rows = BG._load(a)
    by_id = {e.id: e for e in exs}
    missing = [i for i in warm_ids + meas_ids if i not in by_id]
    assert not missing, f"{len(missing)} sampled ids not loaded (e.g. {missing[:3]})"
    paths = BG._paths(a.db_roots)
    card_paths = R.load_cards(a.plans)
    dbs = sorted({by_id[i].db_id for i in warm_ids + meas_ids})
    assert all(db in paths for db in dbs), f"missing SQLite for {[d for d in dbs if d not in paths]}"
    print(f"{len(meas_ids)} measured + {len(warm_ids)} warm-up questions on {len(dbs)} DBs", flush=True)

    # ---- model load (excluded from every latency) ----
    torch.cuda.reset_peak_memory_stats()
    t = now()
    m = R.load_slot_model(a.run, a.base, a.subfolder)
    load_gen = now() - t
    mem_gen = torch.cuda.memory_allocated() / 2**30
    t = now()
    R.validator_settings(a.rr)
    rr = R.load_validator(a.rr, a.base, a.subfolder, a.lora_r)
    rr.eval()
    load_val = now() - t
    mem_both = torch.cuda.memory_allocated() / 2**30
    load = {"generator_s": round(load_gen, 2), "validator_s": round(load_val, 2),
            "generator_gb": round(mem_gen, 2), "generator_plus_validator_gb": round(mem_both, 2)}
    print(f"model load {json.dumps(load)}", flush=True)

    # ---- per-DB setup (built once, excluded, reported): value index, column list, schema tokens ----
    import sqlite3
    vidx, cols, per_db = {}, {}, {}
    for db in dbs:
        t = time.perf_counter()
        con = sqlite3.connect(f"file:{paths[db]}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "replace")
        try:
            vidx[db] = ValueIndex.build(con, cards[db])
        finally:
            con.close()
        t1 = time.perf_counter()
        cols[db] = set(EV.db_columns_path(paths[db]))
        t2 = time.perf_counter()
        per_db[db] = {"value_index_s": round(t1 - t, 3), "columns_s": round(t2 - t1, 3)}
    print(f"per-DB setup {json.dumps(per_db)}", flush=True)

    def lookup(e):
        card = cards[e.db_id]
        body = question_body({"question": e.question, "context": e.context})
        e.cands, _ = select_cands(find_candidates(body, card, extra=vidx[e.db_id].match(body)), order=a.cand_order, body=body)

    def execute(db, sqls):
        return BG._exec_blind((paths[db], [sqls], a.seconds))[0]

    # ---- single query (batch 1) ----
    def one(e, mode: str) -> dict:
        """mode greedy | k32 | k64r -> {stage: seconds, total, n_cands, ...}; every region synchronised."""
        st_t = collections.OrderedDict()
        card = cards[e.db_id]
        g = card.graph()
        cfn = lambda p: compile_plan(p, g)  # noqa: E731
        seed = zlib.crc32(e.id.encode())
        info = {}
        t_start = now()
        t = t_start
        lookup(e)
        t1 = now(); st_t["lookup"] = t1 - t; t = t1
        with torch.no_grad():
            tabs, _chunk = next(iter(serve_chunks(card, [e], 1)))
            st, oix, enc = encode_db(m, card, [e], tabs)
            x = enc[0]
            sc = TorchScores(m, x, st.option_ids, e.cands)
            if mode == "greedy":
                try:
                    plans = [(decode(sc), 0.0)]
                except Exception:
                    plans = []
            else:
                try:
                    plans = candidates(sc, k=32 if mode == "k32" else 64, with_cost=True, wide_singles=0, wide_pairs=0,
                                       runner_up2=0, samples=0, temp=1.0, seed=seed, distinct=True, compile_fn=cfn)
                except Exception:
                    plans = []
            t1 = now(); st_t["generate"] = t1 - t; t = t1
            cl = []
            for rank, (p, cost) in enumerate(plans):
                try:
                    sql = compile_plan(p, g)
                except Exception:
                    sql = None
                cl.append({"rank": rank, "cost": round(cost, 4), "sql": sql, "src": mode})
            t1 = now(); st_t["compile"] = t1 - t; t = t1
            if mode == "greedy":
                info["gen_only_s"] = st_t["lookup"] + st_t["generate"] + st_t["compile"]
            for c, (runs, nonnull, sig, prev) in zip(cl, execute(e.db_id, [c["sql"] for c in cl])):
                c.update(runs=runs, nonnull=nonnull, sig=sig, res=prev)
            t1 = now(); st_t["execute"] = t1 - t; t = t1
            rec = {"id": e.id, "db_id": e.db_id, "question": e.question, "context": e.context, "cands": cl}
            info["n_cands"] = len(cl)
            if mode == "k64r":
                trig = RG.triggers(rec, cols[e.db_id], a.ev_top)
                got = []
                if trig:
                    atoms = EV.parse_evidence(e.context or "", cols[e.db_id], e.question)
                    miss = RG._missing_in_top(atoms, rec, a.ev_top)
                    sc2 = TorchScores(m, x, st.option_ids, e.cands)  # the encoder output is reused
                    got = RG.repair_plans(rec, trig, sc2, cfn, k_rep=a.k_rep, hot_samples=a.hot_samples,
                                          hot_temp=a.hot_temp, atoms=atoms, missing=miss, boost=a.boost, seed=seed,
                                          max_extra=a.max_extra)
                t1 = now(); st_t["repair"] = t1 - t; t = t1
                extra = [{"rank": 0, "cost": cost, "sql": sql, "src": "repair", "repair": v} for v, sql, cost in got]
                if extra:
                    for c, (runs, nonnull, sig, prev) in zip(extra, execute(e.db_id, [c["sql"] for c in extra])):
                        c.update(runs=runs, nonnull=nonnull, sig=sig, res=prev)
                seen = {c.get("sql") for c in cl}  # merge (blind_gen.merge: dedup by SQL, appended)
                for c in extra:
                    if c.get("sql") and c["sql"] not in seen:
                        seen.add(c["sql"])
                        cl.append(dict(c, rank=len(cl)))
                t1 = now(); st_t["repair_execute"] = t1 - t; t = t1
                info.update(triggers=trig, n_repair=len(extra), n_cands=len(cl))
            if mode != "greedy":
                R.score_file(rr, [rec], card_paths, a.budget_chars)
                t1 = now(); st_t["validate"] = t1 - t; t = t1
                p = pick(rec["cands"])
                t1 = now(); st_t["pick"] = t1 - t; t = t1
                info["picked"] = p is not None
        info["total"] = t - t_start
        return {**{k: v for k, v in st_t.items()}, **info}

    MODES = ("greedy", "k32", "k64r")
    torch.cuda.reset_peak_memory_stats()
    for i in warm_ids:  # warm-up: every path, dropped
        for mode in MODES:
            one(by_id[i], mode)
    print(f"warm-up done ({len(warm_ids)} questions x {len(MODES)} paths) at {time.time() - T0:.0f}s", flush=True)
    sq_rows, t_sq = [], time.time()
    stopped = None
    with open(out / "rows.jsonl", "w") as fh:
        for n, i in enumerate(meas_ids):
            e = by_id[i]
            row = {"id": i, "db_id": e.db_id}
            for mode in MODES:
                row[mode] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in one(e, mode).items()}
            sq_rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if (n + 1) % 20 == 0:
                print(f"  single query {n + 1}/{len(meas_ids)}: k32 p50 "
                      f"{pct([r['k32']['total'] for r in sq_rows], 50):.2f}s, k64r p50 "
                      f"{pct([r['k64r']['total'] for r in sq_rows], 50):.2f}s, {time.time() - t_sq:.0f}s", flush=True)
            if a.sq_max_min and time.time() - t_sq > 60 * a.sq_max_min and n + 1 < len(meas_ids):
                stopped = f"time cap {a.sq_max_min} min after {n + 1} questions"
                print(f"single query stopped: {stopped}", flush=True)
                break
    peak_sq = {"allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    single = {"n_questions": len(sq_rows), "stopped": stopped, "wall_s": round(time.time() - t_sq, 1)}
    stages = {"greedy": ["lookup", "generate", "compile", "execute"],
              "k32": ["lookup", "generate", "compile", "execute", "validate", "pick"],
              "k64r": ["lookup", "generate", "compile", "execute", "repair", "repair_execute", "validate", "pick"]}
    for mode in MODES:
        rs = [r[mode] for r in sq_rows]
        d = {"end_to_end_s": summary([r["total"] for r in rs]),
             "stages_s": {s: summary([r.get(s, 0.0) for r in rs]) for s in stages[mode]},
             "candidates": summary([float(r["n_cands"]) for r in rs])}
        if mode == "greedy":
            d["generation_only_s"] = summary([r["gen_only_s"] for r in rs])
        if mode == "k64r":
            d["triggered"] = sum(bool(r.get("triggers")) for r in rs)
            d["repair_candidates"] = summary([float(r.get("n_repair", 0)) for r in rs])
            d["triggers"] = dict(collections.Counter(x for r in rs for x in r.get("triggers") or []))
        if a.price_h:
            d["usd_per_1000_serial"] = round(a.price_h / 3600 * d["end_to_end_s"]["mean"] * 1000, 3)
        single[mode] = d
    print(f"single query done in {single['wall_s']:.0f}s", flush=True)

    # ---- throughput: the chains' batched paths on the same questions ----
    qs = [by_id[i] for i in meas_ids]

    def batch_gen(k, src):
        """blind_gen.gen without the model load / file write: per DB value candidates, chunks of 8, lazy k-best
        --distinct, compile; then execution in a Pool of a.workers processes."""
        by_db = collections.defaultdict(list)
        for e in qs:
            by_db[e.db_id].append(e)
        recs, tasks = [], []
        with torch.no_grad():
            for db, group in by_db.items():
                card = cards[db]
                g = card.graph()
                try:
                    BG._value_cands(a, card, group, paths[db])
                except Exception as ex:
                    print(f"value index failed on {db}: {ex}", flush=True)
                qq = []
                for tabs, chunk in serve_chunks(card, group, 8):
                    st, oix, enc = encode_db(m, card, chunk, tabs)
                    for e, x in zip(chunk, enc):
                        sc = TorchScores(m, x, st.option_ids, e.cands)
                        try:
                            pc = candidates(sc, k=k, with_cost=True, wide_singles=0, wide_pairs=0, runner_up2=0,
                                            samples=0, temp=1.0, seed=zlib.crc32(e.id.encode()), distinct=True,
                                            compile_fn=lambda p: compile_plan(p, g))
                        except Exception:
                            pc = []
                        cl = []
                        for rank, (p, cost) in enumerate(pc):
                            try:
                                sql = compile_plan(p, g)
                            except Exception:
                                sql = None
                            cl.append({"rank": rank, "cost": round(cost, 4), "sql": sql, "src": src})
                        recs.append({"id": e.id, "db_id": db, "question": e.question, "context": e.context, "cands": cl})
                        qq.append([c["sql"] for c in cl])
                tasks.append((paths[db], qq, a.seconds))
        return recs, tasks

    def batch_repair(pool):
        """blind_gen.repair without the model load / file write (re-encodes the triggered questions, as the chain)."""
        recs, tasks = [], []
        todo = {}
        for r in pool:
            t = RG.triggers(r, set(EV.db_columns_path(paths[r["db_id"]])), a.ev_top)
            if t:
                todo[r["id"]] = (r, t)
        by_db = collections.defaultdict(list)
        for e in qs:
            if e.id in todo:
                by_db[e.db_id].append(e)
        with torch.no_grad():
            for db, group in by_db.items():
                card = cards[db]
                g = card.graph()
                try:
                    BG._value_cands(a, card, group, paths[db])
                except Exception as ex:
                    print(f"value index failed on {db}: {ex}", flush=True)
                cc = set(EV.db_columns_path(paths[db]))
                qq = []
                for tabs, chunk in serve_chunks(card, group, 8):
                    st, oix, enc = encode_db(m, card, chunk, tabs)
                    for e, x in zip(chunk, enc):
                        rec, trig = todo[e.id]
                        sc = TorchScores(m, x, st.option_ids, e.cands)
                        atoms = EV.parse_evidence(rec.get("context") or "", cc, rec.get("question", ""))
                        miss = RG._missing_in_top(atoms, rec, a.ev_top)
                        got = RG.repair_plans(rec, trig, sc, lambda p: compile_plan(p, g), k_rep=a.k_rep,
                                              hot_samples=a.hot_samples, hot_temp=a.hot_temp, atoms=atoms, missing=miss,
                                              boost=a.boost, seed=zlib.crc32(e.id.encode()), max_extra=a.max_extra)
                        cl = [{"rank": i, "cost": cost, "sql": sql, "src": "repair", "repair": v}
                              for i, (v, sql, cost) in enumerate(got)]
                        recs.append({k: rec[k] for k in Q_KEYS} | {"triggers": trig, "cands": cl})
                        qq.append([c["sql"] for c in cl])
                tasks.append((paths[db], qq, a.seconds))
        return recs, tasks, len(todo)

    def merge(pool, extra):
        ex = collections.defaultdict(list)
        for r in extra:
            ex[r["id"]].extend(r["cands"])
        for r in pool:
            seen = {c.get("sql") for c in r["cands"]}
            for c in ex.get(r["id"], []):
                if c.get("sql") and c["sql"] not in seen:
                    seen.add(c["sql"])
                    r["cands"].append(dict(c, rank=len(r["cands"]), src="repair"))
        return pool

    def batch(mode):
        st_t = collections.OrderedDict()
        t0 = t = now()
        recs, tasks = batch_gen(64 if mode == "k64r" else 32, mode)
        t1 = now(); st_t["generate"] = t1 - t; t = t1
        BG._execute(a, recs, tasks)
        t1 = now(); st_t["execute"] = t1 - t; t = t1
        extra = {}
        if mode == "k64r":
            xr, xt, n_trig = batch_repair(recs)
            t1 = now(); st_t["repair"] = t1 - t; t = t1
            if xr:
                BG._execute(a, xr, xt)
            merge(recs, xr)
            t1 = now(); st_t["repair_execute"] = t1 - t; t = t1
            extra = {"triggered": n_trig, "repair_candidates": sum(len(r["cands"]) for r in xr)}
        R.score_file(rr, recs, card_paths, a.budget_chars)
        t1 = now(); st_t["validate"] = t1 - t; t = t1
        picks = [pick(r["cands"]) for r in recs]
        t1 = now(); st_t["pick"] = t1 - t; t = t1
        tot = t - t0
        with open(out / f"picks_{mode}.jsonl", "w") as fh:  # predictions only (no label exists here): GPU-parity checks
            for r, p in zip(recs, picks):
                fh.write(json.dumps({"id": r["id"], "sql": p and p["sql"], "n_cands": len(r["cands"])}) + "\n")
        d = {"n_questions": len(recs), "wall_s": round(tot, 2), "questions_per_s": round(len(recs) / tot, 4),
             "s_per_question": round(tot / len(recs), 4), "stages_s": {k: round(v, 2) for k, v in st_t.items()},
             "candidates_per_question": round(sum(len(r["cands"]) for r in recs) / len(recs), 2),
             "picked": sum(p is not None for p in picks), **extra}
        if a.price_h:
            d["usd_per_1000"] = round(a.price_h / 3600 * tot / len(recs) * 1000, 3)
        return d

    throughput = {"workers": a.workers, "encoder_chunk": 8, "validator_q_per_batch": 4}
    if not a.skip_throughput:
        torch.cuda.reset_peak_memory_stats()
        for mode in ("k32", "k64r"):
            throughput[mode] = batch(mode)
            print(f"throughput {mode} {json.dumps(throughput[mode])}", flush=True)
        throughput["peak_memory"] = {"allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                     "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:
        smi = None
    res = {
        "what": "System-1 latency (R1 generator + VALHARD validator); latency only, no accuracy",
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample": {"seed": smp["seed"], "split": "decision", "n_measured": len(meas_ids), "n_warmup": len(warm_ids),
                   "databases": dbs, "by_db": dict(collections.Counter(by_id[i].db_id for i in meas_ids))},
        "settings": {"run": str(a.run), "validator": "VALHARD (runs/phase1/valhard/valhard/rr), lora_r 16",
                     "grammar": os.environ.get("LUTHOR_GRAMMAR"), "seconds": a.seconds, "budget_chars": a.budget_chars,
                     "repair": {"ev_top": a.ev_top, "k_rep": a.k_rep, "hot_samples": a.hot_samples,
                                "hot_temp": a.hot_temp, "boost": a.boost, "max_extra": a.max_extra},
                     "price_h": a.price_h},
        "env": env, "model_load": load, "per_db_setup_s": per_db,
        "single_query": single, "throughput": throughput,
        "memory": {"single_query_peak": peak_sq, "nvidia_smi_used_mib_at_end": smi,
                   "after_load_allocated_gb": load["generator_plus_validator_gb"]},
        "notes": [
            "single query: batch 1, one question at a time, both models resident; model load, per-DB value index / "
            "column list and 3 warm-up questions excluded; torch.cuda.synchronize() around every region",
            "single query execution is serial in-process (one connection per question, 3 s limit per SQL, 100k rows)",
            "k64r single query reuses the k=64 encoder output for repair; the batched path re-encodes (as the chain)",
            "the k=64+repair pool is not topped up with a stored k=32 pool (repair_box.sh's k64_pool step); a live "
            "system has none",
            "rerank_sq2.score_file reloads the schema card per call (inside 'validate')",
            "throughput includes the per-DB value index build and the Pool start-up, as the chains do; excludes model load",
        ],
        "total_box_s": round(time.time() - T0, 1),
    }
    (out / "latency.json").write_text(json.dumps(res, indent=1))
    (out / "latency.txt").write_text(report(res))
    print(report(res), flush=True)
    return 0


def report(res: dict) -> str:
    L = []
    e = res["env"]
    L.append(f"System-1 latency (R1 + VALHARD), {res['sample']['n_measured']} DECISION questions "
             f"(seed {res['sample']['seed']!r}; {res['sample']['n_warmup']} warm-ups dropped). Latency only.")
    L.append(f"GPU {e['gpu']} ({e['gpu_memory_total_gb']} GB), driver {e['driver']}, CUDA {e['cuda']}, "
             f"torch {e.get('torch_version')}, vLLM {e.get('vllm_version')}, transformers {e.get('transformers_version')}, "
             f"peft {e.get('peft_version')}, {e['cpu_count']} CPUs")
    ml = res["model_load"]
    L.append(f"model load (excluded): generator {ml['generator_s']}s ({ml['generator_gb']} GB), validator "
             f"{ml['validator_s']}s (both {ml['generator_plus_validator_gb']} GB)")
    L.append("")
    sq = res["single_query"]
    L.append(f"SINGLE QUERY (batch 1), n={sq['n_questions']}" + (f"  [stopped: {sq['stopped']}]" if sq.get("stopped") else ""))
    L.append(f"  {'path':8} {'p50':>8} {'p90':>8} {'mean':>8} {'max':>8}  cands  $/1000 (serial)")
    for mode, name in (("greedy", "k=1"), ("k32", "k=32"), ("k64r", "k64+rep")):
        d = sq.get(mode)
        if not d or not d["end_to_end_s"].get("n"):
            continue
        s = d["end_to_end_s"]
        L.append(f"  {name:8} {s['p50']:8.3f} {s['p90']:8.3f} {s['mean']:8.3f} {s['max']:8.3f}  "
                 f"{d['candidates']['mean']:5.1f}  {d.get('usd_per_1000_serial', '')}")
    for mode in ("greedy", "k32", "k64r"):
        d = sq.get(mode)
        if not d or not d["end_to_end_s"].get("n"):
            continue
        L.append(f"  {mode} stages (p50 / p90 / mean s): " + ", ".join(
            f"{k} {v['p50']:.3f}/{v['p90']:.3f}/{v['mean']:.3f}" for k, v in d["stages_s"].items()))
    if sq.get("greedy", {}).get("generation_only_s", {}).get("n"):
        g = sq["greedy"]["generation_only_s"]
        L.append(f"  greedy generation only (lookup+encode+decode+compile): p50 {g['p50']:.3f} p90 {g['p90']:.3f} "
                 f"mean {g['mean']:.3f}")
    if sq.get("k64r", {}).get("triggered") is not None:
        L.append(f"  repair triggered on {sq['k64r']['triggered']} of {sq['n_questions']} ({sq['k64r']['triggers']})")
    L.append("")
    tp = res["throughput"]
    L.append(f"THROUGHPUT (chains' batched paths; {tp['workers']} exec workers, encoder chunks of 8, validator 4 q/batch)")
    for mode in ("k32", "k64r"):
        d = tp.get(mode)
        if not d:
            continue
        L.append(f"  {mode:5} {d['questions_per_s']:.3f} q/s ({d['s_per_question']:.3f} s/q, {d['wall_s']}s for "
                 f"{d['n_questions']})  ${d.get('usd_per_1000', '-')}/1000 q  stages {d['stages_s']}")
    if tp.get("peak_memory"):
        L.append(f"  peak GPU memory (batched): {tp['peak_memory']}")
    L.append(f"peak GPU memory (single query): {res['memory']['single_query_peak']}; nvidia-smi at end "
             f"{res['memory']['nvidia_smi_used_mib_at_end']} MiB")
    L.append(f"price ${res['settings']['price_h']}/h; box time in this script {res['total_box_s']}s")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--out", type=Path, default=OUT)
    s.add_argument("--plans", type=Path, default=PLANS)
    s.add_argument("--cards", type=Path, default=CARDS)
    s.add_argument("--rr", type=Path, default=VALHARD, help="VALHARD dir to copy ('' = skip)")
    s.add_argument("--seed", default=SEED)
    s.add_argument("--n", type=int, default=N_MEASURE)
    s.add_argument("--n-warm", type=int, default=N_WARM)
    r = sub.add_parser("run")
    r.add_argument("--run", type=Path, required=True, help="R1 checkpoint (best_lora, best_heads.pt, args.json, vocab.json)")
    r.add_argument("--data", type=Path, required=True, help="the staged dir (qs/, sample.json, rr_valhard/)")
    r.add_argument("--db-roots", nargs="+", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--price-h", type=float, default=0.0)
    r.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    r.add_argument("--n", type=int, default=0, help="measure only the first n sampled questions (0 = all)")
    r.add_argument("--sq-max-min", type=float, default=0.0, help="stop the single-query loop after this many minutes")
    r.add_argument("--skip-throughput", action="store_true")
    r.add_argument("--seconds", type=float, default=3.0)
    r.add_argument("--budget-chars", type=int, default=9000)
    r.add_argument("--lora-r", type=int, default=16)
    r.add_argument("--ev-top", type=int, default=8)
    r.add_argument("--k-rep", type=int, default=8)
    r.add_argument("--hot-samples", type=int, default=16)
    r.add_argument("--hot-temp", type=float, default=1.5)
    r.add_argument("--boost", type=float, default=30.0)
    r.add_argument("--max-extra", type=int, default=24)
    from luthor import grammar_cli
    grammar_cli.add_argument(r)
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        a.rr = a.rr if str(a.rr) not in ("", ".") else None
        return stage(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
