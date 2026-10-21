"""Teacher-data pipeline (Mac; hosted teacher via the local ollama daemon or an OpenAI-compatible API).

  label   (db schema, question, evidence) -> teacher SQL, n samples per question at a temperature. Prompt: OmniSQL's
          own template (sqlw/prompt6.think_messages: every table as CREATE TABLE with descriptions, sample values,
          declared + inferred join keys; evidence then question). Gold SQL never goes into a prompt.
  write   a plan's SQL (data/rare1 mutants, SynSQL plans, ...) -> a natural question + BIRD-style evidence
          (scripts/qwrite.py's prompt: touched-table schema + SQL + sample result rows + 3 real BIRD-train style shots).
          `verify` on a write run emits <run>/questions.jsonl: label it with a checker model; only round trips whose
          SQL returns the source SQL's result are exported (the source SQL is then the training label).
  verify  execute every sample read-only (parallel, time limit); gold match when the row has gold SQL, otherwise
          majority vote over the samples by result signature with the agreement recorded; then plan
          convertibility (luthor.sq.convert, grammar v3; --grammar v4 once the v4 converter is merged).
  export  verified rows -> sharded plans-NNNNN.jsonl in the plan-set schema (train_sq2.py --plans / --extra-train)
          + cards/ + manifest.json. Refuses eval-only runs (the bake-off), held-out / sealed / BIRD-dev databases,
          and, unless --allow-unchecked-contamination, runs when luthor.contamination is unavailable.
  usage   tokens per provider/model per day and ISO week (data/teacher/usage.jsonl).
  bakeoff-input / bakeoff-report   the Phase-0 teacher bake-off on the held-out BIRD-train questions (eval only).

  python scripts/teacher/teacher.py label --input X.jsonl --run data/teacher/runs/NAME [--model deepseek-v4.1-flash:cloud]
         [--n 4 --temp 0.7] [--workers 8] [--limit N] [--week-cap 400000000]
  python scripts/teacher/teacher.py verify --run data/teacher/runs/NAME
  python scripts/teacher/teacher.py export --runs data/teacher/runs/A ... --out data/teacher/export/NAME

Input rows (jsonl): id, db_id, question, context (evidence, may be empty), gold_sql (optional), source.
Write inputs: id, db_id, compiled_sql (or gold_sql), plan; schema / sample_rows are filled in when missing.
Resumable: generations are appended as they arrive (gen/part-NN.jsonl); re-running a command skips done samples.
Rate limits: every worker pauses together and waits out the window (src/luthor/teacher/providers.py).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from luthor.teacher import guards, verify  # noqa: E402
from luthor.teacher.leak import LeakGate  # noqa: E402
from luthor.teacher.ledger import CapReached, Ledger  # noqa: E402
from luthor.teacher.providers import GaveUp, LeakRefused, make  # noqa: E402

D = REPO / "data"
TEACHER = D / "teacher"
DEFAULT_MODEL = "deepseek-v4.1-flash:cloud"
# Owner decision 2026-10-08: GLM burns too much of the subscription quota. The adapter stays generic, but GLM
# models are refused unless LUTHOR_ALLOW_GLM=1 is set explicitly.
BLOCKED_MODEL_PREFIXES = ("glm",)
LABEL_PROMPT = "label-omnisql-think-v1"
S1_PROMPT_SUFFIX = "+s1cands-v1"  # label --s1-cands: System 1's candidates appended (eval-only experiments)
S1_FINAL = "Take a deep breath and think step by step to find the correct SQL query."
WRITE_PROMPT = "write-qwrite-v1"
CARD_DIRS = [D / "sys1/bird_train/cards", D / "sys1/spider_train/cards", D / "sys1/synsql_train/cards"]
DB_ROOTS = [D / "sys1/dbs", D / "sys1/rrdbs", D / "raw/bird_train/train_databases", D / "raw/spider_data/database",
            D / "raw/synsql/databases"]
TEMPLATES = D / "sys1/templates.json"
BAKEOFF_SRC = REPO / "runs/r1r2/r2/rr/dev_scored.jsonl"
BAKEOFF_EXCLUDE = {"retail_world"}  # its gold SQL fails


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def part_of(qid: str, n: int) -> int:
    return int(hashlib.md5(qid.encode()).hexdigest()[:8], 16) % n


class Cards:
    def __init__(self, dirs):
        self.dirs = [Path(d) for d in dirs]
        self.cache, self.lock = {}, threading.Lock()

    def path(self, db: str) -> Path | None:
        return next((d / f"{db}.json" for d in self.dirs if (d / f"{db}.json").exists()), None)

    def get(self, db: str):
        from luthor.sq.schema_card import SchemaCard
        with self.lock:
            if db not in self.cache:
                p = self.path(db)
                self.cache[db] = SchemaCard.load(p) if p else None
            return self.cache[db]


# ---- generation ----------------------------------------------------------------------------------------------
class GenStore:
    """<run>/gen/part-NN.jsonl, one line per (id, sample_idx); appended under a lock, read back for resume."""

    def __init__(self, run: Path, parts: int):
        self.dir = run / "gen"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.parts = parts
        self.locks = [threading.Lock() for _ in range(parts)]

    def done(self) -> set:
        """Keys never retried: answered, or refused by a guard (errors are retried on the next run)."""
        out = set()
        for p in self.dir.glob("part-*.jsonl"):
            for r in read_jsonl(p):
                if r.get("status") in ("ok", "leak_refused", "refused"):
                    out.add((r["id"], r["sample_idx"]))
        return out

    def add(self, rec: dict) -> None:
        k = part_of(rec["id"], self.parts)
        with self.locks[k], open(self.dir / f"part-{k:02d}.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def all(self) -> list[dict]:
        """Latest record per (id, sample_idx)."""
        out = {}
        for p in sorted(self.dir.glob("part-*.jsonl")):
            for r in read_jsonl(p):
                k = (r["id"], r["sample_idx"])
                if k not in out or out[k].get("status") != "ok":
                    out[k] = r
        return list(out.values())


def check_model(model: str) -> None:
    base = model.removeprefix("ollama:").lower()
    if base.startswith(BLOCKED_MODEL_PREFIXES) and os.environ.get("LUTHOR_ALLOW_GLM") != "1":
        raise SystemExit(f"refusing teacher model {model}: GLM is off by owner decision (quota); "
                         f"use {DEFAULT_MODEL}")


def load_inputs(a) -> list[dict]:
    guards.check_input_path(a.input)
    bird_dev = guards.bird_dev_dbs()
    rows, refused = [], collections.Counter()
    for r in read_jsonl(a.input):
        try:
            guards.check_prompt_row(r, bird_dev)
        except guards.Refused as e:
            refused[str(e).split(" (")[0]] += 1
            continue
        rows.append(r)
    if refused:
        print(f"REFUSED (never prompted): {sum(refused.values())} rows {dict(refused)}", flush=True)
    if a.only_dbs:
        rows = [r for r in rows if r["db_id"] in set(a.only_dbs)]
    return rows[:a.limit] if a.limit else rows


def write_run_json(run: Path, a, job: str, n_rows: int) -> dict:
    p = run / "run.json"
    meta = json.loads(p.read_text()) if p.exists() else {"created": now_iso(), "sessions": []}
    pv = LABEL_PROMPT + (S1_PROMPT_SUFFIX if getattr(a, "s1_cands", None) else "") if job == "label" else WRITE_PROMPT
    meta.update({"job": job, "model": a.model, "prompt_version": pv,
                 "temperature": a.temp, "n": a.n, "think": a.think, "input": str(a.input), "max_tokens": a.max_tokens,
                 "eval_only": bool(a.eval_only or meta.get("eval_only"))})
    if getattr(a, "s1_cands", None):
        meta["s1_cands"] = str(a.s1_cands)
    meta["sessions"].append({"start": now_iso(), "rows": n_rows})
    run.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=1))
    return meta


def generate(a, job: str) -> None:
    check_model(a.model)
    run = Path(a.run)
    rows = load_inputs(a)
    cards = Cards(a.cards)
    near = {}
    for db in sorted({r["db_id"] for r in rows}):
        card = cards.get(db)
        near[db] = guards.near_dev_schema(db, [t.name for t in card.tables] if card else [])
    bad = {db: d for db, d in near.items() if d}
    if bad:
        print(f"REFUSED (never prompted): {sum(r['db_id'] in bad for r in rows)} rows on databases near a BIRD dev "
              f"schema {bad}", flush=True)
        rows = [r for r in rows if r["db_id"] not in bad]
    held = guards.held_out_dbs()
    if getattr(a, "s1_cands", None) and not a.eval_only:
        raise SystemExit("--s1-cands runs are evaluation experiments: pass --eval-only")
    if any(r["db_id"] in held for r in rows) and not a.eval_only:
        print("NOTE: input has held-out evaluation databases -> run marked eval_only (export will refuse it)", flush=True)
        a.eval_only = True
    meta = write_run_json(run, a, job, len(rows))
    store = GenStore(run, a.parts)
    done = store.done()
    gate = LeakGate()
    ledger = Ledger(Path(a.ledger), a.day_cap, a.week_cap)
    prov = make(a.model, gate, ledger=ledger, job=f"{job}:{run.name}", timeout=a.timeout, ollama_url=a.ollama_url)
    builder = label_builder(cards, load_s1_cands(a.s1_cands) if a.s1_cands else None) if job == "label" \
        else write_builder(a, cards)
    tasks = [(r, k) for r in rows for k in range(a.n) if (r["id"], k) not in done]
    print(f"{job.upper()} {run} model={a.model} rows={len(rows)} samples todo={len(tasks)} (done {len(done)})", flush=True)
    if not tasks:
        return
    stop = threading.Event()
    stats = collections.Counter()
    t0 = time.time()
    tok = collections.Counter()

    def one(task):
        r, k = task
        if stop.is_set():
            return
        base = {"id": r["id"], "db_id": r["db_id"], "source": r.get("source", ""), "sample_idx": k,
                "teacher_model": a.model, "provider": prov.name, "prompt_version": meta["prompt_version"],
                "temperature": a.temp, "think": a.think, "ts": now_iso()}
        try:
            msgs = builder(r)
        except Exception as e:
            store.add({**base, "status": "refused", "error": f"prompt: {type(e).__name__}: {str(e)[:120]}"})
            stats["prompt_error"] += 1
            return
        try:
            rep = prov.chat(msgs, temperature=a.temp, max_tokens=a.max_tokens,
                            seed=(a.seed + k) if a.temp > 0 else a.seed, think=a.think)
        except LeakRefused as e:
            store.add({**base, "status": "leak_refused", "error": str(e)})
            stats["leak_refused"] += 1
            return
        except (CapReached, GaveUp) as e:
            print("STOP:", e, flush=True)
            stop.set()
            return
        except Exception as e:
            store.add({**base, "status": "error", "error": str(e)[:300]})
            stats["error"] += 1
            return
        rec = {**base, "status": "ok", "text": rep.text, "thinking": rep.thinking, "n_in": rep.n_in, "n_out": rep.n_out,
               "seconds": round(rep.seconds, 2)}
        if job == "label":
            from luthor.sqlw.prompt6 import extract_sql
            rec["sql"] = extract_sql(rep.text)
        else:
            from qwrite import parse_written
            w = parse_written(rep.text)
            rec["written"] = w
            rec["source_sql"] = r.get("compiled_sql") or r.get("gold_sql")
        store.add(rec)
        stats["ok"] += 1
        tok["in"] += rep.n_in
        tok["out"] += rep.n_out
        n = stats["ok"]
        if n % a.log_every == 0:
            el = time.time() - t0
            print(f"  {n}/{len(tasks)} ok, in={tok['in']:,} out={tok['out']:,} {el / 60:.1f} min "
                  f"({n / el * 60:.0f}/min) {dict(stats)}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, tasks))
    el = time.time() - t0
    meta = json.loads((run / "run.json").read_text())
    meta["sessions"][-1].update({"end": now_iso(), "seconds": round(el, 1), "tokens_in": tok["in"],
                                 "tokens_out": tok["out"], "stats": dict(stats), "stopped": stop.is_set()})
    (run / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"{job.upper()} done {dict(stats)} in={tok['in']:,} out={tok['out']:,} {el / 60:.1f} min"
          + (" (STOPPED: re-run to resume)" if stop.is_set() else ""), flush=True)
    if stop.is_set():
        sys.exit(3)


def label_builder(cards: Cards, s1: dict | None = None):
    """s1 (optional, --s1-cands): question id -> System 1's candidate list; its block goes before the final line."""
    from luthor.sqlw.prompt6 import think_messages

    def build(r):
        card = cards.get(r["db_id"])
        if card is None:
            raise FileNotFoundError(f"no card for {r['db_id']}")
        msgs = think_messages(card, r["question"], r.get("context") or "")
        if s1 is not None:
            if r["id"] not in s1:
                raise KeyError(f"{r['id']} not in --s1-cands")
            msgs = add_s1_block(msgs, s1[r["id"]])
        return msgs
    return build


def load_s1_cands(path) -> dict:
    """runs/analysis/hybrid/s1_cands_decision.jsonl: question_id -> [{label, sql, result, n_of_32}] best first."""
    return {r["question_id"]: r.get("candidates") or [] for r in read_jsonl(path)}


def s1_block(cands: list[dict]) -> str:
    if not cands:
        return ("A fast text-to-SQL system was run on this question but found no candidate queries. "
                "Answer the question normally.")
    lines = ["A fast text-to-SQL system proposed these candidate queries (grouped by identical result, best first by "
             "its own confidence; n_of_32 = how many of its 32 samples gave that result):"]
    for c in cands:
        lines.append(f"[{c['label']}] {c['sql']}")
        lines.append(f"    result: {c.get('result', '')}  (n_of_32={c.get('n_of_32')})")
    lines.append("If one candidate correctly answers the question, output that SQL exactly (in the code block as "
                 "above). Otherwise write your own query.")
    return "\n".join(lines)


def add_s1_block(msgs: list[dict], cands: list[dict]) -> list[dict]:
    """Insert System 1's candidate block before the prompt's final instruction line (output format unchanged)."""
    msgs = [dict(m) for m in msgs]
    body = msgs[-1]["content"]
    assert body.endswith(S1_FINAL), "label prompt no longer ends with the expected final instruction"
    msgs[-1]["content"] = body[:-len(S1_FINAL)] + s1_block(cands) + "\n\n" + S1_FINAL
    return msgs


def write_builder(a, cards: Cards):
    import build_rare_mix as brm
    import qwrite
    shots = qwrite.Shots()
    paths = verify.db_paths(a.db_roots)

    def build(r):
        r = dict(r)
        sql = r.get("compiled_sql") or r.get("gold_sql")
        r["compiled_sql"] = sql
        if not r.get("schema"):
            card = cards.get(r["db_id"])
            if card is None:
                raise FileNotFoundError(f"no card for {r['db_id']}")
            r["tables"], r["schema"] = brm.schema_snippet(card, sql)
        if "sample_rows" not in r:
            if r["db_id"] not in paths:
                raise FileNotFoundError(f"no db for {r['db_id']}")
            [(_, status, n, _sig, sample, _)] = brm.exec_task((paths[r["db_id"]], [(r["id"], sql)], a.seconds))
            if status.startswith("error"):
                raise RuntimeError(status)
            r["n_rows"], r["sample_rows"] = n, sample
        if "skel" not in r and r.get("plan"):
            r["skel"] = brm.skeleton(r["plan"])
        return qwrite.write_messages(r, shots.get(r))
    return build


# ---- verify --------------------------------------------------------------------------------------------------
def cmd_verify(a) -> None:
    run = Path(a.run)
    meta = json.loads((run / "run.json").read_text())
    gens = GenStore(run, 1).all()
    if meta["job"] == "write":
        return verify_write(a, run, meta, gens)
    inputs = {r["id"]: r for r in read_jsonl(meta["input"])}
    paths = verify.db_paths(a.db_roots)
    by_q = collections.defaultdict(list)
    for g in gens:
        by_q[g["id"]].append(g)
    items = []
    for qid, gs in by_q.items():
        db = inputs[qid]["db_id"]
        if inputs[qid].get("gold_sql"):
            items.append(((qid, "gold"), db, inputs[qid]["gold_sql"]))
        for g in gs:
            if g.get("status") == "ok":
                items.append(((qid, g["sample_idx"]), db, g.get("sql") or ""))
    t0 = time.time()
    res = verify.execute(items, paths, a.seconds, a.workers)
    print(f"VERIFY executed {len(items)} queries in {time.time() - t0:.0f}s", flush=True)
    out, plan_jobs = [], collections.defaultdict(list)
    for qid, gs in sorted(by_q.items()):
        q = inputs[qid]
        gs.sort(key=lambda g: g["sample_idx"])
        samples = []
        for g in gs:
            st, sig, n = res.get((qid, g["sample_idx"]), (g.get("status"), "", 0))
            samples.append({"sample_idx": g["sample_idx"], "sql": g.get("sql", ""), "exec": st, "sig": sig,
                            "n_rows": n, "n_in": g.get("n_in", 0), "n_out": g.get("n_out", 0),
                            "seconds": g.get("seconds", 0), "ts": g.get("ts"), "gen_status": g.get("status"),
                            "provider": g.get("provider")})
        rec = {"id": qid, "db_id": q["db_id"], "source": q.get("source", ""), "question": q["question"],
               "context": q.get("context") or "", "gold_sql": q.get("gold_sql") or "", "samples": samples}
        for k in ("writer",):
            if q.get(k):
                rec[k] = q[k]
        if q.get("gold_sql"):
            gst, gsig, _ = res.get((qid, "gold"), ("no-db", "", 0))
            rec["gold_exec"], rec["gold_sig"] = gst, gsig
            for s in samples:
                s["match"] = bool(gsig) and s["sig"] == gsig
            if gst != "ok":
                rec["verification"], kept = "gold_failed", []
            else:
                kept = [s for s in samples if s["match"]]
                rec["verification"] = "gold_match" if kept else "gold_mismatch"
            rec["result_sig"] = gsig
            rec["agreement"] = round(sum(s["match"] for s in samples) / max(1, len(samples)), 4)
        else:
            sig, agree, votes = verify.vote([s["sig"] for s in samples])
            rec["result_sig"], rec["agreement"], rec["votes"] = sig, round(agree, 4), votes
            if len(samples) < 2:
                rec["verification"], kept = "single_unverified", []
            elif sig and agree > a.min_agree and votes >= 2:
                rec["verification"], kept = "vote", [s for s in samples if s["sig"] == sig]
            else:
                rec["verification"], kept = "no_majority", []
        rec["kept"] = [s["sample_idx"] for s in kept]
        out.append(rec)
        if kept and not a.no_plans:
            sqls = list(dict.fromkeys(s["sql"] for s in kept))
            write_derived = str(rec["source"]).startswith("teacher_write")
            if rec["gold_sql"] and rec["verification"] == "gold_match":
                sqls = [rec["gold_sql"]] + sqls if write_derived else sqls + [rec["gold_sql"]]
            plan_jobs[q["db_id"]].append((qid, rec["question"], rec["context"], sqls))
    plans = {}
    if plan_jobs:
        cards = Cards(a.cards)
        tasks = []
        for db, lst in plan_jobs.items():
            cp = cards.path(db)
            if db not in paths or cp is None:
                for qid, *_ in lst:
                    plans[qid] = (None, ["no-db-or-card"])
                continue
            tasks.append((paths[db], str(cp), lst, a.grammar, str(a.templates) if a.templates else None))
        t0 = time.time()
        if a.workers > 1 and len(tasks) > 1:
            from multiprocessing import get_context
            with get_context("spawn").Pool(min(a.workers, len(tasks))) as p:
                for part in p.imap_unordered(verify.plan_rows_db, tasks):
                    plans.update({k: (row, why) for k, row, why in part})
        else:
            for t in tasks:
                plans.update({k: (row, why) for k, row, why in verify.plan_rows_db(t)})
        print(f"VERIFY plans for {sum(len(v) for v in plan_jobs.values())} questions in {time.time() - t0:.0f}s", flush=True)
    for rec in out:
        row, why = plans.get(rec["id"], (None, []))
        rec["plan_row"], rec["plan_reasons"] = row, why
        if row is not None:
            sqls = plan_jobs_sqls(plan_jobs, rec)
            rec["plan_sql"] = sqls[row["sql_index"]]
            rec["sql_origin"] = "gold" if rec["plan_sql"] == rec["gold_sql"] else "teacher"
        rec["grammar"] = a.grammar
    (run / "verified.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in out))
    st = collections.Counter(r["verification"] for r in out)
    pc = sum(r["plan_row"] is not None for r in out)
    summary = {"questions": len(out), "verification": dict(st), "plan_rows": pc, "grammar": a.grammar,
               "plan_reasons": dict(collections.Counter(r["plan_reasons"][-1] for r in out
                                                        if r["plan_row"] is None and r["plan_reasons"]))}
    (run / "verify_stats.json").write_text(json.dumps(summary, indent=1))
    print("VERIFIED", run / "verified.jsonl", json.dumps(summary), flush=True)


def plan_jobs_sqls(plan_jobs, rec) -> list[str]:
    for qid, _, _, sqls in plan_jobs.get(rec["db_id"], []):
        if qid == rec["id"]:
            return sqls
    return []


def verify_write(a, run: Path, meta: dict, gens: list[dict]) -> None:
    """Written (question, evidence) -> <run>/questions.jsonl in label-input format (gold = the source SQL)."""
    inputs = {r["id"]: r for r in read_jsonl(meta["input"])}
    out, seen, st = [], set(), collections.Counter()
    for g in sorted(gens, key=lambda g: (g["id"], g["sample_idx"])):
        w = g.get("written") if g.get("status") == "ok" else None
        if not w:
            st["unparsed" if g.get("status") == "ok" else g.get("status")] += 1
            continue
        key = (g["id"], w["question"].lower(), w["evidence"].lower())
        if key in seen:
            st["duplicate"] += 1
            continue
        seen.add(key)
        src = inputs[g["id"]]
        out.append({"id": f"{g['id']}__w{g['sample_idx']}", "db_id": g["db_id"], "question": w["question"],
                    "context": w["evidence"], "gold_sql": src.get("compiled_sql") or src.get("gold_sql"),
                    "source": f"teacher_write:{src.get('source') or 'plans'}", "plan_source_id": g["id"],
                    "writer": {k: g.get(k) for k in ("teacher_model", "provider", "prompt_version", "temperature",
                                                     "sample_idx", "ts", "n_in", "n_out")}})
        st["written"] += 1
    (run / "questions.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in out))
    print("WRITE-VERIFY", run / "questions.jsonl", dict(st), "-> label these with a checker model, then verify",
          flush=True)


# ---- export --------------------------------------------------------------------------------------------------
def cmd_export(a) -> None:
    held = guards.held_out_dbs()
    rows, st, cards_needed, runs_meta = [], collections.Counter(), set(), []
    for rd in a.runs:
        rd = Path(rd)
        meta = json.loads((rd / "run.json").read_text())
        if meta.get("eval_only") or "bakeoff" in str(rd.resolve()).lower():
            raise SystemExit(f"refusing {rd}: eval-only run (bake-off / held-out evaluation data is never training data)")
        if meta["job"] != "label":
            raise SystemExit(f"refusing {rd}: export takes label runs (verify a write run, then label its questions.jsonl)")
        runs_meta.append({"run": str(rd), **{k: meta.get(k) for k in ("model", "prompt_version", "temperature", "n",
                                                                       "input", "created")}})
        for r in read_jsonl(rd / "verified.jsonl"):
            if r["db_id"] in held:
                st["refused_held_out_db"] += 1
                continue
            if r.get("plan_row") is None:
                st["no_plan:" + r["verification"]] += 1
                continue
            kept = [s for s in r["samples"] if s["sample_idx"] in r["kept"]]
            prov = {"teacher_model": meta["model"], "provider": kept[0].get("provider") if kept else None,
                    "prompt_version": meta["prompt_version"], "temperature": meta["temperature"],
                    "sample_idx": [s["sample_idx"] for s in kept], "n_samples": len(r["samples"]),
                    "source_dataset": r["source"], "generated_at": min((s["ts"] or "") for s in kept) if kept else None,
                    "exported_at": now_iso(), "verification": r["verification"], "agreement": r["agreement"],
                    "result_sig": r["result_sig"], "sql_origin": r.get("sql_origin"), "grammar": r.get("grammar"),
                    "run": rd.name}
            if r.get("writer"):
                prov["writer"] = r["writer"]
            pr = {k: v for k, v in r["plan_row"].items() if k != "sql_index"}
            rows.append({"id": f"teacher_{rd.name}_{r['id']}", "source": f"teacher:{r['source']}", "split": "train",
                         "db_id": r["db_id"], "question": r["question"], "context": r["context"],
                         "gold_sql": r["plan_sql"], **pr, "teacher": prov})
            cards_needed.add(r["db_id"])
    cards = Cards(a.cards)
    near = {}
    for db in sorted({r["db_id"] for r in rows}):
        card = cards.get(db)
        near[db] = guards.near_dev_schema(db, [t.name for t in card.tables] if card else [])
    st["refused_near_dev_schema"] = sum(bool(near[r["db_id"]]) for r in rows)
    rows = [r for r in rows if not near[r["db_id"]]]
    flags = guards.near_dup_flags([(r["question"] + " " + (r["context"] or "")).strip() for r in rows])
    if flags is None:
        if not a.allow_unchecked_contamination:
            raise SystemExit("luthor.contamination is not importable: refusing to export without the dev near-duplicate "
                             "check (pass --allow-unchecked-contamination to override; the manifest records it)")
        contamination = "unchecked"
    else:
        st["refused_near_dup"] = sum(flags)
        rows = [r for r, f in zip(rows, flags) if not f]
        contamination = "luthor.contamination.near_dup"
    out = Path(a.out)
    if "bakeoff" in str(out.resolve()).lower():
        raise SystemExit("refusing to export into a bakeoff directory")
    (out / "cards").mkdir(parents=True, exist_ok=True)
    for db in sorted({r["db_id"] for r in rows}):
        shutil.copyfile(cards.path(db), out / "cards" / f"{db}.json")
    rows.sort(key=lambda r: r["id"])
    shards = []
    for i in range(0, max(1, len(rows)), a.shard_rows):
        chunk = rows[i:i + a.shard_rows]
        p = out / f"plans-{i // a.shard_rows:05d}.jsonl"
        data = "".join(json.dumps(r, default=str) + "\n" for r in chunk)
        p.write_text(data)
        shards.append({"file": p.name, "rows": len(chunk), "sha256": hashlib.sha256(data.encode()).hexdigest()})
    manifest = {"created": now_iso(), "rows": len(rows), "dbs": len({r["db_id"] for r in rows}), "shards": shards,
                "runs": runs_meta, "counts": dict(st), "contamination": contamination,
                "held_out_dbs_refused": len(held), "schema": "plan-set rows (prepare_plans.py) + teacher provenance",
                "load": "train_sq2.py --extra-train <shard.jsonl> (cards/ beside it)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("EXPORTED", out, json.dumps({k: manifest[k] for k in ("rows", "dbs", "counts", "contamination")}), flush=True)


# ---- usage / bake-off -----------------------------------------------------------------------------------------
def cmd_usage(a) -> None:
    print(json.dumps(Ledger(Path(a.ledger)).summary(), indent=1))


def cmd_bakeoff_input(a) -> None:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_jsonl(BAKEOFF_SRC) if r["db_id"] not in BAKEOFF_EXCLUDE]
    bird_dev = guards.bird_dev_dbs()
    assert not any(r["db_id"] in bird_dev for r in rows)
    out.write_text("".join(json.dumps({"id": r["id"], "db_id": r["db_id"], "question": r["question"],
                                       "context": r.get("context") or "", "gold_sql": r["gold_sql"],
                                       "source": "bird_train_heldout"}) + "\n" for r in rows))
    print("BAKEOFF input", out, len(rows), dict(collections.Counter(r["db_id"] for r in rows)))


def cmd_bakeoff_report(a) -> None:
    """EX vs gold per run (sample 0), tokens, wall time; plus System-1 (r2 greedy) on the same questions and
    OmniSQL-32B (greedy think_g / any-of-5) on the overlap with data/eval/sel_cands/sel/{train,val}_cands.jsonl."""
    rep = {"runs": {}}
    base = {r["id"]: r for r in read_jsonl(BAKEOFF_SRC) if r["db_id"] not in BAKEOFF_EXCLUDE}
    ids = set()
    per_run = {}
    for rd in a.runs:
        rd = Path(rd)
        meta = json.loads((rd / "run.json").read_text())
        ver = read_jsonl(rd / "verified.jsonl")
        ids |= {r["id"] for r in ver}
        ok = {r["id"]: any(s.get("match") for s in r["samples"] if s["sample_idx"] == 0) for r in ver}
        per_run[rd.name] = ok
        n_in = sum(s["n_in"] for r in ver for s in r["samples"])
        n_out = sum(s["n_out"] for r in ver for s in r["samples"])
        secs = [s["seconds"] for r in ver for s in r["samples"] if s["seconds"]]
        ex_st = collections.Counter(s["exec"] if not str(s["exec"]).startswith("error") else "error"
                                    for r in ver for s in r["samples"])
        wall = sum(x.get("seconds", 0) for x in meta.get("sessions", []))
        rep["runs"][rd.name] = {
            "model": meta["model"], "temperature": meta["temperature"], "n": len(ver),
            "ex": round(sum(ok.values()) / max(1, len(ver)), 4), "correct": sum(ok.values()),
            "exec_status": dict(ex_st),
            "gold_failed": sum(r["verification"] == "gold_failed" for r in ver),
            "plan_convertible_of_correct": sum(r.get("plan_row") is not None for r in ver if ok[r["id"]]),
            "plan_from_teacher_sql": sum(r.get("sql_origin") == "teacher" for r in ver if ok[r["id"]]),
            "tokens_in": n_in, "tokens_out": n_out, "tokens_out_mean": round(n_out / max(1, len(ver)), 1),
            "call_seconds_median": sorted(secs)[len(secs) // 2] if secs else None, "wall_seconds": round(wall, 1),
            "per_db": {db: round(sum(ok[r["id"]] for r in ver if r["db_id"] == db) /
                                 max(1, sum(r["db_id"] == db for r in ver)), 3)
                       for db in sorted({r["db_id"] for r in ver})}}
    s1 = {i: bool(base[i]["cands"] and base[i]["cands"][0].get("ok")) for i in ids if i in base}
    rep["system1_r2_greedy"] = {"n": len(s1), "ex": round(sum(s1.values()) / max(1, len(s1)), 4)}
    # OmniSQL-32B candidates (BIRD-verified question set; re-scored here against THIS gold by execution)
    key = {(base[i]["db_id"], base[i]["question"].strip()): i for i in ids if i in base}
    omni = {}
    for f in ("train_cands.jsonl", "val_cands.jsonl"):
        p = D / "eval/sel_cands/sel" / f
        if p.exists():
            for r in read_jsonl(p):
                i = key.get((r["db_id"], r["question"].strip()))
                if i is not None:
                    omni[i] = r["cands"]
    if omni:
        items = [((i, "gold"), base[i]["db_id"], base[i]["gold_sql"]) for i in omni]
        items += [((i, j), base[i]["db_id"], c.get("sql") or "") for i, cs in omni.items() for j, c in enumerate(cs)]
        res = verify.execute(items, verify.db_paths(DB_ROOTS), 15.0, a.workers)
        g_ok, any_ok = {}, {}
        for i, cs in omni.items():
            gs = res[(i, "gold")][1]
            m = [bool(gs) and res[(i, j)][1] == gs for j in range(len(cs))]
            greedy = [j for j, c in enumerate(cs) if str(c.get("src", "")).endswith("think_g")]
            g_ok[i] = m[greedy[0]] if greedy else m[0]
            any_ok[i] = any(m)
        sub = sorted(omni)
        rep["omnisql32b_overlap"] = {"n": len(sub), "greedy_ex": round(sum(g_ok.values()) / len(sub), 4),
                                     "any_of_5_ex": round(sum(any_ok.values()) / len(sub), 4),
                                     "system1_ex": round(sum(s1.get(i, False) for i in sub) / len(sub), 4),
                                     **{f"{k}_ex": round(sum(v.get(i, False) for i in sub) / len(sub), 4)
                                        for k, v in per_run.items()}}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ---- cli -----------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cards", nargs="+", default=[str(c) for c in CARD_DIRS])
        p.add_argument("--db-roots", nargs="+", default=[str(r) for r in DB_ROOTS])
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--seconds", type=float, default=10.0, help="SQLite time limit per query")
        p.add_argument("--ledger", default=str(TEACHER / "usage.jsonl"))

    for job in ("label", "write"):
        p = sub.add_parser(job)
        common(p)
        p.add_argument("--input", required=True)
        p.add_argument("--run", required=True, help="run directory (resumable)")
        p.add_argument("--model", default=DEFAULT_MODEL, help="ollama model, or openai:<model>@<base_url>#<KEY_ENV>")
        p.add_argument("--ollama-url", default="http://localhost:11434")
        p.add_argument("--n", type=int, default=1 if job == "label" else 2, help="samples per row")
        p.add_argument("--temp", type=float, default=0.0 if job == "label" else 0.7)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--think", type=lambda s: {"true": True, "false": False, "none": None}[s.lower()],
                       default=False, help="ollama think flag: true / false / none (model default)")
        p.add_argument("--max-tokens", type=int, default=4096)
        p.add_argument("--timeout", type=float, default=600)
        p.add_argument("--parts", type=int, default=16, help="generation files per run")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--only-dbs", nargs="*")
        p.add_argument("--day-cap", type=int, default=0, help="stop before this many tokens/day for this model (0=off)")
        p.add_argument("--week-cap", type=int, default=0, help="stop before this many tokens/ISO week (0=off)")
        p.add_argument("--eval-only", action="store_true", help="mark the run evaluation-only (export refuses it)")
        p.add_argument("--log-every", type=int, default=50)
        if job == "label":
            p.add_argument("--s1-cands", default=None, help="eval-only experiment: jsonl of System 1 candidates "
                           "(question_id -> candidates) appended to the prompt; off by default")
    p = sub.add_parser("verify")
    common(p)
    p.add_argument("--run", required=True)
    p.add_argument("--min-agree", type=float, default=0.5, help="vote: majority share must exceed this")
    p.add_argument("--grammar", default=os.environ.get("LUTHOR_GRAMMAR", "v3"), choices=verify.GRAMMARS)
    p.add_argument("--templates", type=Path, default=TEMPLATES, help="grammar-v3 template vocabulary ('' = any)")
    p.add_argument("--no-plans", action="store_true")
    p = sub.add_parser("export")
    common(p)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard-rows", type=int, default=50_000)
    p.add_argument("--allow-unchecked-contamination", action="store_true")
    p = sub.add_parser("usage")
    common(p)
    p = sub.add_parser("bakeoff-input")
    p.add_argument("--out", default=str(TEACHER / "bakeoff/input.jsonl"))
    p = sub.add_parser("bakeoff-report")
    common(p)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=str(TEACHER / "bakeoff/report.json"))
    a = ap.parse_args(argv)
    if a.cmd in ("label", "write"):
        return generate(a, a.cmd)
    if a.cmd == "verify" and a.templates is not None and not str(a.templates):
        a.templates = None
    return {"verify": cmd_verify, "export": cmd_export, "usage": cmd_usage, "bakeoff-input": cmd_bakeoff_input,
            "bakeoff-report": cmd_bakeoff_report}[a.cmd](a)


if __name__ == "__main__":
    main()
