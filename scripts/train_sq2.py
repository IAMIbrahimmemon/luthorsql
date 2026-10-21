"""Train LuthorSQ2 (architecture v2: schema in context, slots, grammar v2). Runs on the GPU pod.

  python scripts/train_sq2.py --plans DIR... --db-roots DIR... --dev-dbs dev_dbs.json --out runs/sq2_pilot \
      [--gate DIR]   # after training, score a gate set once (pilot C: Spider dev, covered questions)

Checkpoint selection uses held-out TRAINING databases (--dev-dbs). Metric: execution accuracy
of compiled plans (5 s per query), on in-grammar questions.

Several GPUs of one node: torchrun --standalone --nproc_per_node=N scripts/train_sq2.py ... (data parallel, class Dist:
the same updates as one GPU, N x fewer databases per GPU per step). Queue time limits: --auto-resume [--job-hours H]
(scripts/hpc/README.md).
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import copy
import json
import math
import random
import re
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":  # --grammar: LUTHOR_GRAMMAR before the first luthor.sq import (workers inherit it)
    from luthor.grammar_cli import preset
    preset(run_flags=())

from luthor.sq.model import Backbone  # noqa: E402
from luthor.sq.plan import compile_plan  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq2.model import LuthorSQ2, TorchScores  # noqa: E402
from luthor.sq2.targets import decode, set_vocab  # noqa: E402
from luthor import grammar_cli  # noqa: E402
from luthor.sq import bird_desc  # noqa: E402
from luthor.sq2.train import BackwardOOM, db_groups, encode_db, group_seed, make_example, predict, train_db  # noqa: E402

_LOG = logging.getLogger("luthorsql")


def _rss_gb() -> float:
    """Host memory of this process (leak watch)."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def _load_chunk(args):
    lines, card_dir, strict, all_rows = args
    card_dirs = card_dir if isinstance(card_dir, list) else [card_dir]  # --extra-train: its own cards, then --plans'
    cards = {}
    out, skipped = [], 0
    rnd = random.Random(0)
    for line in lines:
        r = json.loads(line) if isinstance(line, str) else line
        if (r.get("outcome") != "plan" and not all_rows) or (strict and r.get("empty_result")) or (not strict and r.get("twin_of")):
            continue
        db = r["db_id"]
        if db not in cards:
            p = next((Path(c) / f"{db}.json" for c in card_dirs if (Path(c) / f"{db}.json").exists()), None)
            cards[db] = SchemaCard.load(p) if p is not None else None
        if cards[db] is None:
            continue
        e = make_example(r, cards[db], rnd, strict=strict)
        if e is None:
            skipped += 1
        else:
            out.append((e, r))
    return out, skipped


VOCAB_ARGS = None  # (templates, nums) for grammar v3; worker processes need it too
CAND_ORDER = "v1"  # literal-candidate order (--cand-order; eval / rerank set it from the run's args.json)


def _init_worker(vocab, cand_order: str | None = None):
    if vocab is not None:
        set_vocab(*vocab)
    import luthor.sq2.train as _tr
    _tr.CAND_ORDER = cand_order or CAND_ORDER


_ID = re.compile(r'"id": "([^"]+)"')


def _with_bases(lines: list[str]) -> list:
    """--cand-order v2: each twin row carries its base row's question and lookup hits (row["_base"]), so its
    db_values can be rebuilt as serving finds them (train.twin_db_values). Rows stay strings unless changed."""
    twins = {i: json.loads(l) for i, l in enumerate(lines) if '"twin_of"' in l}
    twins = {i: r for i, r in twins.items() if r.get("twin_of")}
    need = {r["twin_of"] for r in twins.values()}
    base = {}
    for i, l in enumerate(lines):  # only the base rows' few fields are kept (synsql_train is 166k rows)
        m = _ID.search(l)
        if i not in twins and m and m.group(1) in need:
            r = json.loads(l)
            if r["id"] in need:
                base[r["id"]] = {k: r.get(k) for k in ("question", "context", "db_values")}
    return [{**twins[i], "_base": base[twins[i]["twin_of"]]} if i in twins and twins[i]["twin_of"] in base else l
            for i, l in enumerate(lines)]


def raw_rows(path: Path, plans_ids: set[str], prefix: str = "bird_dev") -> list[dict]:
    """Questions of a BIRD-format file (dev.json: question / evidence / SQL / db_id) that have no row in the plan set
    (prepare_plans drops questions whose literals do not ground or whose plan does not verify), as uncovered rows:
    id f"{prefix}_{i}" (prepare_plans' ids), no plan, gold = the original SQL."""
    out = []
    for i, q in enumerate(json.loads(path.read_text())):
        rid = f"{prefix}_{q.get('question_id', i)}"
        if rid not in plans_ids:
            out.append({"id": rid, "db_id": q["db_id"], "question": q["question"].strip(),
                        "context": (q.get("evidence") or "").strip(), "gold_sql": q["SQL"], "outcome": "missing"})
    return out


def load_vocab(path: Path | None):
    """Grammar v3 vocabulary: templates.json + templates_nums.json (scripts/templates_v3.py)."""
    if path is None:
        return None
    return json.loads(path.read_text()), json.loads(path.with_name(path.stem + "_nums.json").read_text())


def load_sets(dirs: list[Path], workers: int = 12, limit: int = 0, strict: bool = True, all_rows: bool = False,
              questions: Path | None = None):
    """strict=False (gate sets): keep every in-grammar question, even ones the slots can't
    express or with an empty result, so the score is over all covered questions.
    all_rows (with strict=False, scoring only): every question, uncovered ones too (no plan: t=None, gold = the
    original SQL, scored by executing the predicted SQL against it); `questions` (a BIRD-format dev.json) adds
    the questions the plan set has no row for at all."""
    cards, exs, gold = {}, [], {}
    for d in dirs:
        for p in (d / "cards").glob("*.json"):
            cards[p.stem] = SchemaCard.load(p)
        lines = open(d / "plans.jsonl").readlines()
        if limit:
            lines = lines[:limit]
        if CAND_ORDER == "v2":
            lines = _with_bases(lines)
        if all_rows and questions is not None:
            ids = {json.loads(l)["id"] if isinstance(l, str) else l["id"] for l in lines}
            extra = [r for r in raw_rows(questions, ids) if (d / "cards" / f"{r['db_id']}.json").exists()]
            print(f"{d}: +{len(extra)} questions from {questions} with no plan row", flush=True)
            lines = lines + extra
        chunks = [lines[i:i + 500] for i in range(0, len(lines), 500)]
        skipped = 0
        with Pool(workers, initializer=_init_worker, initargs=(VOCAB_ARGS, CAND_ORDER)) as pool:
            for part, sk in pool.imap(_load_chunk, [(c, d / "cards", strict, all_rows) for c in chunks]):
                skipped += sk
                for e, r in part:
                    exs.append(e)
                    gold[e.id] = r
        print(f"{d}: {len(exs)} examples so far, {skipped} skipped (outside the v2 slots)", flush=True)
    return cards, exs, gold


_EXTRA = re.compile(r"(.+?)(?::(\d+(?:\.\d*)?|\.\d+))?")


def parse_extra(spec: str) -> tuple[Path, float]:
    """--extra-train PATH[:WEIGHT] -> (path, weight). Weight 1 by default; 2.5 = two copies plus a fixed half."""
    m = _EXTRA.fullmatch(spec)
    return Path(m.group(1)), float(m.group(2) or 1.0)


def load_drop_dbs(spec: str | None) -> set[str]:
    """--drop-dbs: a comma list of db_ids, or a JSON file holding a list or {"dbs": [...]} (other keys, e.g. a
    "comment", are ignored). None / "" -> empty set."""
    if not spec:
        return set()
    p = Path(spec)
    if p.suffix == ".json" or p.is_file():
        d = json.loads(p.read_text())
        dbs = d.get("dbs") if isinstance(d, dict) else d
        if not isinstance(dbs, list) or not all(isinstance(x, str) for x in dbs):
            raise SystemExit(f"--drop-dbs {spec}: expected a JSON list of db_ids or {{\"dbs\": [...]}}")
        return set(dbs)
    return {x.strip() for x in spec.split(",") if x.strip()}


def drop_rows(rows: list, drop: set[str], gold: dict) -> tuple[list, dict]:
    """Rows not on the --drop-dbs databases, and what was removed (total, by database, by source)."""
    gone = [e for e in rows if e.db_id in drop]
    by_db: dict[str, int] = {}
    by_src: dict[str, int] = {}
    for e in gone:
        by_db[e.db_id] = by_db.get(e.db_id, 0) + 1
        src = provenance(gold.get(e.id, {}))
        by_src[src] = by_src.get(src, 0) + 1
    return [e for e in rows if e.db_id not in drop], {"dbs": sorted(drop), "removed": len(gone),
                                                      "by_db": dict(sorted(by_db.items())),
                                                      "by_source": dict(sorted(by_src.items()))}


def provenance(r: dict) -> str:
    """A training row's origin for the mix counts: its source, plus its teacher when a generator wrote it."""
    return str(r.get("source") or "?") + (f"/{r['teacher']}" if r.get("teacher") else "")


def load_extra(spec: str, plan_dirs: list[Path], cards: dict, gold: dict, exclude_dbs: set[str], k: int = 0,
               workers: int = 12, limit: int = 0) -> tuple[list, dict]:
    """--extra-train: more training rows, e.g. teacher-generated plans. PATH is a plan dir (plans.jsonl + cards/) or a
    .jsonl file of rows in the plan-set schema (id, db_id, question, context, outcome "plan", plan; optional db_values,
    source, teacher, ... -- extra fields are kept in `gold` and otherwise ignored). Cards come from PATH's own cards/
    (beside a .jsonl file) first, then from the --plans dirs. Rows on `exclude_dbs` (held-out dev / gate databases)
    are dropped, ids that clash with loaded rows get a suffix, and WEIGHT repeats the rows (fraction: a fixed sample).
    Adds the rows' cards to `cards` and rows to `gold`; returns (training examples, counts for train_mix.json)."""
    path, w = parse_extra(spec)
    plans = path / "plans.jsonl" if path.is_dir() else path
    own = (path if path.is_dir() else path.parent) / "cards"
    card_dirs = ([own] if own.is_dir() else []) + [d / "cards" for d in plan_dirs]
    lines = open(plans).readlines()
    if limit:
        lines = lines[:limit]
    read: dict[str, int] = {}
    for line in lines:
        r = json.loads(line)
        read[provenance(r)] = read.get(provenance(r), 0) + 1
    if CAND_ORDER == "v2":
        lines = _with_bases(lines)
    chunks = [lines[i:i + 500] for i in range(0, len(lines), 500)]
    got, skipped = [], 0
    with Pool(workers, initializer=_init_worker, initargs=(VOCAB_ARGS, CAND_ORDER)) as pool:
        for part, sk in pool.imap(_load_chunk, [(c, card_dirs, True, False) for c in chunks]):
            skipped += sk
            got += part
    kept, held_out, renamed = [], 0, 0
    for e, r in got:
        if e.db_id in exclude_dbs:
            held_out += 1
            continue
        if e.id in gold:  # never let an extra row overwrite another row's gold (or an earlier extra row's)
            base, j = e.id, 1
            while e.id in gold:
                e.id = f"{base}~x{k}" + (f".{j}" if j > 1 else "")
                j += 1
            r["id"] = e.id
            renamed += 1
        gold[e.id] = r
        if e.db_id not in cards:
            cards[e.db_id] = SchemaCard.load(next(c / f"{e.db_id}.json" for c in card_dirs if (c / f"{e.db_id}.json").exists()))
        kept.append(e)
    whole, frac = int(w), w - int(w)
    out = kept * whole + random.Random(0).sample(kept, round(frac * len(kept)))
    by_src: dict[str, int] = {}
    for e in kept:
        by_src[provenance(gold[e.id])] = by_src.get(provenance(gold[e.id]), 0) + 1
    counts = {"path": str(path), "weight": w, "rows": len(lines), "read_by_source": read, "kept": len(kept),
              "kept_by_source": by_src, "outside_slots": skipped, "held_out_dbs": held_out, "renamed_ids": renamed,
              "train_rows": len(out)}
    print(f"extra {path} x{w:g}: {len(kept)} of {len(lines)} rows kept ({skipped} outside the slots, {held_out} on "
          f"held-out dbs, {renamed} ids renamed), +{len(out)} training rows; by source {by_src}", flush=True)
    return out, counts


class TimedDB:
    """Read-only SQLite with a per-query deadline (a bad plan must not hang evaluation)."""

    def __init__(self, path: Path, seconds: float = 5.0):
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.con.text_factory = lambda b: b.decode("utf-8", "replace")
        self.deadline, self.seconds = 0.0, seconds
        self.con.set_progress_handler(lambda: 1 if time.time() > self.deadline else 0, 10_000)

    def rows(self, sql: str):
        self.deadline = time.time() + self.seconds
        return set(map(tuple, self.con.execute(sql).fetchmany(100_000)))


@torch.no_grad()
def evaluate(model, exs, cards, gold, db_paths, limit: int = 1500, dump: Path | None = None) -> dict:
    model.eval()
    exs = exs[:limit]
    by_db: dict[str, list] = {}
    for e in exs:
        by_db.setdefault(e.db_id, []).append(e)
    right = total = 0
    rows_out = []
    for db, group in by_db.items():
        con = TimedDB(db_paths[db]) if db in db_paths else None
        plans = predict(model, cards[db], group)
        g = cards[db].graph()
        for e, p in zip(group, plans):
            total += 1
            ok, sql = False, None
            if p is not None and con is not None:
                try:
                    sql = compile_plan(p, g)
                    ok = con.rows(sql) == con.rows(gold[e.id]["compiled_sql"])
                except Exception:
                    _LOG.debug("%s: eval candidate failed to compile or run", e.id, exc_info=True)
            right += ok
            rows_out.append({"id": e.id, "db_id": db, "question": e.question, "pred_sql": sql,
                             "gold_sql": gold[e.id]["compiled_sql"], "correct": ok})
    if dump:
        dump.write_text("".join(json.dumps(r) + "\n" for r in rows_out))
    model.train()
    return {"ex": right / max(total, 1), "n": total}


@torch.no_grad()
def latency(model, cards, exs, n: int = 50) -> dict:
    """Per-question time with the database's schema already cached (what serving pays)."""
    model.eval()
    ts = []
    from luthor.sq2.serialize import schema_text
    for e in exs[:n]:
        card = cards[e.db_id]
        st = schema_text(card)
        oix = {o: i for i, o in enumerate(st.option_ids)}
        se = model.encode_schema(model.prepare_schema(st))
        se2 = copy.copy(se)
        se2.cache = copy.deepcopy(se.cache)
        torch.cuda.synchronize()
        t = time.time()
        prep = model.prepare_question(e.question, e.cands, e.context)
        x = model.project(model.encode_questions(se2, [prep], [e.cands], oix)[0])
        decode(TorchScores(model, x, st.option_ids, e.cands))
        torch.cuda.synchronize()
        ts.append(time.time() - t)
    ts.sort()
    model.train()
    return {"p50_ms": 1000 * ts[len(ts) // 2], "p90_ms": 1000 * ts[int(len(ts) * 0.9)]}



class OOMLedger:
    """Questions lost to out-of-memory errors. Nothing is dropped silently: every loss is printed, the totals go
    in the step log and oom.json, and the run stops once more than `max_frac` of the questions seen were lost
    (round R2 lost 3,926 questions, mostly works_cycles, to OOM skips that only showed in the raw log)."""

    def __init__(self, max_frac: float, min_seen: int = 2000, defer: bool = False):
        self.max_frac, self.min_seen = max_frac, min_seen
        self.seen = self.skipped = self.retries = self.dropped_steps = 0
        self.by_db: dict[str, int] = {}
        # defer (data parallel): skip() never stops the run itself; train_step asks every process for over_limit()
        # at the step boundary and they all stop together (one process exiting alone would hang the others)
        self.defer = defer

    def skip(self, db: str, n: int) -> None:
        self.skipped += n
        self.by_db[db] = self.by_db.get(db, 0) + n
        if not self.defer and self.over_limit():
            raise SystemExit(self.limit_message())

    def over_limit(self) -> bool:
        return self.seen >= self.min_seen and self.skipped > self.max_frac * self.seen

    def limit_message(self) -> str:
        return (f"OOM: {self.skipped} of {self.seen} questions seen were skipped "
                f"({100 * self.skipped / max(self.seen, 1):.2f}% > --max-oom-skip {100 * self.max_frac:.2f}%); "
                f"by database: {self.top()}. Use --grad-ckpt or a smaller --budget-chars / --q-per-db.")

    def top(self, k: int = 5) -> str:
        return ", ".join(f"{d} {n}" for d, n in sorted(self.by_db.items(), key=lambda x: -x[1])[:k])

    def summary(self) -> dict:
        return {"questions_seen": self.seen, "questions_skipped": self.skipped, "retries": self.retries,
                "dropped_steps": self.dropped_steps, "by_db": self.by_db}

    def load(self, d: dict) -> None:
        self.seen, self.skipped, self.retries = d["questions_seen"], d["questions_skipped"], d["retries"]
        self.dropped_steps, self.by_db = d["dropped_steps"], dict(d["by_db"])

    @staticmethod
    def merged(parts: list[dict]) -> dict:
        """The run's totals from every process's summary (steps are dropped by all processes together: counted once)."""
        by_db: dict[str, int] = {}
        for p in parts:
            for d, n in p["by_db"].items():
                by_db[d] = by_db.get(d, 0) + n
        return {"questions_seen": sum(p["questions_seen"] for p in parts),
                "questions_skipped": sum(p["questions_skipped"] for p in parts),
                "retries": sum(p["retries"] for p in parts), "dropped_steps": parts[0]["dropped_steps"], "by_db": by_db}


class Dist:
    """Data parallelism under torchrun (one process per GPU; WORLD_SIZE / RANK / LOCAL_RANK from the launcher).
    Every process holds the whole model and the same training list; each step's chunk of --dbs-per-step databases is
    split between them (chunk[rank::world]), each database still weighted 1/len(chunk), and the gradients are SUMMED,
    so a step's update is the single-process update of the same chunk (tests/test_ddp.py). Not torch DDP: a step is
    many backward calls (one per database, OOM halves), parameters go unused, and an OOM-dropped step must be
    agreed -- a plain all-reduce of the finished gradients after the step's local work covers all three.
    world == 1 (no torchrun): every method is a no-op and nothing touches torch.distributed."""

    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        self.main = self.rank == 0
        self.device = torch.device("cpu")

    def setup(self, timeout_min: float = 180) -> "Dist":
        if self.world > 1:
            import datetime
            import torch.distributed as dist
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local)
                self.device = torch.device("cuda", self.local)
            # the other processes wait in their next collective while rank 0 evaluates / saves: a long timeout
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                    timeout=datetime.timedelta(minutes=timeout_min))
        return self

    def close(self) -> None:
        if self.world > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()

    def shard(self, chunk: list) -> list:
        return chunk[self.rank::self.world]

    def reduce(self, xs: list[float], op: str = "sum") -> list[float]:
        if self.world == 1:
            return list(xs)
        import torch.distributed as dist
        t = torch.tensor(xs, dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op={"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}[op])
        return t.tolist()

    def gather(self, obj) -> list:
        if self.world == 1:
            return [obj]
        import torch.distributed as dist
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def broadcast_params(self, params: list) -> None:
        """Rank 0's values everywhere (start of training, and after every eval as a drift guard)."""
        if self.world == 1:
            return
        import torch.distributed as dist
        with torch.no_grad():
            for p in params:
                t = p.data.to(self.device)
                dist.broadcast(t, 0)
                p.data.copy_(t)

    def reduce_grads(self, params: list) -> None:
        """Sum every process's gradients (fp32, one flat buffer) and keep a parameter's grad None when no process
        produced one (as a single process would: AdamW then leaves it untouched, weight decay included)."""
        if self.world == 1:
            return
        import torch.distributed as dist
        flat = torch.cat([(p.grad.float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32))
                          .reshape(-1).to(self.device) for p in params]
                         + [torch.tensor([float(p.grad is not None) for p in params], device=self.device)])
        dist.all_reduce(flat)
        used = flat[-len(params):].tolist()
        o = 0
        for p, u in zip(params, used):
            n = p.numel()
            if u:
                g = flat[o:o + n].view_as(p).to(device=p.device, dtype=p.dtype)
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.copy_(g)
            else:
                p.grad = None
            o += n


OOM_DROP_DESC = [False, 0]  # --oom-drop-desc on/off, questions retried without descriptions (logged at the end)


def run_db(model, card, db: str, es: list, budget_chars: int, rnd, scale: float, ledger: OOMLedger,
           drop_desc: bool = False):
    """train_db on one database's questions. An OOM in the forward pass leaves the gradients untouched, so the
    questions are retried in halves (each half weighted by its share), down to single questions; a question that
    still does not fit is skipped and counted. An OOM in backward (BackwardOOM) is raised to the caller.
    Returns (mean loss, loss parts, questions trained)."""
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, parts = (train_db(model, card, es, budget_chars, rnd, scale=scale, drop_desc=True) if drop_desc
                        else train_db(model, card, es, budget_chars, rnd, scale=scale))
        return l, parts, len(es)
    except BackwardOOM:
        raise
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(es) == 1 and OOM_DROP_DESC[0] and not drop_desc:
            # --oom-drop-desc: one question that does not fit is retried once with the column / table descriptions
            # left out of its schema text (shorter prompt) instead of being skipped
            print(f"OOM on {db} (1 question, forward): retried without descriptions", flush=True)
            OOM_DROP_DESC[1] += 1
            return run_db(model, card, db, es, budget_chars, rnd, scale, ledger, drop_desc=True)
        if len(es) == 1:
            print(f"OOM on {db} (1 question, forward): skipped", flush=True)
            ledger.skip(db, 1)
            return 0.0, {}, 0
        ledger.retries += 1
        h = len(es) // 2
        res = [run_db(model, card, db, part, budget_chars, rnd, scale * len(part) / len(es), ledger)
               for part in (es[:h], es[h:])]
        n = sum(r[2] for r in res)
        if not n:
            return 0.0, {}, 0
        parts: dict = {}
        for _, p, k in res:
            for key, v in p.items():
                parts[key] = parts.get(key, 0.0) + v * k / n
        return sum(r[0] * r[2] for r in res) / n, parts, n


def step_grads(model, chunk: list, mine: list, cards: dict, budget_chars: int, rnd, ledger: OOMLedger):
    """Accumulate one step's gradients over this process's databases `mine` (all of `chunk` in one process), each
    weighted 1/len(chunk). Returns (dropped, loss, loss parts): dropped after an OOM in backward, with the gradients
    cleared and this process's questions counted as skipped."""
    parts_acc: dict = {}
    loss_sum = 0.0
    for db, es in mine:
        try:
            l, parts, n = run_db(model, cards[db], db, es, budget_chars, rnd, 1.0 / len(chunk), ledger)
        except BackwardOOM:
            # partial gradients from a half-finished backward: the whole step is unusable
            nq = sum(len(x) for _, x in chunk)
            print(f"OOM in backward on {db}: step dropped ({nq} questions: "
                  f"{', '.join(d for d, _ in chunk)})", flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            ledger.dropped_steps += 1
            for d, x in mine:
                ledger.skip(d, len(x))
            return True, 0.0, {}
        if n:
            loss_sum += l * n / len(es) / len(chunk)
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v / len(chunk)
    return False, loss_sum, parts_acc


def train_step(model, opt, sched, chunk: list, cards: dict, budget_chars: int, rnd, ledger: OOMLedger,
               dist: Dist, params: list, step: int = 0) -> tuple[str, float, dict]:
    """One optimizer step over `chunk` ([(db, questions)]). Returns (status, loss, loss parts); status "ok" (the
    update was applied), "dropped" (OOM in backward on any process) or "nonfinite" (gradient norm). With several
    processes every branch is decided on all-reduced values, so all of them take it together."""
    mine = dist.shard(chunk)
    ledger.seen += sum(len(es) for _, es in mine)
    dropped, loss_sum, parts_acc = step_grads(model, chunk, mine, cards, budget_chars, rnd, ledger)
    if dist.world > 1:
        from luthor.sq2.train import W
        keys = list(W)
        v = dist.reduce([float(dropped), float(ledger.over_limit()), loss_sum] + [parts_acc.get(k, 0.0) for k in keys]
                        + [float(k in parts_acc) for k in keys])
        if v[1]:  # some process lost too many questions to OOM: every process stops
            raise SystemExit(ledger.limit_message() if ledger.over_limit() else
                             "OOM: another process passed --max-oom-skip (see its log); stopping all processes")
        if v[0]:
            if not dropped:  # another process ran out of memory in backward: this process's share is lost too
                model.zero_grad(set_to_none=True)
                ledger.dropped_steps += 1
                for d, x in mine:
                    ledger.skip(d, len(x))
            return "dropped", 0.0, {}
        loss_sum = v[2]
        parts_acc = {k: x for k, x, has in zip(keys, v[3:3 + len(keys)], v[3 + len(keys):]) if has}
        dist.reduce_grads(params)
    elif dropped:
        return "dropped", 0.0, {}
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gn):
        print(f"non-finite gradient at step {step}: update skipped ({[db for db, _ in chunk]})", flush=True)
        opt.zero_grad(set_to_none=True)
        return "nonfinite", loss_sum, parts_acc
    opt.step()
    opt.zero_grad(set_to_none=True)
    sched.step()
    return "ok", loss_sum, parts_acc


# ---- resumable state (--auto-resume): out/resume/{lora/, heads.pt, state.pt}, replaced as a whole -----------------

RESUME_KEYS = ("plans", "extra_train", "repeat", "seed", "q_per_db", "dbs_per_step", "max_steps", "epochs", "base",
               "subfolder", "lora_r", "vocab", "cand_order", "value_side", "col_mentions", "joint_layers", "grammar",
               "drop_dbs", "bird_desc")
RESUME_DEFAULTS = {"grammar": "v3", "drop_dbs": "None", "bird_desc": ""}  # keys added later: what a state saved before them means


def save_resume(out: Path, model, opt, sched, dist: Dist, rnd, a, **state) -> None:
    """Everything needed to continue the run where it is: weights, optimizer, LR schedule, data position, best, OOM
    ledger, each process's random state. A collective (all processes call it at the same step); rank 0 writes."""
    rng = dist.gather({"rnd": rnd.getstate(), "torch": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None})
    ledgers = dist.gather(state.pop("ledger"))
    if not dist.main:
        return
    tmp, cur, old = out / "resume.tmp", out / "resume", out / "resume.old"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    model.backbone.model.save_pretrained(tmp / "lora")
    torch.save(model.head_state(), tmp / "heads.pt")
    torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng, "ledgers": ledgers,
                "world": dist.world, "args": {k: str(getattr(a, k, None)) for k in RESUME_KEYS}}, tmp / "state.pt")
    shutil.rmtree(old, ignore_errors=True)
    if cur.exists():
        cur.rename(old)
    tmp.rename(cur)
    shutil.rmtree(old, ignore_errors=True)


def load_resume(out: Path, model, opt, sched, dist: Dist, rnd, a) -> dict | None:
    """The state save_resume wrote (out/resume, or out/resume.old if a save was cut between its renames), loaded into
    the model / optimizer / schedule / random state; None when the run has no saved state."""
    d = next((x for x in (out / "resume", out / "resume.old") if (x / "state.pt").exists()), None)
    if d is None:
        return None
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    st = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
    now = {k: str(getattr(a, k, None)) for k in RESUME_KEYS}
    diff = {k: (st["args"].get(k, RESUME_DEFAULTS.get(k)), v) for k, v in now.items()
            if st["args"].get(k, RESUME_DEFAULTS.get(k)) != v}
    if diff:
        raise SystemExit(f"--auto-resume: {out} was trained with other settings (saved, now): {diff}")
    set_peft_model_state_dict(model.backbone.model, load_file(str(d / "lora" / "adapter_model.safetensors")))
    model.load_state_dict(torch.load(d / "heads.pt", map_location="cpu"), strict=False)
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    if st["world"] == dist.world:  # same process count: each process continues its own random streams
        r = st["rng"][dist.rank]
        rnd.setstate(r["rnd"])
        torch.set_rng_state(r["torch"])
        if r["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(r["cuda"])
        st["ledger"] = st["ledgers"][dist.rank]
    else:
        print(f"--auto-resume: saved with {st['world']} processes, now {dist.world}: random streams restart", flush=True)
        st["ledger"] = OOMLedger.merged(st["ledgers"]) if dist.main else OOMLedger(0).summary()
    print(f"resumed {d}: step {st['step']}/{st['total']}, epoch {st['epoch']} position {st['pos']}, "
          f"best {st['best']:.3f}", flush=True)
    return st


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", type=Path, required=True)
    ap.add_argument("--db-roots", nargs="*", default=[])
    ap.add_argument("--base", default="AlexWortega/openjev")
    ap.add_argument("--subfolder", default="qwen3.5-4b-nli-v5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--dbs-per-step", type=int, default=4)
    ap.add_argument("--q-per-db", type=int, default=8)
    ap.add_argument("--budget-chars", type=int, default=9000, help="schema text budget in training (larger schemas are sampled)")
    ap.add_argument("--dev-dbs", type=Path, required=True)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="rows per plan set (smoke)")
    ap.add_argument("--gate", type=Path, help="plan set scored once after training (covered questions)")
    ap.add_argument("--resume", type=Path, help="run dir: load its best_lora + best_heads.pt")
    ap.add_argument("--dev-plans", nargs="*", type=Path, default=[],
                    help="plan sets used only for their --dev-dbs rows (when the training mix excludes the dev databases)")
    ap.add_argument("--schema-aug", type=float, default=0.0, help="share of database-steps with renamed identifiers (round 3: 0.35)")
    ap.add_argument("--hours", type=float, default=0, help="stop training after this many hours (then the gate still runs)")
    ap.add_argument("--vocab", type=Path, help="grammar v3: templates.json (with templates_nums.json beside it)")
    ap.add_argument("--gate-total", type=int, default=0, help="questions in the whole gate set (all-question EX)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing (keeps the schema prefix cache): ~8-24x fewer saved activations, "
                         "~30%% slower; needed for 14B/32B and for the largest training schemas")
    ap.add_argument("--oom-drop-desc", action="store_true",
                    help="a single question that runs out of memory in the forward pass is retried once without "
                         "column / table descriptions in its schema text before it is skipped (default off: skipped)")
    ap.add_argument("--max-oom-skip", type=float, default=0.002,
                    help="stop the run if more than this share of the questions seen is lost to out-of-memory errors")
    ap.add_argument("--repeat", nargs="*", default=[], metavar="SET=N",
                    help="repeat a plan set's training rows N times per epoch, e.g. bird_train=3 (SET = the --plans dir name)")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="stop at this step without changing the LR schedule (paired A/B against a time-limited run)")
    ap.add_argument("--fit-hours", action="store_true",
                    help="with --hours: size the step count (and so the LR decay) to the time budget, from the measured step time")
    ap.add_argument("--col-mentions", action="store_true", help="add question states at written column names to option keys")
    ap.add_argument("--mention-weight", choices=["flat", "inv"], default="flat",
                    help="inv: a written name that matches N options counts 1/N for each")
    ap.add_argument("--desc-drop", type=float, default=0.0, help="share of database-steps shown with no descriptions")
    ap.add_argument("--mention-drop", type=float, default=0.0, help="training: hide this share of written column names")
    ap.add_argument("--bird-filter", type=Path, help="bird23-train-filtered jsonl: train only on BIRD's clean subset")
    ap.add_argument("--joint-layers", type=int, default=0, help="joint head layers over slots/options/candidates (0 = off)")
    ap.add_argument("--cand-order", choices=["v1", "v2"], default="v1",
                    help="literal candidates: v1 = round-6 (gold literals force-kept past the cap, enumerated values "
                         "shuffled); v2 = the serve-time list exactly (deterministic relevance order cut at the cap; "
                         "a value target whose literal is cut is dropped; twins' lookup hits rebuilt)")
    ap.add_argument("--value-side", action="store_true",
                    help="write the selected candidates after the question ('Values: ...') and add their states there "
                         "to the candidate vectors (question-conditioned values)")
    ap.add_argument("--value-entries", type=int, default=24,
                    help="--value-side: write the N most relevant candidates (the cut-order ranking); the rest keep "
                         "their mention / listed vectors only")
    ap.add_argument("--seed", type=int, default=0,
                    help="training randomness: model init, schema sampling / augmentation, database grouping and "
                         "order (0 = every earlier run's order exactly); the dev subset and the data mix stay fixed")
    ap.add_argument("--extra-train", action="append", default=[], metavar="PATH[:WEIGHT]",
                    help="more training rows (repeatable): a plan dir (plans.jsonl + cards/) or a .jsonl of plan rows "
                         "(e.g. teacher-generated; source / teacher fields are kept for the mix counts). Cards from "
                         "its own cards/ then the --plans dirs; rows on --dev-dbs / --gate databases are dropped; "
                         "WEIGHT (default 1) repeats them, a fraction keeps a fixed sample")
    ap.add_argument("--auto-resume", action="store_true",
                    help="keep a resumable state in OUT/resume (at every eval and every --save-every steps) and, when "
                         "OUT already has one, continue from it (same command again after a queue time limit)")
    ap.add_argument("--save-every", type=int, default=0, help="--auto-resume: also save the state every N steps")
    ap.add_argument("--job-hours", type=float, default=0,
                    help="end this JOB (not the run) after H hours: save the resumable state, write OUT/job_stopped, "
                         "exit 0; resubmit the same command (needs --auto-resume). --hours still ends the run")
    grammar_cli.add_argument(ap)
    ap.add_argument("--drop-dbs", default=None, metavar="DB,DB|FILE.json",
                    help="remove every row on these databases from ALL training sources (--plans, --repeat copies, "
                         "--extra-train), logged with counts; a comma list or a JSON file (list, or {\"dbs\": [...]}), "
                         "e.g. data/sys1/contaminated_dbs.json. Default: none (R1/R2 reproduce unchanged)")
    ap.add_argument("--bird-desc", nargs="?", const="1", default="",
                    help="BIRD-train column descriptions in the schema cards (luthor.sq.bird_desc): bare = "
                         "data/raw/bird_train_desc, or a directory. Default off (cards unchanged)")
    ap.add_argument("--dist-timeout-min", type=float, default=180,
                    help="torchrun: collective timeout (the other processes wait while rank 0 evaluates)")
    return ap


def main():
    a = parser().parse_args()
    a.grammar = grammar_cli.resolve(a.grammar)  # recorded in args.json: the grammar the heads are built for
    bird_desc.enable(a.bird_desc)  # recorded in args.json; eval_sq2 / rerank_sq2 gen follow the run
    if a.resume:
        grammar_cli.check_run(a.resume)
    print(f"grammar {a.grammar}", flush=True)
    if a.seed < 0:
        raise SystemExit("--seed must be >= 0")
    if (a.save_every or a.job_hours) and not a.auto_resume:
        raise SystemExit("--save-every / --job-hours need --auto-resume")
    dist = Dist().setup(a.dist_timeout_min)
    global VOCAB_ARGS, CAND_ORDER
    VOCAB_ARGS = load_vocab(a.vocab)
    CAND_ORDER = a.cand_order
    _init_worker(VOCAB_ARGS)
    torch.manual_seed(a.seed)  # also seeds every CUDA device
    random.seed(a.seed)
    try:
        import numpy as np
        np.random.seed(a.seed)
    except ImportError:
        pass
    # cuDNN's fused attention backward failed (mha_graph.execute) and gave non-finite gradients on long
    # schema prefixes (2026-10-05 pilot); the flash / memory-efficient SDPA kernels do not
    torch.backends.cuda.enable_cudnn_sdp(False)
    rnd = random.Random(0)
    import luthor.sq2.train as _tr
    _tr.AUG_P = a.schema_aug
    _tr.DESC_DROP = a.desc_drop
    a.out.mkdir(parents=True, exist_ok=True)
    if dist.main:
        (a.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
        if VOCAB_ARGS is not None:  # head sizes depend on it: keep it with the checkpoint
            (a.out / "vocab.json").write_text(json.dumps({"templates": VOCAB_ARGS[0], "nums": VOCAB_ARGS[1]}))
        (a.out / "job_stopped").unlink(missing_ok=True)
    import contextlib
    quiet = contextlib.redirect_stdout(open(os.devnull, "w")) if not dist.main else contextlib.nullcontext()
    with quiet:  # data loading logs once (every process loads the same data)
        cards, exs, gold = load_sets(a.plans, limit=a.limit)
    if a.bird_filter:  # BIRD's official clean subset (bird23-train-filtered): drop other BIRD train rows and their twins
        norm = lambda q: " ".join(q.lower().split())
        clean = {(r["db_id"], norm(r["question"])) for r in map(json.loads, open(a.bird_filter))}

        def kept(e) -> bool:
            r = gold[e.id]
            base = gold.get(r.get("twin_of"), r)
            return not (base.get("source") == "bird" and base.get("split") == "train") or \
                (base["db_id"], norm(base["question"])) in clean
        n0 = len(exs)
        exs = [e for e in exs if kept(e)]
        print(f"bird filter: dropped {n0 - len(exs)} BIRD train rows outside the official clean subset", flush=True)
    dev_dbs = set(json.loads(a.dev_dbs.read_text()))
    train = [e for e in exs if e.db_id not in dev_dbs]
    dev = [e for e in exs if e.db_id in dev_dbs]
    for spec in a.repeat:
        name, n = spec.split("=")
        d = next(d for d in a.plans if d.name == name)
        dbs = {p.stem for p in (d / "cards").glob("*.json")}
        # by database, but never the generated rare-skeleton rows (source bird_mut) that share BIRD-train databases
        extra = [e for e in train if e.db_id in dbs and gold.get(e.id, {}).get("source") != "bird_mut"]
        train += extra * (int(n) - 1)
        print(f"repeat {name} x{n}: +{len(extra) * (int(n) - 1)} rows", flush=True)
    mix = []
    if a.extra_train:  # after --repeat / --bird-filter / the dev split: extra rows are only ever training rows
        held = dev_dbs | ({p.stem for p in (a.gate / "cards").glob("*.json")} if a.gate else set())
        with quiet:
            for k, spec in enumerate(a.extra_train):
                rows, counts = load_extra(spec, a.plans, cards, gold, held, k, limit=a.limit)
                train += rows
                mix.append(counts)
    drop = load_drop_dbs(a.drop_dbs)
    dropped = None
    if drop:  # after --repeat and --extra-train: every training source, copies included
        train, dropped = drop_rows(train, drop, gold)
        print(f"drop dbs {dropped['dbs']}: removed {dropped['removed']} training rows, by database "
              f"{dropped['by_db']}, by source {dropped['by_source']}", flush=True)
        if drop & dev_dbs:
            print(f"WARNING --drop-dbs: {sorted(drop & dev_dbs)} are --dev-dbs databases (dev rows are kept)", flush=True)
    by_src: dict[str, int] = {}
    for e in train:
        by_src[provenance(gold.get(e.id, {}))] = by_src.get(provenance(gold.get(e.id, {})), 0) + 1
    print(f"training rows by source: {dict(sorted(by_src.items(), key=lambda x: -x[1]))}", flush=True)
    if (mix or dropped) and dist.main:
        (a.out / "train_mix.json").write_text(json.dumps({"train_by_source": by_src, "extra": mix,
                                                          **({"dropped_dbs": dropped} if dropped else {})}, indent=1))
    if a.dev_plans:
        with quiet:
            dcards, dexs, dgold = load_sets(a.dev_plans)
        for e in dexs:
            if e.db_id in dev_dbs:
                dev.append(e)
                gold[e.id] = dgold[e.id]
                cards.setdefault(e.db_id, dcards[e.db_id])
    rnd.shuffle(dev)  # with Random(0) whatever --seed is: every seed is selected on the same dev subset
    if a.seed:
        rnd = random.Random(a.seed)
    if dist.world > 1:  # each process its own stream (rank 0 of a multi-process run is not the single-process run)
        rnd = random.Random(f"{a.seed}:{dist.rank}")
    db_paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            db_paths.setdefault(p.stem, p)
    print(f"train {len(train)} / {len({e.db_id for e in train})} dbs, dev {len(dev)} / {len({e.db_id for e in dev})} dbs, "
          f"{len(db_paths)} sqlite files" + (f", {dist.world} processes" if dist.world > 1 else ""), flush=True)

    backbone = Backbone.from_pretrained(a.base, a.subfolder or None, lora_r=a.lora_r, grad_ckpt=a.grad_ckpt)
    model = LuthorSQ2(backbone, joint_layers=a.joint_layers, col_mentions=a.col_mentions,
                      mention_weight=a.mention_weight, mention_drop=a.mention_drop, value_side=a.value_side,
                      value_entries=a.value_entries).cuda()
    model.backbone.model.to(torch.bfloat16)
    if a.resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model.backbone.model, load_file(str(a.resume / "best_lora" / "adapter_model.safetensors")))
        model.load_state_dict(torch.load(a.resume / "best_heads.pt"), strict=False)
        print(f"resumed from {a.resume}", flush=True)
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    lora = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    if dist.world > 1:
        dist.broadcast_params(lora + heads)  # same start everywhere (the seed already gives it; this makes sure)
        torch.manual_seed((a.seed << 16) + dist.rank + 1)  # LoRA dropout: a different mask on each process
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": heads, "lr": a.head_lr}], weight_decay=0.01)
    q_per_step = a.dbs_per_step * a.q_per_db
    # --max-steps N: exactly N optimizer steps with the cosine spanning exactly N (--fit-hours is then ignored;
    # --hours only stops a run that overruns, saving first). Otherwise epochs, or --fit-hours sizes it at step 50.
    total = a.max_steps or int(math.ceil(len(train) / q_per_step) * a.epochs)
    span = [total]  # the cosine's length (fit-hours may shorten it; --max-steps never changes it)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(span[0], 1)))))
    model.train()
    ledger = OOMLedger(a.max_oom_skip, defer=dist.world > 1)
    OOM_DROP_DESC[0] = bool(a.oom_drop_desc)
    step, best, t0, epoch, pos = 0, -1.0, time.time(), 0, 0
    st = load_resume(a.out, model, opt, sched, dist, rnd, a) if a.auto_resume else None
    if st is not None:
        step, best, epoch, pos, total = st["step"], st["best"], st["epoch"], st["pos"], st["total"]
        span[0] = st["span"]
        t0 = time.time() - st["elapsed"]
        ledger.load(st["ledger"])
    print(f"schedule: {total} steps" + (" (--max-steps)" if a.max_steps else ""), flush=True)
    t_job, s_job = time.time(), step  # this job's start (s/step after a resume)
    params = lora + heads

    def save_state(next_pos: int):
        if a.auto_resume:
            save_resume(a.out, model, opt, sched, dist, rnd, a, step=step, best=best, epoch=epoch, pos=next_pos,
                        total=total, span=span[0], elapsed=time.time() - t0, ledger=ledger.summary())

    def eval_and_save():
        nonlocal best
        if dist.main:
            m = evaluate(model, dev, cards, gold, db_paths)
            print(f"eval step {step}: {m}", flush=True)
            model.backbone.model.save_pretrained(a.out / "last_lora")
            torch.save(model.head_state(), a.out / "last_heads.pt")
            if m["ex"] > best:
                best = m["ex"]
                model.backbone.model.save_pretrained(a.out / "best_lora")
                torch.save(model.head_state(), a.out / "best_heads.pt")
                (a.out / "best.json").write_text(json.dumps({"step": step, **m}))
        dist.broadcast_params(params)  # drift guard: every process continues from rank 0's weights

    stopped = False
    while step < total and not stopped:
        groups = db_groups(train, a.q_per_db, seed=group_seed(a.seed, epoch))
        start, pos = pos, 0
        for i in range(start, len(groups), a.dbs_per_step):
            chunk = groups[i:i + a.dbs_per_step]
            status, loss_sum, parts_acc = train_step(model, opt, sched, chunk, cards, a.budget_chars, rnd, ledger,
                                                     dist, params, step)
            if status != "ok":
                continue
            step += 1
            if step % 20 == 0:
                gc.collect()
            if os.environ.get("DEBUG_MEM"):
                print(f"mem step {step}: allocated {torch.cuda.memory_allocated() / 2**30:.2f}GB", flush=True)
            if step % 25 == 0 and dist.main:
                print(f"step {step}/{total} ep {epoch} loss {loss_sum:.3f} "
                      + " ".join(f"{k}={v:.3f}" for k, v in parts_acc.items())
                      + f" {(time.time() - t_job) / max(step - s_job, 1):.2f}s/step mem {torch.cuda.max_memory_allocated() / 2**30:.0f}GB"
                      + f" live {torch.cuda.memory_allocated() / 2**30:.1f}GB reserved {torch.cuda.memory_reserved() / 2**30:.0f}GB"
                      + f" rss {_rss_gb():.1f}GB"
                      + (f" oom_skipped {ledger.skipped}/{ledger.seen}q ({ledger.top(3)})" if ledger.skipped else ""),
                      flush=True)
            saved = False
            if step % a.eval_every == 0 or step == total:
                eval_and_save()
                save_state(i + a.dbs_per_step)
                saved = True
            if a.fit_hours and a.hours and step == 50 and not a.max_steps:  # LR schedule ends when the time budget does
                per = (time.time() - t0) / step + 120 / a.eval_every  # + about 2 min per held-out eval
                per = dist.reduce([per], "max")[0]  # every process the same step count
                total = min(total, int(a.hours * 3600 * 0.97 / per))
                span[0] = total
                print(f"fit-hours: {per:.2f}s/step -> {total} steps", flush=True)
            if a.stop_at and step == a.stop_at and step < total:
                a.hours = 1e-9  # end exactly as the time limit does: final eval, then the gate
            if a.hours and dist.reduce([float(time.time() - t0 > a.hours * 3600)], "max")[0] and step < total:
                print(f"time limit at step {step}" + (f" (safety stop: --max-steps {a.max_steps} not reached)"
                                                       if a.max_steps else ""), flush=True)
                eval_and_save()
                total = step
                save_state(i + a.dbs_per_step)
                saved = True
            if a.save_every and step % a.save_every == 0 and not saved:
                save_state(i + a.dbs_per_step)
                saved = True
            if step >= total:
                break
            if a.job_hours and dist.reduce([float(time.time() - t_job > a.job_hours * 3600)], "max")[0]:
                if not saved:
                    save_state(i + a.dbs_per_step)
                print(f"JOB_STOPPED at step {step}/{total}: state saved in {a.out / 'resume'}; resubmit to continue",
                      flush=True)
                if dist.main:
                    (a.out / "job_stopped").write_text(json.dumps({"step": step, "total": total}))
                stopped = True
                break
        epoch += 1
    merged = OOMLedger.merged(dist.gather(ledger.summary()))
    if stopped or not dist.main:
        dist.close()
        return
    dist.close()
    print(f"done in {(time.time() - t0) / 3600:.2f} h, best dev EX {best:.3f}", flush=True)
    if OOM_DROP_DESC[0]:
        merged = {**merged, "desc_fallback_this_process": OOM_DROP_DESC[1]}
    (a.out / "oom.json").write_text(json.dumps(merged, indent=1))
    print(f"OOM: {merged['questions_skipped']} of {merged['questions_seen']} questions skipped, {merged['retries']} split "
          f"retries, {merged['dropped_steps']} steps dropped"
          + (f" ({ledger.top()})" if merged["questions_skipped"] else ""), flush=True)
    if a.gate:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model.backbone.model, load_file(str(a.out / "best_lora" / "adapter_model.safetensors")))
        model.load_state_dict(torch.load(a.out / "best_heads.pt"), strict=False)
        gcards, gx, gg = load_sets([a.gate], strict=False)
        m = evaluate(model, gx, gcards, gg, db_paths, limit=100_000, dump=a.out / "gate_predictions.jsonl")
        lat = latency(model, gcards, gx)
        if a.gate_total:
            m["ex_all"] = m["ex"] * m["n"] / a.gate_total  # uncovered questions count as wrong
            m["total"] = a.gate_total
        print(f"GATE {a.gate.name}: {m} latency {lat}", flush=True)
        (a.out / "gate.json").write_text(json.dumps({"set": str(a.gate), **m, **lat}, indent=1))


if __name__ == "__main__":
    main()
