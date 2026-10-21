"""Round-3 reranker: candidates, training, held-out evaluation. Runs on the GPU pod.

  gen    top-K candidate plans from a trained slot model, compiled and executed, each marked right / wrong
         against the gold rows (serve-time literal candidates: card values + live database lookup).
           python scripts/rerank_sq2.py gen --run runs/sq3_pilot --plans DIR... --db-roots DIR... \
               --dev-dbs dev.json --split train --max-q 12000 --out runs/sq3_pilot/cands_train.jsonl
  train  the rerank adapter on train candidates; scores the dev candidates at the end.
           python scripts/rerank_sq2.py train --cards DIR... --train-cands X.jsonl --dev-cands Y.jsonl --out runs/rr3
  score  re-score a candidates file with a trained reranker (no training).

Only held-out TRAINING databases (--dev-dbs) are used for choices; sealed sets are never touched here.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import logging
import math
import random
import shutil
import sqlite3
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":  # --grammar: LUTHOR_GRAMMAR before the first luthor.sq import (workers inherit it)
    from luthor.grammar_cli import preset
    preset()

from luthor import grammar_cli  # noqa: E402
from luthor.sq import bird_desc  # noqa: E402
from luthor.sq.candidates import ValueIndex, find_candidates  # noqa: E402
from luthor.sq.model import Backbone  # noqa: E402
from luthor.sq.plan import compile_plan  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq2.model import LuthorSQ2, TorchScores  # noqa: E402
from luthor.sq2.rerank import (JOINT_MAX_RESULTS, Reranker, average_states, full_attention, gradcache_backward,  # noqa: E402
                               group_loss, joint_prefix_ids, joint_texts, listwise_loss, multipos_loss, prefix_kv,
                               prefix_kv_ids, result_classes, result_preview, row_ids, rows_over_prefix)
from luthor.sq2.serialize import SCHEMA_HEAD, schema_text  # noqa: E402
from luthor.sq2.topk import candidates  # noqa: E402
from luthor.sq2.train import encode_db, question_body, select_cands, serve_chunks  # noqa: E402

_LOG = logging.getLogger("luthorsql")


# ---- execution (worker processes) ------------------------------------------------------------

def _connect(path: str, seconds: float):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)

    def rows(sql):
        box["t"] = time.time() + seconds
        return con.execute(sql).fetchmany(100_000)
    return rows


def _exec_db(task):
    """(db path, [(gold_sql, [sql | None])]) -> per question: gold ok?, per candidate (runs, nonnull, ok, sig).
    sig fingerprints the result as a set of rows (EX compares sets), so candidates that return the same answer
    can be grouped at selection time (execution-output clustering)."""
    path, qs, seconds = task
    rows = _connect(path, seconds)
    out = []
    for gold_sql, sqls in qs:
        try:
            ref = set(map(tuple, rows(gold_sql)))
        except Exception:
            _LOG.debug("reference query failed to run", exc_info=True)
            out.append(None)
            continue
        res, seen = [], {}
        for s in sqls:
            if s is None:
                res.append((False, False, False, None, None))
            elif s in seen:
                res.append(seen[s])
            else:
                try:
                    got = rows(s)
                    gs = set(map(tuple, got))
                    sig = hashlib.md5(repr(sorted(map(repr, gs))).encode()).hexdigest()[:12]
                    r = (True, any(v is not None for x in got for v in x), gs == ref, sig, result_preview(got))
                except Exception:
                    _LOG.debug("candidate failed to run", exc_info=True)
                    r = (False, False, False, None, None)
                seen[s] = r
                res.append(r)
        out.append(res)
    return out


# ---- gen ----------------------------------------------------------------------------------------

def load_slot_model(run: Path, base: str, sub: str) -> LuthorSQ2:
    from eval_sq2 import load_model  # reads the run's args.json (LoRA rank, joint head)
    return load_model(run, base, sub, "cuda")


def use_run_settings(a):
    """Grammar-v3 runs carry their backbone (args.json) and template vocabulary (vocab.json)."""
    import train_sq2
    from eval_sq2 import run_args, use_cand_order
    a.grammar = grammar_cli.check_run(a.run, getattr(a, "grammar", None))  # args.json grammar (missing = v3)
    ra = run_args(a.run)
    if not getattr(a, "bird_desc", "") and ra.get("bird_desc"):  # gen: cards as the generator was trained
        bird_desc.enable(ra["bird_desc"])
    a.cand_order = use_cand_order(ra)  # candidates (and the model, in load_model) exactly as the run was trained
    print(f"candidate order {a.cand_order}, value-side {ra.get('value_side') == 'True'}", flush=True)
    if ra.get("base"):
        a.base, a.subfolder = ra["base"], ra.get("subfolder", "")
    if (a.run / "vocab.json").exists():
        v = json.loads((a.run / "vocab.json").read_text())
        train_sq2.VOCAB_ARGS = (v["templates"], v["nums"])
        train_sq2._init_worker(train_sq2.VOCAB_ARGS)


def gen(a):
    from train_sq2 import load_sets
    use_run_settings(a)
    from eval_sq2 import gold_sql
    # dev: every covered question; with --all-questions every question (uncovered: no plan, gold = original SQL)
    cards, exs, gold = load_sets(a.plans, strict=a.split == "train", all_rows=a.all_questions and a.split == "dev",
                                 questions=a.questions)
    exs = [e for e in exs if not gold[e.id].get("twin_of")]
    dev = set(json.loads(a.dev_dbs.read_text()))
    paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, str(p))
    exs = [e for e in exs if (e.db_id in dev) == (a.split == "dev") and e.db_id in paths]
    if a.split == "train":  # spread over databases; drop exact question repeats
        rnd = random.Random(a.seed)
        rnd.shuffle(exs)
        seenq, keep = set(), []
        for e in exs:
            k = (e.db_id, e.question.lower())
            if k not in seenq:
                seenq.add(k)
                keep.append(e)
        exs = keep
    if a.max_q:
        exs = exs[:a.max_q]
    by_db = collections.defaultdict(list)
    for e in exs:
        by_db[e.db_id].append(e)
    print(f"gen {a.split}: {len(exs)} questions on {len(by_db)} databases", flush=True)
    m = load_slot_model(a.run, a.base, a.subfolder)
    recs, tasks, t0 = [], [], time.time()
    with torch.no_grad():
        for n_db, (db, group) in enumerate(by_db.items()):
            card = cards[db]
            g = card.graph()
            try:
                con = sqlite3.connect(f"file:{paths[db]}?mode=ro", uri=True)
                con.text_factory = lambda b: b.decode("utf-8", "replace")
                vindex = ValueIndex.build(con, card)
                for e in group:
                    body = question_body(gold[e.id])
                    e.cands, _ = select_cands(find_candidates(body, card, extra=vindex.match(body)),
                                              order=a.cand_order, body=body)
                con.close()
            except Exception as ex:
                print(f"value index failed on {db}: {ex}", flush=True)
            qs = []
            for tabs, chunk in serve_chunks(card, group, 8):
                st, oix, enc = encode_db(m, card, chunk, tabs)
                for e, x in zip(chunk, enc):
                    sc = TorchScores(m, x, st.option_ids, e.cands)
                    try:
                        pc = candidates(sc, k=a.k, with_cost=True, wide_singles=a.wide_singles,
                                        wide_pairs=a.wide_pairs, runner_up2=a.runner_up2, samples=a.samples,
                                        temp=a.temp, seed=zlib.crc32(e.id.encode()), distinct=a.distinct,
                                        compile_fn=lambda p: compile_plan(p, g))
                    except Exception:
                        _LOG.warning("%s: candidate search failed, no candidates", e.id, exc_info=True)
                        pc = []
                    cl = []
                    for rank, (p, cost) in enumerate(pc):
                        try:
                            sql = compile_plan(p, g)
                        except Exception:
                            _LOG.warning("%s: plan %d did not compile", e.id, rank, exc_info=True)
                            sql = None
                        cl.append({"rank": rank, "cost": round(cost, 4), "sql": sql})
                    r = gold[e.id]
                    recs.append({"id": e.id, "db_id": db, "question": e.question, "context": r.get("context", ""),
                                 "gold_sql": gold_sql(r), "in_slots": e.t is not None,
                                 "covered": r.get("outcome") == "plan", "cands": cl})
                    qs.append((gold_sql(r), [c["sql"] for c in cl]))
            tasks.append((paths[db], qs, a.seconds))
            if n_db % 50 == 0:
                print(f"  decoded {len(recs)} questions, {n_db + 1}/{len(by_db)} dbs, {time.time() - t0:.0f}s", flush=True)
    del m
    torch.cuda.empty_cache()
    print(f"executing on {a.workers} workers", flush=True)
    i = 0
    with Pool(a.workers) as pool:
        for res in pool.imap(_exec_db, tasks):
            for r in res:
                rec = recs[i]
                i += 1
                if r is None:
                    rec["gold_error"] = True
                    continue
                for c, (runs, nonnull, ok, sig, res) in zip(rec["cands"], r):
                    c.update(runs=runs, nonnull=nonnull, ok=ok, sig=sig, res=res)
    if a.all_questions:  # every question stays in n; one whose gold does not run (time limit) scores 0
        print(f"gold SQL failed on {sum(bool(r.get('gold_error')) for r in recs)} questions (kept, scored 0)", flush=True)
    else:
        recs = [r for r in recs if not r.get("gold_error")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    s = summarize(recs)
    print(f"GEN {a.split} {json.dumps(s)} in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---- picking + metrics -----------------------------------------------------------------------------

def unique(cands: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cands:
        if c.get("sql") and c["sql"] not in seen:
            seen.add(c["sql"])
            out.append(c)
    return out


def pick_fallback(cands: list[dict], max_cost: float = 1.0) -> dict | None:
    """Round-3 phase-1 rule: greedy unless it errors / returns nothing, then the first close alternative."""
    if not cands:
        return None
    g = cands[0]
    if g.get("runs") and g.get("nonnull"):
        return g
    return next((c for c in cands[1:] if c.get("runs") and c.get("nonnull") and c.get("cost", 0.0) <= max_cost), g)


def pick_rerank(cands: list[dict], lam: float = 0.0) -> dict | None:
    u = [c for c in unique(cands) if "rr" in c]
    if not u:
        return cands[0] if cands else None
    live = [c for c in u if c.get("runs") and c.get("nonnull")] or u
    return max(live, key=lambda c: c["rr"] - lam * c.get("cost", 0.0))  # prebuilt lists: mutants carry no cost


def cut(recs: list[dict], n: int) -> list[dict]:
    """The same questions with only the first n candidates (the pool a top-n decoder would have made)."""
    return [{**r, "cands": r["cands"][:n]} for r in recs]


def pool_sizes(recs: list[dict]) -> list[int]:
    m = max((len(r["cands"]) for r in recs), default=0)
    return [n for n in (8, 16, 32) if n < m] + ([m] if m else [])


def summarize(recs: list[dict], lam: float | None = None, pools: bool = True) -> dict:
    if pools and recs and max(len(r["cands"]) for r in recs) > 8:
        s = summarize(recs, pools=False)
        for n in pool_sizes(recs):
            sub = summarize(cut(recs, n), pools=False)
            s[f"oracle@{n}"] = sub["oracle"]
            if "rerank" in sub:
                s[f"rerank@{n}"] = sub["rerank"]
                s[f"rerank_cv@{n}"] = sub["rerank_cv"]
        s["mean_unique"] = round(sum(len(unique(r["cands"])) for r in recs) / len(recs), 2)
        return s
    n = len(recs)
    ok = lambda c: bool(c and c.get("ok"))
    s = {"n": n,
         "greedy": sum(ok(r["cands"][0] if r["cands"] else None) for r in recs) / max(n, 1),
         "fallback": sum(ok(pick_fallback(r["cands"])) for r in recs) / max(n, 1),
         "oracle": sum(any(c.get("ok") for c in r["cands"]) for r in recs) / max(n, 1)}
    if recs and any("rr" in c for r in recs for c in r["cands"]):
        s["rerank"] = sum(ok(pick_rerank(r["cands"])) for r in recs) / n
        s.update(cv_lambda(recs))
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}


LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def cv_lambda(recs: list[dict]) -> dict:
    """rerank score - lam * decoder cost, lam chosen on one half of the databases and scored on the other."""
    dbs = sorted({r["db_id"] for r in recs})
    half = {d: i % 2 for i, d in enumerate(dbs)}
    right = 0
    picks = []
    for f in (0, 1):
        fit = [r for r in recs if half[r["db_id"]] != f]
        test = [r for r in recs if half[r["db_id"]] == f]
        lam = max(LAMS, key=lambda l: sum(bool((pick_rerank(r["cands"], l) or {}).get("ok")) for r in fit))
        picks.append(lam)
        right += sum(bool((pick_rerank(r["cands"], lam) or {}).get("ok")) for r in test)
    return {"rerank_cv": right / len(recs), "cv_lams": str(picks)}


# ---- train / score ------------------------------------------------------------------------------------

def load_cards(dirs: list[Path]) -> dict[str, Path]:
    """db_id -> card path (first directory wins)."""
    cards = {}
    for d in dirs:
        for p in (d / "cards").glob("*.json"):
            cards.setdefault(p.stem, p)
    return cards


def card_of(cache: dict, paths: dict, db: str) -> SchemaCard:
    if db not in cache:
        if len(cache) > 512:
            cache.clear()
        cache[db] = SchemaCard.load(paths[db])
    return cache[db]


def tables_for(card: SchemaCard, sqls: list[str], budget: int) -> list[str] | None:
    """Full schema when it fits; otherwise every table a candidate uses, then others in schema order."""
    if len(schema_text(card).text) <= budget:
        return None
    used = [t.name for t in card.tables if any(f'"{t.name}"' in s for s in sqls)]
    size = {t.name: len(schema_text(card, [t.name]).text) - len(SCHEMA_HEAD) for t in card.tables}
    keep, total = list(used), len(SCHEMA_HEAD) + sum(size[t] for t in used)
    for t in card.tables:
        if t.name not in keep and total + size[t.name] <= budget:
            keep.append(t.name)
            total += size[t.name]
    names = [t.name for t in card.tables]
    return sorted(keep, key=names.index)


def train_items(rec: dict, k: int, rnd: random.Random) -> list[dict] | None:
    u = [c for c in unique(rec["cands"]) if c.get("runs")]
    pos = [c for c in u if c["ok"]]
    neg = [c for c in u if not c["ok"]]
    if not pos or not neg:
        return None
    rnd.shuffle(neg)
    items = [rnd.choice(pos)] + neg[:k - 1]
    rnd.shuffle(items)
    return items


@torch.no_grad()
def score_rows(rr: Reranker, card: SchemaCard, items: list, tabs, max_rows: int) -> list[float]:
    """Scores in chunks of at most max_rows; halves the chunk on out-of-memory."""
    out, i = [], 0
    while i < len(items):
        chunk = items[i:i + max_rows]
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out += rr.scores(card, chunk, tabs).tolist()
            i += len(chunk)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if max_rows == 1:
                raise
            max_rows //= 2
    return out


PREVIEW = False  # round 7: the validator also sees a preview of each candidate's result (--result-preview)
JOINT = {"on": False, "orders": 2}  # --joint-results: every candidate scored with all distinct results in the prefix


def init_seed(a) -> int:
    """--init-seed (LoRA / head init and everything before training); default = --seed."""
    return a.seed if getattr(a, "init_seed", None) is None else a.init_seed


def reseed_after_init(a):
    """Data order and dropout follow --seed. Reseeded only when it differs from the init seed, so --seed 0 (the
    default) keeps the RNG stream of the hard-coded torch.manual_seed(0) runs exactly."""
    if getattr(a, "seed", 0) != init_seed(a):
        torch.manual_seed(a.seed)


def item(r: dict, c: dict) -> tuple:
    return (r["question"], r["context"], c["sql"]) + ((c.get("res") or "error",) if PREVIEW else ())


def joint_orders(r: dict, n_shown: int, n_orders: int) -> list[list[int]]:
    """Score-time table orders of one question: a permutation seeded by the question id, then its reverse, then more
    seeded permutations (deterministic across processes: str seeds are hashed with sha512)."""
    rnd = random.Random(f"joint:{r.get('id', '')}")
    first = list(range(n_shown))
    rnd.shuffle(first)
    out = [first, first[::-1]]
    while len(out) < n_orders:
        p = list(range(n_shown))
        rnd.shuffle(p)
        out.append(p)
    return out[:max(1, n_orders)]


def joint_train_shown(n: int, rnd: random.Random) -> list[int]:
    """Training: a random subset of at most JOINT_MAX_RESULTS results in random order (prebuilt lists put the right
    candidate first, so neither the cut nor the order may follow list order)."""
    return rnd.sample(range(n), min(n, JOINT_MAX_RESULTS))


@torch.no_grad()
def score_ids_rows(rr: Reranker, ids: list[int], rows: list[list[int]], max_rows: int) -> list[float]:
    """score_rows for a token prefix + token rows (joint scoring)."""
    out, i = [], 0
    while i < len(rows):
        chunk = rows[i:i + max_rows]
        try:
            with _autocast(rr):
                out += rr.scores_ids(ids, chunk).float().tolist()
            i += len(chunk)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if max_rows == 1:
                raise
            max_rows //= 2
    return out


@torch.no_grad()
def score_file_joint(rr: Reranker, recs: list[dict], card_paths: dict, budget: int, max_rows: int = 32,
                     max_tok: int = 384):
    """Joint scoring: per question, the first JOINT_MAX_RESULTS distinct live results (pool order) in the prefix;
    'rr' = mean score over JOINT['orders'] table orders."""
    rr.eval()
    rr.activate(train=False)
    tok = rr.backbone.tok
    cache = {}
    for r in recs:
        cs = unique(r["cands"])
        if not cs:
            continue
        card = card_of(cache, card_paths, r["db_id"])
        tabs = tables_for(card, [c["sql"] for c in cs], budget)
        n = len(result_classes(cs)[0])
        tot = [0.0] * len(cs)
        orders = joint_orders(r, min(n, JOINT_MAX_RESULTS), JOINT["orders"])
        for shown in orders:
            head, texts = joint_texts(r["question"], r["context"], cs, shown)
            rows = [tok(t, add_special_tokens=False)["input_ids"][-max_tok:] for t in texts]
            for k, v in enumerate(score_ids_rows(rr, joint_prefix_ids(tok, card, tabs, head), rows, max_rows)):
                tot[k] += v
        for c, v in zip(cs, tot):
            c["rr"] = v / len(orders)
    rr.train()


@torch.no_grad()
def score_file(rr: Reranker, recs: list[dict], card_paths: dict, budget: int, q_per_batch: int = 4,
               max_rows: int = 32):
    if JOINT["on"]:
        return score_file_joint(rr, recs, card_paths, budget, max_rows)
    rr.eval()
    rr.activate(train=False)
    cache = {}
    by_db = collections.defaultdict(list)
    for r in recs:
        by_db[r["db_id"]].append(r)
    for db, group in by_db.items():
        card = card_of(cache, card_paths, db)
        for i in range(0, len(group), q_per_batch):
            chunk = group[i:i + q_per_batch]
            items, where = [], []
            for r in chunk:
                for c in unique(r["cands"]):
                    items.append(item(r, c))
                    where.append(c)
            if not items:
                continue
            tabs = tables_for(card, [x[2] for x in items], budget)
            for c, v in zip(where, score_rows(rr, card, items, tabs, max_rows)):
                c["rr"] = v
    rr.train()


def db_step(rr: Reranker, card: SchemaCard, rs: list[dict], a, rnd: random.Random, weight: float):
    """Forward + backward for some questions on one database. On out-of-memory, retry each half separately
    (the schema pass is repeated), down to one question; only a single question that still does not fit
    is skipped. Returns (summed loss, passes that ran, questions skipped)."""
    items, groups = [], []
    for r in rs:
        it = train_items(r, a.train_k, rnd)
        if it:
            groups.append((len(items), [c["ok"] for c in it]))
            items += [item(r, c) for c in it]
    if not items:
        return 0.0, 0, 0
    tabs = tables_for(card, [x[2] for x in items], a.budget_chars)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = rr.scores(card, items, tabs)
            loss = torch.stack([listwise_loss(s[o:o + len(ok)], ok) for o, ok in groups]).mean()
        (loss * weight).backward()
        return float(loss), 1, 0
    except torch.OutOfMemoryError:
        pass
    s = loss = None  # drop the failed graph before retrying
    gc.collect()
    torch.cuda.empty_cache()
    if len(rs) == 1:
        print(f"OOM on {card.db_id}: one question skipped", flush=True)
        return 0.0, 0, 1
    h = len(rs) // 2
    l1, n1, s1 = db_step(rr, card, rs[:h], a, rnd, weight * h / len(rs))
    l2, n2, s2 = db_step(rr, card, rs[h:], a, rnd, weight * (len(rs) - h) / len(rs))
    return l1 + l2, n1 + n2, s1 + s2


def train(a):
    global PREVIEW
    if v2_requested(a):
        return train_v2(a)
    PREVIEW = a.result_preview
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "settings.json").write_text(json.dumps({"result_preview": PREVIEW}))
    card_paths = load_cards(a.cards)
    tr = [json.loads(l) for l in open(a.train_cands)]
    dv = [json.loads(l) for l in open(a.dev_cands)] if a.dev_cands else []
    rnd = random.Random(a.seed)
    usable = [r for r in tr if train_items(r, a.train_k, random.Random(0)) and r["db_id"] in card_paths]
    print(f"train questions with right and wrong candidates: {len(usable)} of {len(tr)}", flush=True)
    bb = Backbone.from_pretrained(a.base, a.subfolder or None, lora_r=32, grad_ckpt=False)
    rr = Reranker(bb, lora_r=a.lora_r).cuda()
    rr.backbone.model.to(torch.bfloat16)
    reseed_after_init(a)
    rr.activate(train=True)
    params = [p for p in rr.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([{"params": [p for n, p in rr.named_parameters() if p.requires_grad and n.startswith("backbone.")], "lr": a.lr},
                             {"params": list(rr.score.parameters()), "lr": a.head_lr}], weight_decay=0.01)
    by_db = collections.defaultdict(list)
    for r in usable:
        by_db[r["db_id"]].append(r)
    total = a.max_steps or int(math.ceil(len(usable) / (a.q_per_db * a.dbs_per_step)) * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 50) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    cache, step, t0, ep = {}, 0, time.time(), 0
    n_skip = n_split = 0
    a.out.mkdir(parents=True, exist_ok=True)
    rr.train()
    while step < total:
        chunks = []
        for db, rs in by_db.items():
            rs = rs[:]
            rnd.shuffle(rs)
            chunks += [(db, rs[i:i + a.q_per_db]) for i in range(0, len(rs), a.q_per_db)]
        rnd.shuffle(chunks)
        for i in range(0, len(chunks), a.dbs_per_step):
            loss_sum, n_ok = 0.0, 0
            for db, rs in chunks[i:i + a.dbs_per_step]:
                card = card_of(cache, card_paths, db)
                l, n, skipped = db_step(rr, card, rs, a, rnd, 1.0 / a.dbs_per_step)
                loss_sum += l
                n_ok += n
                n_skip += skipped
                n_split += max(0, n - 1)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 25 == 0:
                print(f"rr step {step}/{total} ep {ep} loss {loss_sum / max(n_ok, 1):.3f} skipped {n_skip} "
                      f"split {n_split} "
                      f"{(time.time() - t0) / step:.2f}s/step mem {torch.cuda.memory_allocated() / 2**30:.1f}/"
                      f"{torch.cuda.max_memory_allocated() / 2**30:.0f}GB", flush=True)
            if a.hours and time.time() - t0 > a.hours * 3600:
                print(f"time limit at step {step}", flush=True)
                total = step
            if step >= total:
                break
        ep += 1
    rr.backbone.model.save_pretrained(a.out / "rerank_lora", selected_adapters=["rerank"])
    torch.save(rr.head_state(), a.out / "rerank_head.pt")
    print(f"trained {step} steps in {(time.time() - t0) / 3600:.2f} h; questions skipped for memory {n_skip}", flush=True)
    if dv:
        score_file(rr, dv, card_paths, a.budget_chars)
        (a.out / "dev_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dv))
        s = summarize(dv)
        print(f"RERANK DEV {json.dumps(s)}", flush=True)
        (a.out / "dev_metrics.json").write_text(json.dumps(s, indent=1))
        cov = [r for r in dv if r["in_slots"]]
        print(f"RERANK DEV (slot-expressible only) {json.dumps(summarize(cov))}", flush=True)


# ---- train v2 (docs/paper_notes/research/selection_memo_2026-10-10.md, step 1 "Recipe A+") ------------------------------
# Prebuilt lists (--train-file: one live representative per result class, mutants as ordinary negatives), every
# candidate of a list scored; token-budget batching with the gradient cache for lists that do not fit, so no question
# is skipped for memory; multipos loss with label smoothing; per-question "weight"; warmup + cosine to 0 over the real
# step count; val pick accuracy per epoch with the best checkpoint kept; optional db_id folds with out-of-fold scores.
# Only these flags switch it on (v2_requested); without them `train` is the round-3 trainer above, unchanged.
# Checkpoint layout is the same (rerank_lora/ + rerank_head.pt + settings.json), so `score` reads it as before.

ROW_COST = 8  # a row token (activations saved for backward) costs ~8 context tokens it attends over (cache + attention)


def v2_requested(a) -> bool:
    return bool(a.train_file or a.tok_budget or a.loss != "listwise" or a.sched != "legacy" or a.val_file or a.fold
                or getattr(a, "joint_results", False) or getattr(a, "eval_every", 0) or getattr(a, "avg_last", 0))


def list_items(r: dict) -> list[dict]:
    """A prebuilt list's candidates: unique SQL that runs (mutants are ordinary candidates)."""
    return [c for c in unique(r["cands"]) if c.get("runs") or c.get("mutant")]


def usable_list(r: dict) -> bool:
    oks = [bool(c.get("ok")) for c in list_items(r)]
    return any(oks) and not all(oks)


def fold_of(key: str, n: int) -> int:
    """Stable across processes and machines (Python's hash() is salted per process)."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % n


def parse_fold(s: str) -> tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    if not (n >= 2 and 0 <= i < n):
        raise SystemExit(f"--fold {s}: want i/n with n >= 2 and 0 <= i < n")
    return i, n


def chunk_cost(n: int, T: int, P: int) -> int:
    """Memory units of scoring n rows of padded length T after a P-token schema prefix with a graph: each row attends
    over (and holds an expanded copy of) the prefix, its own tokens save activations (ROW_COST each), and the prefix
    pass's graph (ROW_COST * P) stays alive while the rows run."""
    return n * (P + ROW_COST * T) + ROW_COST * P


def plan_chunks(lens: list[int], P: int, budget: int) -> list[list[int]]:
    """Row indices, longest first (little padding), cut so each chunk fits the budget; a row that alone exceeds it is
    its own chunk."""
    chunks, cur = [], []
    for i in sorted(range(len(lens)), key=lambda i: -lens[i]):
        if cur and chunk_cost(len(cur) + 1, lens[cur[0]], P) > budget:
            chunks.append(cur)
            cur = []
        cur.append(i)
    return chunks + ([cur] if cur else [])


def _autocast(rr):
    return torch.autocast(rr.device.type, dtype=torch.bfloat16, enabled=rr.device.type == "cuda")


def question_loss(a, s: torch.Tensor, ok: list[bool], gid: list | None = None) -> torch.Tensor:
    if a.loss == "group":  # VALHARD: one term per result group (gid = result signature)
        return group_loss(s, ok, gid if gid is not None else list(range(len(ok))), a.label_smooth)
    return multipos_loss(s, ok, a.label_smooth) if a.loss == "multipos" else listwise_loss(s, ok)


def new_stats() -> dict:
    return {"direct": 0, "gradcache": 0, "chunks": 0, "oom": 0, "skipped": collections.Counter()}


def db_step_v2(rr: Reranker, card: SchemaCard, rs: list[dict], a, scale: float, wbar: float,
               stats: dict) -> tuple[float, int]:
    """Forward + backward for some questions on one database, every candidate of each list. One pass when all rows fit
    --tok-budget (chunk_cost), otherwise the gradient cache over chunks (exact: rerank.gradcache_backward). On
    out-of-memory the gradients of the failed try are restored, the budget halves and the questions run again; they are
    skipped (stats['skipped'][db]) only if one-row chunks still do not fit. Question q adds scale * weight_q / wbar *
    loss_q. Returns (sum of the questions' unweighted losses, questions trained).
    --joint-results: each question has its own prefix (schema + question + results table), so the questions run one
    by one through the same one-pass / gradient-cache / out-of-memory logic (losses add, so gradients are the same sum)."""
    if getattr(a, "joint_results", False):
        return joint_step_v2(rr, card, rs, a, scale, wbar, stats)
    items, groups = [], []
    for r in rs:
        it = list_items(r)
        groups.append((len(items), [bool(c.get("ok")) for c in it], float(r.get("weight", 1.0)) / wbar,
                       [c.get("sig") or f"_row{k}" for k, c in enumerate(it)]))
        items += [item(r, c) for c in it]
    if not items:
        return 0.0, 0
    tabs = tables_for(card, [x[2] for x in items], a.budget_chars)
    tok = rr.backbone.tok
    P = len(tok(schema_text(card, tabs).text, add_special_tokens=False)["input_ids"])
    rows = row_ids(tok, items)
    return backward_lists(rr, card.db_id, P, rows, groups, a, scale, stats,
                          direct=lambda: rr.scores(card, items, tabs),
                          prefix=lambda: prefix_kv(rr, card, tabs)[0],
                          hybrid=lambda idx: rr.scores(card, [items[i] for i in idx], tabs))


def joint_step_v2(rr: Reranker, card: SchemaCard, rs: list[dict], a, scale: float, wbar: float,
                  stats: dict) -> tuple[float, int]:
    """db_step_v2 with --joint-results: one prefix per question (schema + question + a random subset/order of its
    distinct results, rerank.joint_texts), its candidates as rows."""
    tok = rr.backbone.tok
    jrnd = a.__dict__.setdefault("_jrnd", random.Random(f"joint-train:{getattr(a, 'seed', 0)}"))
    loss_sum, n_q = 0.0, 0
    for r in rs:
        it = list_items(r)
        if not it:
            continue
        group = (0, [bool(c.get("ok")) for c in it], float(r.get("weight", 1.0)) / wbar,
                 [c.get("sig") or f"_row{k}" for k, c in enumerate(it)])
        head, texts = joint_texts(r["question"], r["context"], it, joint_train_shown(len(result_classes(it)[0]), jrnd))
        tabs = tables_for(card, [c["sql"] for c in it], a.budget_chars)
        ids = joint_prefix_ids(tok, card, tabs, head)
        rows = [tok(t, add_special_tokens=False)["input_ids"][-384:] for t in texts]
        l, n = backward_lists(rr, card.db_id, len(ids), rows, [group], a, scale, stats,
                              direct=lambda: rr.scores_ids(ids, rows),
                              prefix=lambda: prefix_kv_ids(rr, ids)[0],
                              hybrid=lambda idx: rr.scores_ids(ids, [rows[i] for i in idx]))
        loss_sum += l
        n_q += n
    return loss_sum, n_q


def backward_lists(rr: Reranker, db_id: str, P: int, rows: list[list[int]], groups: list[tuple], a, scale: float,
                   stats: dict, direct, prefix, hybrid) -> tuple[float, int]:
    """The one-pass / gradient-cache / out-of-memory loop of db_step_v2 over token rows after a P-token prefix.
    direct() scores all rows in one pass; prefix() -> the prefix key/values with a graph (full attention);
    hybrid(idx) scores rows idx recomputing the prefix. groups: (offset, ok flags, weight, result ids) per question."""
    lens = [len(x) for x in rows]
    terms = []

    def loss_fn(s):
        ls = [question_loss(a, s[o:o + len(ok)], ok, gid) for o, ok, _, gid in groups]
        terms.append(sum(float(x.detach()) for x in ls))
        return scale * sum(w * x for (_, _, w, _), x in zip(groups, ls))

    params = [p for p in rr.parameters() if p.requires_grad]
    budget = a.tok_budget
    while True:
        saved = [None if p.grad is None else p.grad.clone() for p in params]
        oom = False
        try:
            if chunk_cost(len(rows), max(lens), P) <= budget:
                with _autocast(rr):
                    loss = loss_fn(direct().float())
                loss.backward()
                stats["direct"] += 1
            else:
                chunks = plan_chunks(lens, P, budget)
                order = torch.tensor([i for c in chunks for i in c])
                inv = torch.empty_like(order)
                inv[order] = torch.arange(len(order))
                inv = inv.to(rr.device)
                with _autocast(rr):
                    if full_attention(rr.body):  # schema pass once; its gradient accumulated over the chunks
                        gradcache_backward(prefix,
                                           lambda kv, c: rows_over_prefix(rr, kv, P, [rows[i] for i in chunks[c]]),
                                           len(chunks), lambda s: loss_fn(s[inv]))
                    else:  # hybrid caches: each chunk recomputes the schema pass from the same RNG state
                        gradcache_backward(lambda: [], lambda kv, c: hybrid(chunks[c]),
                                           len(chunks), lambda s: loss_fn(s[inv]), replay_prefix=True)
                stats["gradcache"] += 1
                stats["chunks"] += len(chunks)
            return terms[-1], len(groups)
        except torch.OutOfMemoryError:
            oom = True
        if oom:  # outside the except block, so the failed graph is not kept alive by the traceback
            loss = None
            for p, g in zip(params, saved):
                p.grad = g
            saved = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stats["oom"] += 1
            if budget < chunk_cost(1, max(lens), P):  # one-row chunks were just tried
                stats["skipped"][db_id] += len(groups)
                print(f"OOM on {db_id}: {len(groups)} question(s) skipped even with one-row chunks "
                      f"(prefix {P} tokens)", flush=True)
                return 0.0, 0
            budget //= 2


def save_rr(rr: Reranker, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    rr.backbone.model.save_pretrained(d / "rerank_lora", selected_adapters=["rerank"])
    torch.save(rr.head_state(), d / "rerank_head.pt")


def eval_points(n_steps: int, every: float) -> dict[int, float]:
    """--eval-every f (0 < f < 1): steps inside an epoch of n_steps after which val is scored -> {step in epoch:
    epoch fraction}; the epoch end is always evaluated by the caller. 0 = epoch ends only (the old behaviour)."""
    if not every or every >= 1:
        return {}
    out, k = {}, 1
    while k * every < 1 - 1e-9:
        j = int(round(n_steps * k * every))
        if 0 < j < n_steps:
            out[j] = round(k * every, 6)
        k += 1
    return out


def val_group_nll(recs: list[dict]) -> float:
    """Mean unsmoothed group loss (--loss group's objective, label smoothing 0) of the scored val lists: a continuous
    checkpoint signal next to pick accuracy (03_objective.md §2.3)."""
    out = []
    for r in recs:
        it = [c for c in list_items(r) if "rr" in c]
        oks = [bool(c.get("ok")) for c in it]
        if any(oks) and not all(oks):
            s = torch.tensor([float(c["rr"]) for c in it])
            out.append(float(group_loss(s, oks, [c.get("sig") or f"_row{k}" for k, c in enumerate(it)])))
    return sum(out) / max(len(out), 1)


def pick_acc(recs: list[dict]) -> float:
    """Share of questions whose best-scored live candidate is right (pick_rerank)."""
    return sum(bool((pick_rerank(r["cands"]) or {}).get("ok")) for r in recs) / max(len(recs), 1)


def lr_factor(a, s: int, total: int) -> float:
    if a.sched == "cosine":
        warm = max(1, int(round(a.warmup * total)))
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(total - warm, 1))))
    return min(1.0, s / 50) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1))))


def train_v2(a):
    global PREVIEW
    PREVIEW = a.result_preview
    JOINT.update(on=bool(a.joint_results), orders=a.joint_orders or 2)
    src = a.train_file or a.train_cands
    if src is None:
        raise SystemExit("train: --train-file (or --train-cands) is required")
    if a.avg_last and not a.val_file:
        raise SystemExit("train: --avg-last averages the checkpoints taken at val evaluations; it needs --val-file")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "settings.json").write_text(json.dumps({
        "result_preview": PREVIEW, "trainer": "v2", "train_file": str(src), "val_file": str(a.val_file or ""),
        "loss": a.loss, "label_smooth": a.label_smooth, "sched": a.sched, "warmup": a.warmup, "epochs": a.epochs,
        "tok_budget": a.tok_budget, "fold": a.fold, "fold_by": a.fold_by, "lora_r": a.lora_r, "base": a.base,
        "lr": a.lr, "head_lr": a.head_lr, "q_per_db": a.q_per_db, "dbs_per_step": a.dbs_per_step,
        "seed": a.seed, "init_seed": init_seed(a), "eval_every": a.eval_every, "avg_last": a.avg_last,
        "joint_results": JOINT["on"], "joint_orders": JOINT["orders"], "joint_max_results": JOINT_MAX_RESULTS}))
    card_paths = load_cards(a.cards)
    recs = [json.loads(l) for l in open(src)]
    missing = sorted({r["db_id"] for r in recs if r["db_id"] not in card_paths})
    if missing:
        print(f"no card for {len(missing)} databases, their questions dropped: {missing[:20]}", flush=True)
    recs = [r for r in recs if r["db_id"] in card_paths]
    held = []
    if a.fold:
        fi, fn = parse_fold(a.fold)
        held = [r for r in recs if fold_of(str(r[a.fold_by]), fn) == fi]
        recs = [r for r in recs if fold_of(str(r[a.fold_by]), fn) != fi]
        print(f"fold {fi}/{fn} by {a.fold_by}: train on {len({r[a.fold_by] for r in recs})} groups, hold out "
              f"{len({r[a.fold_by] for r in held})} ({len(held)} questions, scored out-of-fold at the end)", flush=True)
    usable = [r for r in recs if usable_list(r)]
    wbar = sum(float(r.get("weight", 1.0)) for r in usable) / max(len(usable), 1)
    n_mut = sum(bool(c.get("mutant")) for r in usable for c in list_items(r))
    print(f"train lists with right and wrong candidates: {len(usable)} of {len(recs)}; "
          f"{sum(len(list_items(r)) for r in usable) / max(len(usable), 1):.1f} candidates per list, {n_mut} mutants; "
          f"mean weight {wbar:.3f}", flush=True)
    vl = [json.loads(l) for l in open(a.val_file)] if a.val_file else []
    vl = [r for r in vl if r["db_id"] in card_paths]
    dv = [json.loads(l) for l in open(a.dev_cands)] if a.dev_cands else []
    bb = Backbone.from_pretrained(a.base, a.subfolder or None, lora_r=32, grad_ckpt=False)
    rr = Reranker(bb, lora_r=a.lora_r).cuda()
    rr.backbone.model.to(torch.bfloat16)
    reseed_after_init(a)
    rr.activate(train=True)
    params = [p for p in rr.parameters() if p.requires_grad]
    print(f"seed {a.seed} (init {init_seed(a)}); eval every {a.eval_every or 1} epoch; avg last {a.avg_last}; "
          f"joint results {JOINT['on']} (orders {JOINT['orders']})", flush=True)
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M; full-attention prefix cache: "
          f"{full_attention(rr.body)}", flush=True)
    opt = torch.optim.AdamW([{"params": [p for n, p in rr.named_parameters() if p.requires_grad and n.startswith("backbone.")], "lr": a.lr},
                             {"params": list(rr.score.parameters()), "lr": a.head_lr}], weight_decay=0.01)
    by_db = collections.defaultdict(list)
    for r in usable:
        by_db[r["db_id"]].append(r)
    per_epoch = math.ceil(sum(math.ceil(len(rs) / a.q_per_db) for rs in by_db.values()) / a.dbs_per_step)
    box = {"total": a.max_steps or max(1, int(round(per_epoch * a.epochs)))}
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(a, s, box["total"]))
    print(f"{per_epoch} steps per epoch, {box['total']} steps ({a.sched} schedule)", flush=True)
    rnd = random.Random(a.seed)
    cache, step, t0, ep = {}, 0, time.time(), 0
    stats = new_stats()
    hist, best = [], {"acc": -1.0, "state": None, "step": -1}
    val_secs, last_eval = 0.0, -1
    tail = collections.deque(maxlen=a.avg_last or None)  # (step, trainable state on the CPU) at the last evaluations
    best_dir = a.out / "best" if a.avg_last else a.out  # with --avg-last, --out itself holds the average
    evals_per_epoch = len(eval_points(per_epoch, a.eval_every)) + 1

    def trainable_state():
        return {n: p.detach().to("cpu", copy=True) for n, p in rr.named_parameters() if p.requires_grad}

    def evaluate(tag, keep=True):
        nonlocal val_secs, last_eval
        t = time.time()
        score_file(rr, vl, card_paths, a.budget_chars)
        rr.activate(train=True)
        rr.train()
        acc, s = pick_acc(vl), summarize(vl, pools=False)
        extra = {"group_nll": round(val_group_nll(vl), 5)} if (a.eval_every or a.avg_last) else {}
        hist.append({"tag": tag, "step": step, "pick_acc": acc, **extra, **s})
        (a.out / "val_history.json").write_text(json.dumps(hist, indent=1))
        print(f"VAL {tag} step {step} pick_acc {acc:.4f} {json.dumps({**extra, **s})}", flush=True)
        if keep and a.avg_last:
            tail.append((step, trainable_state()))
            ck = a.out / "ckpt"
            save_rr(rr, ck / f"step{step:06d}")
            keep_dirs = {f"step{st:06d}" for st, _ in tail}
            for d in ck.iterdir():  # only the last --avg-last checkpoints stay on disk
                if d.is_dir() and d.name not in keep_dirs:
                    shutil.rmtree(d, ignore_errors=True)
        if keep and acc > best["acc"]:
            best.update(acc=acc, step=step, state=trainable_state())
            save_rr(rr, best_dir)
            if not a.avg_last:
                (a.out / "val_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in vl))
            print(f"best checkpoint (val pick_acc {acc:.4f}, step {step}) -> {best_dir}", flush=True)
        val_secs, last_eval = time.time() - t, step

    rr.train()
    stop = False
    while step < box["total"] and not stop:
        chunks = []
        for db, rs in by_db.items():
            rs = rs[:]
            rnd.shuffle(rs)
            chunks += [(db, rs[i:i + a.q_per_db]) for i in range(0, len(rs), a.q_per_db)]
        rnd.shuffle(chunks)
        for i in range(0, len(chunks), a.dbs_per_step):
            batch = chunks[i:i + a.dbs_per_step]
            nq = sum(len(rs) for _, rs in batch)
            loss_sum, n_q = 0.0, 0
            for db, rs in batch:
                l, n = db_step_v2(rr, card_of(cache, card_paths, db), rs, a, 1.0 / nq, wbar, stats)
                loss_sum += l
                n_q += n
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            el = time.time() - t0
            if step % 25 == 0:
                print(f"rr step {step}/{box['total']} ep {ep} loss {loss_sum / max(n_q, 1):.3f} lr {sched.get_last_lr()[0]:.2e} "
                      f"direct {stats['direct']} gradcache {stats['gradcache']} ({stats['chunks']} chunks) oom {stats['oom']} "
                      f"skipped {sum(stats['skipped'].values())} {el / step:.2f}s/step mem "
                      f"{torch.cuda.memory_allocated() / 2**30:.1f}/{torch.cuda.max_memory_allocated() / 2**30:.0f}GB", flush=True)
                if a.hours and a.sched == "cosine":  # finish the cosine inside the time limit (val passes included)
                    left_evals = (math.ceil((box["total"] - step) / per_epoch * evals_per_epoch) + 1
                                  + (1 if a.avg_last else 0)) if vl else 0
                    fit = step + int((a.hours * 3600 - el - left_evals * val_secs) / (el / step))
                    if fit < box["total"]:
                        print(f"time limit: schedule shortened {box['total']} -> {max(fit, step + 1)} steps", flush=True)
                        box["total"] = max(fit, step + 1)
            if a.hours and el > a.hours * 3600:
                print(f"time limit at step {step}", flush=True)
                box["total"] = step
            if step >= box["total"]:
                stop = i + a.dbs_per_step < len(chunks)  # stopped inside the epoch
                break
            j = i // a.dbs_per_step + 1  # steps done in this epoch
            mids = eval_points(math.ceil(len(chunks) / a.dbs_per_step), a.eval_every)
            if vl and j in mids:
                evaluate(f"epoch {ep + mids[j]:g}")
        if not stop:
            ep += 1
            if vl:
                evaluate(f"epoch {ep}")
        if step >= box["total"]:
            break
    print(f"trained {step} steps ({ep} full epochs) in {(time.time() - t0) / 3600:.2f} h; one pass {stats['direct']}, "
          f"gradient cache {stats['gradcache']} ({stats['chunks']} chunks), out-of-memory retries {stats['oom']}", flush=True)
    print(f"SKIPPED_FOR_MEMORY {sum(stats['skipped'].values())} {json.dumps(dict(stats['skipped']))}", flush=True)
    if vl:
        if last_eval != step:
            evaluate("final")
        save_rr(rr, a.out / "last")
    if vl and a.avg_last:
        steps = [st for st, _ in tail]
        avg = average_states([sd for _, sd in tail])
        with torch.no_grad():
            for n, p in rr.named_parameters():
                if n in avg:
                    p.copy_(avg[n].to(p.device))
        save_rr(rr, a.out)
        (a.out / "avg_steps.json").write_text(json.dumps(steps))
        print(f"averaged {len(steps)} checkpoints (steps {steps}) -> {a.out}; best single (val pick_acc "
              f"{best['acc']:.4f}, step {best['step']}) kept in {best_dir}", flush=True)
        evaluate(f"avg_last{len(steps)}", keep=False)
        (a.out / "val_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in vl))
    elif vl:
        if best["state"] is not None and best["step"] != step:
            with torch.no_grad():
                for n, p in rr.named_parameters():
                    if n in best["state"]:
                        p.copy_(best["state"][n].to(p.device))
        print(f"VAL best pick_acc {best['acc']:.4f} at step {best['step']} (kept in {a.out})", flush=True)
    else:
        save_rr(rr, a.out)
    if a.avg_last or JOINT["on"]:  # sub-checkpoints score as trained (`score --rr` reads settings.json: joint)
        for d in [a.out / "best", a.out / "last"] + sorted((a.out / "ckpt").glob("step*")):
            if (d / "rerank_head.pt").exists():
                (d / "settings.json").write_text((a.out / "settings.json").read_text())
    if held:
        score_file(rr, held, card_paths, a.budget_chars)
        (a.out / "oof_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held))
        print(f"OOF {json.dumps(summarize(held, pools=False))}", flush=True)
    if dv:
        score_file(rr, dv, card_paths, a.budget_chars)
        (a.out / "dev_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dv))
        s = summarize(dv)
        print(f"RERANK DEV {json.dumps(s)}", flush=True)
        (a.out / "dev_metrics.json").write_text(json.dumps(s, indent=1))


def validator_settings(rr_dir: Path, joint_results: bool = False, joint_orders: int | None = None):
    """score: the validator's settings.json (result preview, joint scoring) into PREVIEW / JOINT; --joint-results /
    --joint-orders override. (Also imported by scripts/analysis/latency/sys1_latency.py.)"""
    global PREVIEW
    st = rr_dir / "settings.json"
    cfg = json.loads(st.read_text()) if st.exists() else {}
    PREVIEW = cfg.get("result_preview", False)
    # joint scoring as the validator was trained (settings.json); --joint-results / --joint-orders override
    JOINT.update(on=bool(cfg.get("joint_results", False) or joint_results),
                 orders=joint_orders or cfg.get("joint_orders", 2))
    if joint_results and not cfg.get("joint_results", False):
        print("WARNING: --joint-results on a validator not trained with joint results", flush=True)
    if JOINT["on"]:
        print(f"joint results scoring, {JOINT['orders']} table order(s)", flush=True)


def rerank_head_kind(rr_dir: Path) -> str:
    """settings.json "rerank_head": linear (default, every validator before option 3) or mlp."""
    st = rr_dir / "settings.json"
    return json.loads(st.read_text()).get("rerank_head", "linear") if st.exists() else "linear"


def shared_adapter(rr_dir: Path) -> bool:
    """A single-LoRA System 1 dir (scripts/single_lora.py): the validator scores with the generator's own adapter."""
    st = rr_dir / "settings.json"
    return st.exists() and bool(json.loads(st.read_text()).get("shared_adapter"))


def shared_validator(model: LuthorSQ2, rr_dir: Path) -> Reranker:
    """A shared-adapter dir's validator on an already loaded slot model (one backbone in memory, not two): the rerank
    head over the slot model's own adapter. score_file leaves the backbone in train mode, so a caller that generates
    after scoring puts it back in eval mode (model.eval())."""
    rr = Reranker(model.backbone, adapter="default", head=rerank_head_kind(rr_dir))
    _load_head(rr, rr_dir)
    return rr.to(model.device)


def _load_head(rr: Reranker, rr_dir: Path) -> None:
    """The rerank head, checked: every head tensor must be in the file (a linear/mlp mismatch fails loudly)."""
    sd = torch.load(rr_dir / "rerank_head.pt", map_location="cpu")
    missing = [k for k in rr.head_state() if k not in sd]
    if missing:
        raise SystemExit(f"{rr_dir}/rerank_head.pt lacks {missing} (head kind {rerank_head_kind(rr_dir)})")
    rr.load_state_dict(sd, strict=False)


def load_validator(rr_dir: Path, base: str, subfolder: str, lora_r: int) -> Reranker:
    """score: the frozen backbone + the trained rerank LoRA + head, on the GPU in bf16. A shared-adapter dir
    (single_lora.py) holds one LoRA (best_lora/, rank from args.json) used by the generator and the validator."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    if shared_adapter(rr_dir):
        ra = json.loads((rr_dir / "args.json").read_text())
        bb = Backbone.from_pretrained(base, subfolder or None, lora_r=int(ra.get("lora_r", 32)), grad_ckpt=False)
        rr = Reranker(bb, adapter="default", head=rerank_head_kind(rr_dir))
        set_peft_model_state_dict(rr.backbone.model, load_file(str(rr_dir / "best_lora" / "adapter_model.safetensors")))
        _load_head(rr, rr_dir)
        rr = rr.cuda()
        rr.backbone.model.to(torch.bfloat16)
        return rr
    bb = Backbone.from_pretrained(base, subfolder or None, lora_r=32, grad_ckpt=False)
    rr = Reranker(bb, lora_r=lora_r)
    set_peft_model_state_dict(rr.backbone.model, load_file(str(rr_dir / "rerank_lora" / "rerank" / "adapter_model.safetensors")),
                              adapter_name="rerank")
    rr.load_state_dict(torch.load(rr_dir / "rerank_head.pt", map_location="cpu"), strict=False)
    rr = rr.cuda()
    rr.backbone.model.to(torch.bfloat16)
    return rr


def score(a):
    """Score a candidates file (e.g. slot + SQL-writer union) with a trained reranker; print the metrics."""
    validator_settings(a.rr, a.joint_results, a.joint_orders)
    card_paths = load_cards(a.cards)
    recs = [json.loads(l) for l in open(a.dev_cands)]
    recs = [r for r in recs if r["db_id"] in card_paths]
    rr = load_validator(a.rr, a.base, a.subfolder, a.lora_r)
    score_file(rr, recs, card_paths, a.budget_chars)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    s = summarize(recs, pools=False)
    W = [{**r, "cands": [c for c in r["cands"] if c.get("src") == "w"]} for r in recs]
    s["writer_rerank"] = summarize(W, pools=False).get("rerank")
    print(f"SCORE {json.dumps(s)}", flush=True)
    print(f"SCORE (slot-expressible only) {json.dumps(summarize([r for r in recs if r.get('in_slots')], pools=False))}", flush=True)


def main():
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN attention backward failed on long schemas (train_sq2.py)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "train", "score"])
    ap.add_argument("--rr", type=Path, help="score: directory with rerank_lora/ and rerank_head.pt")
    ap.add_argument("--run", type=Path)
    ap.add_argument("--plans", nargs="*", type=Path, default=[])
    ap.add_argument("--cards", nargs="*", type=Path, default=[])
    ap.add_argument("--db-roots", nargs="*", default=[])
    ap.add_argument("--dev-dbs", type=Path)
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--train-k", type=int, default=8, help="candidates per question in a reranker training list")
    ap.add_argument("--wide-singles", type=int, default=0)
    ap.add_argument("--wide-pairs", type=int, default=0)
    ap.add_argument("--runner-up2", type=int, default=0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--distinct", action="store_true", help="gen: k different plans (round 7)")
    ap.add_argument("--result-preview", action="store_true", help="train: show the validator each result's shape and first row")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--all-questions", action="store_true",
                    help="gen --split dev: every question, uncovered ones too (no plan: scored by executing each candidate "
                         "against the original SQL's result); BIRD dev then has all 1,534 with --questions")
    ap.add_argument("--questions", type=Path, help="with --all-questions: BIRD-format dev.json, adds questions with no plan row")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-cands", type=Path)
    ap.add_argument("--dev-cands", type=Path)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--hours", "--max-hours", dest="hours", type=float, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--q-per-db", type=int, default=4)
    ap.add_argument("--dbs-per-step", type=int, default=4)
    ap.add_argument("--budget-chars", type=int, default=9000)
    # train v2 (any of these switches it on; see train_v2)
    ap.add_argument("--train-file", type=Path, help="train v2: prebuilt lists (cands_train.jsonl format; 'mutant' and "
                    "'weight' fields honoured); every candidate of a list is scored")
    ap.add_argument("--val-file", type=Path, help="train v2: lists scored after each epoch (pick accuracy); the best "
                    "epoch's checkpoint is kept in --out (the last one in --out/last)")
    ap.add_argument("--tok-budget", type=int, default=0, help="train v2: memory units per pass (chunk_cost); lists over "
                    "it are split into chunks with the gradient cache instead of skipped")
    ap.add_argument("--loss", choices=["listwise", "multipos", "group"], default="listwise",
                    help="group (train v2): multipos over result groups, candidates with the same 'sig' = one group")
    ap.add_argument("--label-smooth", type=float, default=0.0, help="multipos loss: label smoothing")
    ap.add_argument("--sched", choices=["legacy", "cosine"], default="legacy",
                    help="cosine: linear warmup (--warmup of the steps) then cosine to 0 over the real step count")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--fold", default="", help="train v2: i/n -- train on groups whose stable hash mod n != i, write "
                    "out-of-fold scores for the rest to --out/oof_scores.jsonl")
    ap.add_argument("--fold-by", choices=["db_id"], default="db_id")
    ap.add_argument("--base", default="AlexWortega/openjev")
    ap.add_argument("--subfolder", default="qwen3.5-4b-nli-v5")
    ap.add_argument("--bird-desc", nargs="?", const="1", default="",
                    help="BIRD-train column descriptions in the schema cards (luthor.sq.bird_desc): bare = "
                         "data/raw/bird_train_desc, or a directory. Default off; gen follows the run's args.json")
    grammar_cli.add_argument(ap)
    ap.add_argument("--seed", type=int, default=0, help="data order (all stages' random.Random) and dropout; default 0 = "
                    "the old hard-coded seed, bit-identical")
    ap.add_argument("--init-seed", type=int, default=None, help="torch seed for the LoRA / head init (default: --seed); "
                    "e.g. --seed 1 --init-seed 0 varies only order + dropout, so two seeds can be weight-averaged")
    ap.add_argument("--eval-every", type=float, default=0.0, help="train v2: also score --val-file every this fraction "
                    "of an epoch (0.5 = half-epoch); 0 = epoch ends only")
    ap.add_argument("--avg-last", type=int, default=0, help="train v2 (needs --val-file): keep the checkpoints of the "
                    "last K val evaluations (--out/ckpt/) and write their uniform average (LoRA + head) as --out; the "
                    "best single checkpoint goes to --out/best")
    ap.add_argument("--joint-results", action="store_true", help="train v2 / score: every candidate is scored with a "
                    "table of all distinct live results of its list in the prefix (rerank.joint_texts); score follows "
                    "the validator's settings.json")
    ap.add_argument("--joint-orders", type=int, default=None, help="joint scoring: table orders averaged at score time "
                    "(val and score; default 2 = a seeded permutation and its reverse)")
    a = ap.parse_args()
    bird_desc.enable(a.bird_desc)
    if a.run is None:  # train / score without a run: an explicit --grammar must still be the process's
        a.grammar = grammar_cli.resolve(a.grammar)
    torch.manual_seed(init_seed(a))
    {"gen": gen, "train": train, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
