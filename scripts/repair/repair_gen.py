"""Repair stage for the grammar generator (generator-misses fixes #3-#5, docs/paper_notes/research/generator_misses_2026-10-10.md).

Runs on the GPU box AFTER normal candidate generation (rerank_sq2.py gen) and writes EXTRA grammar candidates for a
pool, in the pool jsonl schema, every one tagged src="repair" (and repair=<variant>). It never rewrites the pool.

Why constrained resampling, not text feedback: the slot model (OmniSQL-7B + LoRA filling grammar slots) reads only
the question, its evidence and the schema card; it was never trained on "your last answer returned 0 rows" text, so
appending feedback to the input would be out of distribution. The decoder, however, is a pure function of per-slot
score vectors (targets.Scores: choice / ptr / pair / vals / limit). So feedback is applied by wrapping the model's
Scores in `ConstrainedScores`, which adds a large negative bias to banned options and a large positive bias to forced
ones, and re-running the lazy top-k decoder (topk.kbest) or temperature sampling on top. The encoder pass is reused;
only cheap head calls and Python decoding are repeated. The same wrapper works for grammar v3 and v4 checkpoints
(v4 only adds slots: m.j<k> join pointers, offset, order2, cond2; bans on m.j<k> are applied as on m.via).

Triggers (per question, from the pool alone, CPU):
  degenerate  every live (running) candidate returns empty / NULL-only / a single 0 or NULL       -> F variants
  all_same    two or more live candidates and every one returns the same result (one sig)       -> F variants
  no_live     no candidate runs                                                                   -> F variants
  evidence    no candidate among the top --ev-top ranks covers every required evidence clause     -> E variants
              (scripts/repair/evidence.py)
Variants (each a kbest(k=--k-rep) on the constrained scores; plans already in the pool are dropped):
  F1 ban_vals      literals the failing candidates compare against are banned in every value slot
  F2 ban_cols      the (table.column) those literals were compared on are banned in filter / condition pointers
  F3 ban_tables    tables every failing candidate joins but none projects are banned in via / require / join slots
  F4 hot           F1+F2 bans, softmax sampling at --hot-temp (--hot-samples draws)
  E1 force         missing evidence columns boosted in filter pointers, their literals in value slots, their op
  E2 force+count   E1 and at least one more filter than the E1 greedy plan has
  E3 arith         missing '/' or constant N: boost '/' expression / compose templates and the class of N in number heads

  # box (GPU): extra candidates for a pool
  python scripts/repair/repair_gen.py gen --run runs/r1r2/r1 --pool POOL.jsonl --plans DIR --db-roots DIR \
      --out POOL.repair_r1.jsonl [--grammar v3|v4] [--skip-ids reserved.json]
  # CPU: which questions trigger, and how many decodes the box will run (GPU-minute estimate)
  python scripts/repair/repair_gen.py plan --pool POOL.jsonl [--dbs data/raw/bird_train/train_databases]
  # CPU: pool + extra -> pool with the extra candidates appended (deduplicated by SQL)
  python scripts/repair/repair_gen.py merge --pool POOL.jsonl --extra A.jsonl [B.jsonl] --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))
if __name__ == "__main__":  # --grammar / the run's args.json -> LUTHOR_GRAMMAR before any luthor.sq import
    from luthor.grammar_cli import preset
    preset()

import evidence as EV  # noqa: E402

BAN = -1.0e4
FILTER_SLOT = re.compile(r"^[mso]\.(filter|c\d|d\d|having)$")
VALUE_SLOT = re.compile(r"^[mso]\.(filter|c\d|d\d|having)(\|t\d)?$")
TABLE_SLOT = re.compile(r"^(m\.via|[mso]\.require|m\.j\d+|m\.join)$")
PROJ_SLOT = re.compile(r"^[mso]\.p\d$")


# ---------------------------------------------------------------------------------------------------- triggers
_FIRST = re.compile(r"first: \((.*)\)$")


def bad_result(c: dict) -> bool:
    """A live candidate whose result is empty, NULL-only, or one row of zeros / NULLs (from the pool's res preview)."""
    if not c.get("runs"):
        return True
    if not c.get("nonnull"):
        return True
    res = c.get("res") or ""
    if res.startswith("1 row "):
        m = _FIRST.search(res)
        if m:
            vals = [v.strip() for v in m.group(1).split(",") if v.strip()]
            return bool(vals) and all(v in ("0", "0.0", "None", "'0'", "'0.0'") for v in vals)
    return False


def pool_state(rec: dict) -> dict:
    cands = [c for c in rec["cands"] if c.get("sql")]
    live = [c for c in cands if c.get("runs")]
    sigs = {c.get("sig") for c in live}
    return {"n": len(cands), "live": len(live), "no_live": not live,
            "degenerate": bool(live) and all(bad_result(c) for c in live),
            "all_same": len(live) >= 2 and len(sigs) == 1}


def triggers(rec: dict, columns=None, ev_top: int = 8) -> list[str]:
    st = pool_state(rec)
    out = [t for t in ("no_live", "degenerate", "all_same") if st[t]]
    atoms = EV.parse_evidence(rec.get("context") or "", columns, rec.get("question", ""))
    if any(a.strict and a.required for a in atoms):
        top = [c for c in rec["cands"] if c.get("sql") and c.get("rank", 0) < ev_top]
        if not any(EV.coverage(atoms, c["sql"])["covers"] for c in top):
            out.append("evidence")
    return out


# ---------------------------------------------------------------------------------------------------- constraints
def _opt_maps(option_ids: list[str]):
    by_q = {o.lower(): i for i, o in enumerate(option_ids)}
    by_col = collections.defaultdict(list)
    by_tab = collections.defaultdict(list)
    for i, o in enumerate(option_ids):
        t, _, c = o.partition(".")
        by_col[c.lower()].append(i)
        by_tab[t.lower()].append(i)
    return by_q, by_col, by_tab


def failing_constraints(rec: dict, option_ids: list[str]) -> dict:
    """Bans from the pool's failing candidates (all live ones for a degenerate / all-same pool; all of them when none
    runs). Returns {"vals": set(norm literal), "cols": set(option idx), "tables": set(option idx)}."""
    cands = [c for c in rec["cands"] if c.get("sql")]
    live = [c for c in cands if c.get("runs")]
    failing = live or cands
    by_q, by_col, by_tab = _opt_maps(option_ids)
    vals, cols, every, proj = set(), set(), None, set()
    for c in failing:
        f = EV.sql_facts(c["sql"])
        for (qc, op, vs) in f.qpreds:
            vals |= {v for v in vs}
            if qc and vs:
                i = by_q.get(qc)
                if i is not None:
                    cols.add(i)
                elif "." not in qc:
                    cols |= {j for j in by_col.get(qc, [])}
        every = set(f.tables) if every is None else every & f.tables
        proj |= f.select_tables
    extra_tabs = (every or set()) - proj
    tabs = {i for t in extra_tabs for i in by_tab.get(t, []) if option_ids[i].endswith(".*")}
    tab_cols = {i for t in extra_tabs for i in by_tab.get(t, [])}
    return {"vals": vals, "cols": cols, "tables": tabs, "table_cols": tab_cols}


def evidence_constraints(atoms: list, missing: list[int], option_ids: list[str]) -> dict:
    """Boosts for the evidence clauses no top candidate implements."""
    _, by_col, _ = _opt_maps(option_ids)
    cols, vals, ops, nums, arith = set(), set(), {}, set(), set()
    n_filter_cols = set()
    for i in missing:
        a = atoms[i]
        if a.kind in ("cmp", "in", "between", "null", "col"):
            idx = by_col.get(a.col or "", [])
            cols |= set(idx)
            if a.kind != "col" and a.col:
                n_filter_cols.add(a.col)
            for j in idx:
                if a.op and a.kind != "col":
                    ops[j] = a.op
            for v in a.values:
                vals.add(v)
                for _, v2 in EV._equiv(v, a.op or ""):
                    vals.add(v2)
                if v.endswith("%") or v.startswith("%"):
                    vals.add(v.strip("%"))
        elif a.kind == "arith":
            arith.add(a.op)
        elif a.kind == "const":
            nums |= set(a.values)
    return {"cols": cols, "vals": vals, "ops": ops, "nums": nums, "arith": arith, "n_new_filters": len(n_filter_cols)}


class ConstrainedScores:
    """targets.Scores with biases: banned options get BAN added, forced ones +boost. Grammar-agnostic: it acts on
    slot names, so v3 and v4 decoders both use it."""

    def __init__(self, base, *, ban_vals=(), ban_ptr: dict | None = None, boost_ptr: dict | None = None,
                 boost_vals=(), boost_ops: dict | None = None, min_choice: dict | None = None,
                 boost_choice: dict | None = None, boost_fn=None, boost_nums=(), boost: float = 30.0):
        self.base, self.option_ids, self.cands = base, base.option_ids, base.cands
        self.ban_vals = {EV.norm_lit(v) for v in ban_vals}
        self.boost_vals = {EV.norm_lit(v) for v in boost_vals}
        self.ban_ptr = ban_ptr or {}        # slot regex -> set(option idx)
        self.boost_ptr = boost_ptr or {}    # slot regex -> set(option idx)
        self.boost_ops = boost_ops or {}    # option idx -> op name (targets OPS)
        self.min_choice = min_choice or {}  # choice key -> minimum class index
        self.boost_choice = boost_choice or {}  # choice key -> set(class idx)
        self.boost_fn = boost_fn            # callable(template str) -> bool, for pair head "fn"
        self.boost_nums = {EV.norm_lit(v) for v in boost_nums}
        self.b = boost
        self._cand_keys = [EV.norm_lit(c.value) for c in self.cands]

    @staticmethod
    def _match(table: dict, slot: str) -> set:
        out = set()
        for rx, s in table.items():
            if re.match(rx, slot):
                out |= s
        return out

    def choice(self, key):
        xs = list(self.base.choice(key))
        lo = self.min_choice.get(key)
        if lo:
            for i in range(min(lo, len(xs))):
                xs[i] += BAN
        for i in self.boost_choice.get(key, ()):
            if i < len(xs):
                xs[i] += self.b
        if self.boost_nums and re.match(r"^m\.cn\d$", key):
            self._nums(xs)
        return xs

    def _nums(self, xs):
        from luthor.sq2.targets import VOCAB
        names = ["-"] + [EV.norm_lit(v) for v in VOCAB["nums"]]
        for i, n in enumerate(names[:len(xs)]):
            if n in self.boost_nums:
                xs[i] += self.b

    def ptr(self, slot):
        xs = list(self.base.ptr(slot))
        for i in self._match(self.ban_ptr, slot):
            if i < len(xs):
                xs[i] += BAN
        for i in self._match(self.boost_ptr, slot):
            if i < len(xs):
                xs[i] += self.b
        return xs

    def pair(self, slot, opt, head):
        xs = list(self.base.pair(slot, opt, head))
        if head == "op" and opt in self.boost_ops:
            from luthor.sq2.targets import OPS
            op = self.boost_ops[opt]
            if op in OPS and OPS.index(op) < len(xs):
                xs[OPS.index(op)] += self.b
        elif head == "fn" and self.boost_fn is not None and PROJ_SLOT.match(slot):
            from luthor.sq2.targets import PAIRS
            for i, t in enumerate(PAIRS.get("fn", ())[:len(xs)]):
                if t and self.boost_fn(t):
                    xs[i] += self.b
        elif re.fullmatch(r"n\d", head) and self.boost_nums:
            self._nums(xs)
        return xs

    def vals(self, slot, opt):
        xs = list(self.base.vals(slot, opt))
        if VALUE_SLOT.match(slot) or slot.endswith("|t0") or slot.endswith("|t1"):
            for i, k in enumerate(self._cand_keys[:len(xs)]):
                if k in self.ban_vals:
                    xs[i] += BAN
                elif k in self.boost_vals:
                    xs[i] += self.b
        return xs

    def limit(self, slot):
        return self.base.limit(slot)


def _compose_classes(want_div: bool, want_pct: bool) -> set:
    from luthor.sq2.targets import MAIN_CHOICES
    out = set()
    for i, t in enumerate(MAIN_CHOICES.get("compose", ())):
        if (want_pct and t == "percent") or (want_div and (t == "ratio" or ("/" in t and "{P" in t))):
            out.add(i)
    return out


def variants(rec: dict, trig: list[str], sc, atoms=None, missing=None, *, boost: float = 30.0) -> list[tuple[str, object]]:
    """(name, constrained Scores) per repair variant this question gets. Sampling variants are named 'F4:hot'."""
    out = []
    ids = sc.option_ids
    if {"degenerate", "all_same", "no_live"} & set(trig):
        fc = failing_constraints(rec, ids)
        out.append(("F1:ban_vals", ConstrainedScores(sc, ban_vals=fc["vals"], boost=boost)))
        if fc["cols"]:
            out.append(("F2:ban_cols", ConstrainedScores(sc, ban_ptr={FILTER_SLOT.pattern: fc["cols"]}, boost=boost)))
        if fc["tables"] or fc["table_cols"]:
            out.append(("F3:ban_tables", ConstrainedScores(sc, ban_ptr={TABLE_SLOT.pattern: fc["tables"] | fc["table_cols"],
                                                                        FILTER_SLOT.pattern: fc["table_cols"]}, boost=boost)))
        out.append(("F4:hot", ConstrainedScores(sc, ban_vals=fc["vals"], ban_ptr={FILTER_SLOT.pattern: fc["cols"]},
                                                boost=boost)))
    if "evidence" in trig and atoms is not None and missing:
        ec = evidence_constraints(atoms, missing, ids)
        if ec["cols"] or ec["vals"]:
            kw = dict(boost_ptr={FILTER_SLOT.pattern: ec["cols"]}, boost_vals=ec["vals"], boost_ops=ec["ops"], boost=boost)
            out.append(("E1:force", ConstrainedScores(sc, **kw)))
            if ec["n_new_filters"]:
                out.append(("E2:force+count", (ConstrainedScores, kw, ec["n_new_filters"])))  # needs E1's greedy plan
        if ec["arith"] or ec["nums"]:
            div = "/" in ec["arith"] or bool(ec["nums"])
            nums = ec["nums"]

            def fn_ok(t, div=div, nums=nums):
                return ("/" in t if div else True) and ("{N" in t if nums else True)
            out.append(("E3:arith", ConstrainedScores(
                sc, boost_fn=fn_ok, boost_nums=nums, boost=boost,
                boost_choice={"m.compose": _compose_classes(div, "100" in nums)})))
    return out


def repair_plans(rec: dict, trig: list[str], sc, compile_fn, *, k_rep: int = 8, hot_samples: int = 16,
                 hot_temp: float = 1.5, atoms=None, missing=None, boost: float = 30.0, seed: int = 0,
                 max_extra: int = 24) -> list[tuple[str, str, float]]:
    """(variant, sql, cost) of new distinct plans, SQL not already in the pool, at most max_extra."""
    from luthor.sq2.targets import decode
    from luthor.sq2.topk import _Steer, kbest
    have = {c["sql"] for c in rec["cands"] if c.get("sql")}
    out = []

    def keep(name, plans):
        for p, cost in plans:
            if len(out) >= max_extra:
                return
            try:
                sql = compile_fn(p)
            except Exception:
                continue
            if sql and sql not in have:
                have.add(sql)
                out.append((name, sql, round(float(cost), 4)))

    for name, csc in variants(rec, trig, sc, atoms, missing, boost=boost):
        if len(out) >= max_extra:
            break
        try:
            if name == "F4:hot":
                rnd = random.Random(seed)
                cache: dict = {}
                plans = []
                for _ in range(hot_samples):
                    st = _Steer(csc, cache, rnd=rnd, temp=hot_temp)
                    try:
                        plans.append((decode(st), 100.0 + st.cost))
                    except Exception:
                        continue
                keep(name, plans)
            elif name == "E2:force+count":
                cls, kw, n_new = csc
                g = decode(cls(sc, **kw))
                from luthor.sq2.targets import CHOICES
                lo = min(len(g.filters) + 1, len(CHOICES["n_filter"]) - 1)
                keep(name, kbest(cls(sc, **kw, min_choice={"m.n_filter": lo}), k_rep, compile_fn, with_cost=True))
            else:
                keep(name, kbest(csc, k_rep, compile_fn, with_cost=True))
        except Exception as ex:  # one variant failing never stops the others
            print(f"  {rec['id']} {name}: {type(ex).__name__}: {ex}", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- box: gen
def _load_ids(p):
    if not p:
        return set()
    d = json.load(open(EV.guard(p)))
    return set(d["ids"] if isinstance(d, dict) else d)


def gen(a):
    import sqlite3

    import torch
    import rerank_sq2 as R
    from train_sq2 import load_sets
    from luthor.sq.candidates import ValueIndex, find_candidates
    from luthor.sq.plan import compile_plan
    from luthor.sq2.model import TorchScores
    from luthor.sq2.train import encode_db, question_body, select_cands, serve_chunks

    R.use_run_settings(a)  # grammar check (v3/v4 from args.json), backbone, vocab.json, candidate order
    pool = [json.loads(l) for l in open(EV.guard(a.pool))]
    skip = _load_ids(a.skip_ids)
    keep = set().union(*(_load_ids(p) for p in a.ids)) if a.ids else None
    paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, str(p))
    todo = {}
    for r in pool:
        if r["id"] in skip or (keep is not None and r["id"] not in keep) or r["db_id"] not in paths:
            continue
        cols = set(EV.db_columns_path(paths[r["db_id"]]))
        t = triggers(r, cols, a.ev_top)
        if a.only:
            t = [x for x in t if x in a.only]
        if t:
            todo[r["id"]] = (r, t)
    print(f"repair: {len(todo)} of {len(pool)} questions trigger "
          f"{dict(collections.Counter(x for _, t in todo.values() for x in t))}", flush=True)
    if a.max_q:
        todo = dict(list(todo.items())[:a.max_q])
    cards, exs, gold = load_sets(a.plans, strict=False, all_rows=True)
    exs = [e for e in exs if e.id in todo]
    missing_ids = set(todo) - {e.id for e in exs}
    if missing_ids:
        print(f"  {len(missing_ids)} triggered questions have no row in --plans (skipped)", flush=True)
    by_db = collections.defaultdict(list)
    for e in exs:
        by_db[e.db_id].append(e)
    m = R.load_slot_model(a.run, a.base, a.subfolder)
    recs, tasks, t0 = [], [], time.time()
    vc = collections.Counter()
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
            cols = set(EV.db_columns_path(paths[db]))
            qs = []
            for tabs, chunk in serve_chunks(card, group, 8):
                st, oix, enc = encode_db(m, card, chunk, tabs)
                for e, x in zip(chunk, enc):
                    rec, trig = todo[e.id]
                    sc = TorchScores(m, x, st.option_ids, e.cands)
                    atoms = EV.parse_evidence(rec.get("context") or "", cols, rec.get("question", ""))
                    missing = _missing_in_top(atoms, rec, a.ev_top)
                    got = repair_plans(rec, trig, sc, lambda p: compile_plan(p, g), k_rep=a.k_rep,
                                       hot_samples=a.hot_samples, hot_temp=a.hot_temp, atoms=atoms, missing=missing,
                                       boost=a.boost, seed=zlib.crc32(e.id.encode()), max_extra=a.max_extra)
                    vc.update(v for v, _, _ in got)
                    cl = [{"rank": i, "cost": cost, "sql": sql, "src": "repair", "repair": v}
                          for i, (v, sql, cost) in enumerate(got)]
                    recs.append({k: rec[k] for k in ("id", "db_id", "question", "context", "gold_sql", "in_slots", "covered")
                                 if k in rec} | {"triggers": trig, "cands": cl})
                    qs.append((rec["gold_sql"], [c["sql"] for c in cl]))
            tasks.append((paths[db], qs, a.seconds))
            if n_db % 20 == 0:
                print(f"  repaired {len(recs)} questions, {n_db + 1}/{len(by_db)} dbs, {time.time() - t0:.0f}s", flush=True)
    del m
    torch.cuda.empty_cache()
    from multiprocessing import Pool
    i = 0
    with Pool(a.workers) as pool_:
        for res in pool_.imap(R._exec_db, tasks):
            for r in res:
                rec = recs[i]
                i += 1
                if r is None:
                    rec["gold_error"] = True
                    continue
                for c, (runs, nonnull, ok, sig, prev) in zip(rec["cands"], r):
                    c.update(runs=runs, nonnull=nonnull, ok=ok, sig=sig, res=prev)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    gained = sum(1 for r in recs if any(c.get("ok") for c in r["cands"]))
    lab = "" if a.blind else f"; questions with an ok extra: {gained} (labels only, not a selection score)"
    print(f"REPAIR {len(recs)} questions, {sum(len(r['cands']) for r in recs)} extra candidates, variants {dict(vc)}"
          f"{lab} in {(time.time() - t0) / 60:.1f} min", flush=True)


def _missing_in_top(atoms, rec, ev_top):
    """Required strict atoms that NO top candidate covers (the clauses to force)."""
    strict = [i for i, a in enumerate(atoms) if a.strict and a.required]
    top = [c for c in rec["cands"] if c.get("sql") and c.get("rank", 0) < ev_top]
    covs = [EV.coverage(atoms, c["sql"])["covered"] for c in top]
    return [i for i in strict if not any(cv[i] for cv in covs)]


# ---------------------------------------------------------------------------------------------------- CPU: plan / merge
def plan(a):
    pool = [json.loads(l) for l in open(EV.guard(a.pool))]
    skip = _load_ids(a.skip_ids)
    keep = _load_ids(a.ids) if a.ids else None
    c, n, q_ok, trig_ok = collections.Counter(), 0, collections.Counter(), collections.Counter()
    n_var = 0
    for r in pool:
        if r["id"] in skip or (keep is not None and r["id"] not in keep):
            continue
        n += 1
        cols = set(EV.db_columns(str(a.dbs), r["db_id"])) if a.dbs else None
        t = triggers(r, cols, a.ev_top)
        if a.only:
            t = [x for x in t if x in a.only]
        has_ok = any(x.get("ok") for x in r["cands"])
        for x in t:
            c[x] += 1
            trig_ok[x] += has_ok
        if t:
            c["any"] += 1
            trig_ok["any"] += has_ok
            n_var += (4 if {"degenerate", "all_same", "no_live"} & set(t) else 0) + (3 if "evidence" in t else 0)
    out = {"questions": n, "triggered": dict(c), "triggered_with_ok_already": dict(trig_ok),
           "variant_runs_upper_bound": n_var,
           "decodes_upper_bound": n_var * max(a.k_rep * 8, a.hot_samples)}
    print(json.dumps(out, indent=1))
    return out


def merge(a):
    extra, extra_rows = {}, {}
    for p in a.extra:
        for line in open(EV.guard(p)):
            r = json.loads(line)
            extra.setdefault(r["id"], []).extend(r["cands"])
            extra_rows.setdefault(r["id"], r)
    n_add, n_q, seen_ids = 0, 0, set()
    with open(a.out, "w") as f:
        for line in open(EV.guard(a.pool)):
            r = json.loads(line)
            seen_ids.add(r["id"])
            cl = list(r["cands"])
            seen = {c.get("sql") for c in cl}
            add = 0
            for c in extra.get(r["id"], []):
                if c.get("sql") and c["sql"] not in seen:
                    seen.add(c["sql"])
                    cl.append(dict(c, rank=len(cl), src=a.src))
                    add += 1
            n_add += add
            n_q += add > 0
            f.write(json.dumps(dict(r, cands=cl)) + "\n")
        n_new = 0
        if a.add_missing:  # questions only the extra files have (e.g. a gold that timed out in one run): kept whole
            for i, r in extra_rows.items():
                if i in seen_ids:
                    continue
                cl, seen = [], set()
                for c in extra[i]:
                    if c.get("sql") and c["sql"] not in seen:
                        seen.add(c["sql"])
                        cl.append(dict(c, rank=len(cl), src=a.src))
                f.write(json.dumps(dict(r, cands=cl)) + "\n")
                n_new += 1
    print(f"merged {n_add} {a.src} candidates into {n_q} questions"
          + (f", {n_new} questions only in --extra added whole" if a.add_missing else "") + f" -> {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--run", type=Path, required=True, help="slot-model checkpoint dir (args.json, vocab.json, best_lora)")
    g.add_argument("--pool", type=Path, required=True, help="candidate pool (rerank_sq2 gen / union_pools format)")
    g.add_argument("--plans", nargs="+", type=Path, required=True, help="plan dirs holding the pool's questions + cards")
    g.add_argument("--db-roots", nargs="+", required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--skip-ids", type=Path, help="ids never repaired (e.g. RESERVED)")
    g.add_argument("--ids", nargs="+", type=Path,
                   help="repair only these ids (json lists or {ids: [...]}; several files = union, e.g. DECISION RESERVED)")
    g.add_argument("--blind", action="store_true", help="print no label-derived count (pools holding RESERVED rows)")
    g.add_argument("--only", nargs="*", choices=["no_live", "degenerate", "all_same", "evidence"])
    g.add_argument("--max-q", type=int, default=0)
    g.add_argument("--ev-top", type=int, default=8)
    g.add_argument("--k-rep", type=int, default=8, help="lazy k-best size per variant")
    g.add_argument("--hot-samples", type=int, default=16)
    g.add_argument("--hot-temp", type=float, default=1.5)
    g.add_argument("--boost", type=float, default=30.0)
    g.add_argument("--max-extra", type=int, default=24, help="extra candidates per question at most")
    g.add_argument("--seconds", type=float, default=3.0)
    g.add_argument("--workers", type=int, default=12)
    g.add_argument("--base", default="seeklhy/OmniSQL-7B")
    g.add_argument("--subfolder", default="")
    from luthor import grammar_cli
    grammar_cli.add_argument(g)
    p = sub.add_parser("plan")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--dbs", type=Path)
    p.add_argument("--ids", type=Path)
    p.add_argument("--skip-ids", type=Path)
    p.add_argument("--only", nargs="*", choices=["no_live", "degenerate", "all_same", "evidence"])
    p.add_argument("--ev-top", type=int, default=8)
    p.add_argument("--k-rep", type=int, default=8)
    p.add_argument("--hot-samples", type=int, default=16)
    mg = sub.add_parser("merge")
    mg.add_argument("--pool", type=Path, required=True)
    mg.add_argument("--extra", nargs="+", type=Path, required=True)
    mg.add_argument("--out", type=Path, required=True)
    mg.add_argument("--src", default="repair", help="src tag of appended candidates")
    mg.add_argument("--add-missing", action="store_true", help="also write questions that only --extra has")
    a = ap.parse_args(argv)
    {"gen": gen, "plan": plan, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
