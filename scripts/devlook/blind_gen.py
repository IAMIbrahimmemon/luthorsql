"""BIRD dev look #1, variant B, ON the box: R1's k=64 distinct lists and the repair candidates, BLIND (no gold).

The box never holds a BIRD dev gold query or label: the questions file (prep.py questions) has id / db_id / question /
context only, and execution records runs / nonnull / sig / res (what the validator and the repair triggers read),
never ok. Nothing label-derived exists here, so nothing label-derived can be printed.

  gen     same decode as rerank_sq2.py gen (load_sets on the --plans dirs, value index, serve_chunks of 8, lazy k-best
          --distinct, compile, execute), minus the gold: the per-candidate execution is rerank_sq2._exec_db without the
          reference query (same connection, time limit, row cap, sig and preview).
  repair  same constrained re-decoding as scripts/repair/repair_gen.py gen (its triggers / variants / repair_plans are
          imported unchanged: banned -1e4, forced +30, k-rep 8, hot 16 @ T=1.5, at most 24 extra), minus the gold.
          repair_gen.py itself refuses BIRD dev paths (evidence.guard) and executes against gold, so its gen loop is
          copied here instead of called.
  merge   pool + extra candidates (dedup by SQL, extras appended with src), like repair_gen.py merge.

  python scripts/devlook/blind_gen.py gen --run runs/r1r2/r1 --plans QS --db-roots DBS --k 64 --out k64_gen.jsonl
  python scripts/devlook/blind_gen.py repair --run runs/r1r2/r1 --pool k64_pool.jsonl --plans QS --db-roots DBS \
      --out repair_extra.jsonl
  python scripts/devlook/blind_gen.py merge --pool A.jsonl --extra B.jsonl --src k32 --out C.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

_LOG = logging.getLogger("luthorsql")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "repair"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))
if __name__ == "__main__":  # LUTHOR_GRAMMAR before the first luthor.sq import (workers inherit it)
    from luthor.grammar_cli import preset
    preset()

# fields a pool row / candidate may carry on the box; anything else (ok, gold_sql, covered, in_slots, rr, ...) is dropped
Q_KEYS = ("id", "db_id", "question", "context")
C_KEYS = ("rank", "cost", "sql", "runs", "nonnull", "sig", "res", "src", "repair")
LABEL_KEYS = ("ok", "gold_sql", "gold_error", "covered", "in_slots", "compiled_sql", "plan", "SQL", "gold")


def assert_blind(rec: dict) -> None:
    bad = [k for k in LABEL_KEYS if k in rec] + [k for c in rec.get("cands", []) for k in LABEL_KEYS if k in c]
    if bad:
        raise SystemExit(f"DEVLOOK_FAIL label field(s) {sorted(set(bad))} in row {rec.get('id')}: refusing (box is blind)")


def _exec_blind(task):
    """(db path, [[sql | None]], seconds) -> per question, per candidate (runs, nonnull, sig, preview).
    rerank_sq2._exec_db without the reference query: same connection, time limit, row cap and fingerprints."""
    import rerank_sq2 as R
    path, qs, seconds = task
    rows = R._connect(path, seconds)
    out = []
    for sqls in qs:
        res, seen = [], {}
        for s in sqls:
            if s is None:
                res.append((False, False, None, None))
            elif s in seen:
                res.append(seen[s])
            else:
                try:
                    got = rows(s)
                    gs = set(map(tuple, got))
                    sig = hashlib.md5(repr(sorted(map(repr, gs))).encode()).hexdigest()[:12]
                    r = (True, any(v is not None for x in got for v in x), sig, R.result_preview(got))
                except Exception:
                    _LOG.debug("candidate failed to run", exc_info=True)
                    r = (False, False, None, None)
                seen[s] = r
                res.append(r)
        out.append(res)
    return out


def _paths(roots) -> dict:
    paths = {}
    for root in roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, str(p))
    return paths


def _load(a, ids=None):
    """Questions as rerank_sq2 gen loads them (load_sets, strict=False, every row); the --plans rows hold no gold."""
    from train_sq2 import load_sets
    cards, exs, rows = load_sets(a.plans, strict=False, all_rows=True)
    for r in rows.values():
        assert_blind(r)
    exs = [e for e in exs if not rows[e.id].get("twin_of") and (ids is None or e.id in ids)]
    return cards, exs, rows


def _value_cands(a, card, group, path):
    import sqlite3
    from luthor.sq.candidates import ValueIndex, find_candidates
    from luthor.sq2.train import question_body, select_cands
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    try:
        vindex = ValueIndex.build(con, card)
        for e in group:
            body = question_body({"question": e.question, "context": e.context})
            e.cands, _ = select_cands(find_candidates(body, card, extra=vindex.match(body)), order=a.cand_order, body=body)
    finally:
        con.close()


def _execute(a, recs, tasks):
    i = 0
    with Pool(a.workers) as pool:
        for res in pool.imap(_exec_blind, tasks):
            for r in res:
                rec = recs[i]
                i += 1
                for c, (runs, nonnull, sig, prev) in zip(rec["cands"], r):
                    c.update(runs=runs, nonnull=nonnull, sig=sig, res=prev)
    assert i == len(recs), (i, len(recs))


def _write(path: Path, recs):
    for r in recs:
        assert_blind(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def gen(a):
    import torch
    import rerank_sq2 as R
    from luthor.sq.plan import compile_plan
    from luthor.sq2.model import TorchScores
    from luthor.sq2.topk import candidates
    from luthor.sq2.train import encode_db, serve_chunks
    R.use_run_settings(a)
    cards, exs, rows = _load(a)
    paths = _paths(a.db_roots)
    exs = [e for e in exs if e.db_id in paths]
    by_db = collections.defaultdict(list)
    for e in exs:
        by_db[e.db_id].append(e)
    print(f"gen blind: {len(exs)} questions on {len(by_db)} databases, k={a.k} distinct={a.distinct}", flush=True)
    m = R.load_slot_model(a.run, a.base, a.subfolder)
    recs, tasks, t0 = [], [], time.time()
    with torch.no_grad():
        for n_db, (db, group) in enumerate(by_db.items()):
            card = cards[db]
            g = card.graph()
            try:
                _value_cands(a, card, group, paths[db])
            except Exception as ex:
                print(f"value index failed on {db}: {ex}", flush=True)
            qs = []
            for tabs, chunk in serve_chunks(card, group, 8):
                st, oix, enc = encode_db(m, card, chunk, tabs)
                for e, x in zip(chunk, enc):
                    sc = TorchScores(m, x, st.option_ids, e.cands)
                    try:
                        pc = candidates(sc, k=a.k, with_cost=True, wide_singles=0, wide_pairs=0, runner_up2=0,
                                        samples=0, temp=1.0, seed=zlib.crc32(e.id.encode()), distinct=a.distinct,
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
                        cl.append({"rank": rank, "cost": round(cost, 4), "sql": sql, "src": a.src})
                    r = rows[e.id]
                    recs.append({"id": e.id, "db_id": db, "question": e.question, "context": r.get("context", ""),
                                 "cands": cl})
                    qs.append([c["sql"] for c in cl])
            tasks.append((paths[db], qs, a.seconds))
            print(f"  decoded {len(recs)} questions, {n_db + 1}/{len(by_db)} dbs, {time.time() - t0:.0f}s", flush=True)
    del m
    torch.cuda.empty_cache()
    print(f"executing on {a.workers} workers", flush=True)
    _execute(a, recs, tasks)
    _write(a.out, recs)
    nc = sum(len(r["cands"]) for r in recs)
    print(f"gen blind done: {len(recs)} questions, {nc} candidates ({nc / max(len(recs), 1):.1f} per question) "
          f"in {(time.time() - t0) / 60:.1f} min -> {a.out}", flush=True)


def repair(a):
    import torch
    import repair_gen as RG
    import rerank_sq2 as R
    from luthor.sq.plan import compile_plan
    from luthor.sq2.model import TorchScores
    from luthor.sq2.train import encode_db, serve_chunks
    EV = RG.EV
    R.use_run_settings(a)
    pool = [json.loads(l) for l in open(a.pool)]
    for r in pool:
        assert_blind(r)
    paths = _paths(a.db_roots)
    todo = {}
    for r in pool:
        if r["db_id"] not in paths:
            continue
        t = RG.triggers(r, set(EV.db_columns_path(paths[r["db_id"]])), a.ev_top)
        if t:
            todo[r["id"]] = (r, t)
    print(f"repair blind: {len(todo)} of {len(pool)} questions trigger "
          f"{dict(collections.Counter(x for _, t in todo.values() for x in t))}", flush=True)
    cards, exs, rows = _load(a, set(todo))
    missing = set(todo) - {e.id for e in exs}
    if missing:
        print(f"  {len(missing)} triggered questions have no row in --plans (skipped)", flush=True)
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
                _value_cands(a, card, group, paths[db])
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
                    miss = RG._missing_in_top(atoms, rec, a.ev_top)
                    got = RG.repair_plans(rec, trig, sc, lambda p: compile_plan(p, g), k_rep=a.k_rep,
                                          hot_samples=a.hot_samples, hot_temp=a.hot_temp, atoms=atoms, missing=miss,
                                          boost=a.boost, seed=zlib.crc32(e.id.encode()), max_extra=a.max_extra)
                    vc.update(v for v, _, _ in got)
                    cl = [{"rank": i, "cost": cost, "sql": sql, "src": "repair", "repair": v}
                          for i, (v, sql, cost) in enumerate(got)]
                    recs.append({k: rec[k] for k in Q_KEYS} | {"triggers": trig, "cands": cl})
                    qs.append([c["sql"] for c in cl])
            tasks.append((paths[db], qs, a.seconds))
            print(f"  repaired {len(recs)} questions, {n_db + 1}/{len(by_db)} dbs, {time.time() - t0:.0f}s", flush=True)
    del m
    torch.cuda.empty_cache()
    _execute(a, recs, tasks)
    _write(a.out, recs)
    print(f"repair blind done: {len(recs)} questions, {sum(len(r['cands']) for r in recs)} extra candidates, "
          f"variants {dict(vc)} in {(time.time() - t0) / 60:.1f} min -> {a.out}", flush=True)


def merge(a):
    """pool + extras (dedup by SQL; appended with rank = position and src = --src); --add-missing also writes questions
    only the extras have. Only Q_KEYS / C_KEYS travel."""
    extra, extra_rows = collections.defaultdict(list), {}
    for p in a.extra:
        for line in open(p):
            r = json.loads(line)
            assert_blind(r)
            extra[r["id"]].extend(r["cands"])
            extra_rows.setdefault(r["id"], r)
    n_add = n_q = n_new = 0
    seen_ids, out = set(), []
    clean = lambda c: {k: c[k] for k in C_KEYS if k in c}  # noqa: E731
    for line in open(a.pool):
        r = json.loads(line)
        assert_blind(r)
        seen_ids.add(r["id"])
        cl = [clean(c) for c in r["cands"]]
        seen = {c.get("sql") for c in cl}
        add = 0
        for c in extra.get(r["id"], []):
            if c.get("sql") and c["sql"] not in seen:
                seen.add(c["sql"])
                cl.append(dict(clean(c), rank=len(cl), src=a.src))
                add += 1
        n_add += add
        n_q += add > 0
        out.append({k: r[k] for k in Q_KEYS} | {"cands": cl})
    if a.add_missing:
        for i, r in extra_rows.items():
            if i in seen_ids:
                continue
            cl, seen = [], set()
            for c in extra[i]:
                if c.get("sql") and c["sql"] not in seen:
                    seen.add(c["sql"])
                    cl.append(dict(clean(c), rank=len(cl), src=a.src))
            out.append({k: r[k] for k in Q_KEYS} | {"cands": cl})
            n_new += 1
    _write(a.out, out)
    print(f"merged {n_add} {a.src} candidates into {n_q} questions"
          + (f", {n_new} questions only in --extra added whole" if a.add_missing else "")
          + f"; {len(out)} questions, {sum(len(r['cands']) for r in out)} candidates -> {a.out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "repair"):
        g = sub.add_parser(name)
        g.add_argument("--run", type=Path, required=True)
        g.add_argument("--plans", nargs="+", type=Path, required=True, help="dirs with cards/ + plans.jsonl (no gold)")
        g.add_argument("--db-roots", nargs="+", required=True)
        g.add_argument("--out", type=Path, required=True)
        g.add_argument("--seconds", type=float, default=3.0)
        g.add_argument("--workers", type=int, default=12)
        g.add_argument("--base", default="seeklhy/OmniSQL-7B")
        g.add_argument("--subfolder", default="")
        from luthor import grammar_cli
        grammar_cli.add_argument(g)
        if name == "gen":
            g.add_argument("--k", type=int, default=64)
            g.add_argument("--distinct", action="store_true")
            g.add_argument("--src", default="k64")
        else:
            g.add_argument("--pool", type=Path, required=True)
            g.add_argument("--ev-top", type=int, default=8)
            g.add_argument("--k-rep", type=int, default=8)
            g.add_argument("--hot-samples", type=int, default=16)
            g.add_argument("--hot-temp", type=float, default=1.5)
            g.add_argument("--boost", type=float, default=30.0)
            g.add_argument("--max-extra", type=int, default=24)
    mg = sub.add_parser("merge")
    mg.add_argument("--pool", type=Path, required=True)
    mg.add_argument("--extra", nargs="+", type=Path, required=True)
    mg.add_argument("--out", type=Path, required=True)
    mg.add_argument("--src", required=True)
    mg.add_argument("--add-missing", action="store_true")
    a = ap.parse_args(argv)
    {"gen": gen, "repair": repair, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
