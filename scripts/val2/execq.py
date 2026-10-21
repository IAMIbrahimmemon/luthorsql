"""Execution helpers for the val2 build: exactly the gen-time labelling of scripts/rerank_sq2.py (_exec_db), so a
re-executed candidate gets the same runs / nonnull / ok / sig / res it would have got at generation time.

Copied (not imported) because rerank_sq2.py imports torch at module level; keep the two in sync.
  sig  md5 of the sorted reprs of the result as a SET of row tuples, first 12 hex chars
  ok   result set == gold result set (raw Python tuple equality, so 1 == 1.0, as at gen time)
"""
from __future__ import annotations

import hashlib
import sqlite3
import time


def connect(path: str, seconds: float):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)

    def rows(sql):
        box["t"] = time.time() + seconds
        return con.execute(sql).fetchmany(100_000)
    return rows


def result_preview(rows: list) -> str:
    """Copy of luthor.sq2.rerank.result_preview (that module imports torch)."""
    if not rows:
        return "0 rows"
    first = rows[0]
    kinds = ", ".join("null" if v is None else "integer" if isinstance(v, int) else "real" if isinstance(v, float)
                      else "text" for v in first)
    shown = "(" + ", ".join(repr(v)[:40] for v in first) + ")"
    return f"{len(rows)} row{'s' if len(rows) != 1 else ''} x {len(first)} col{'s' if len(first) != 1 else ''} ({kinds}); first: {shown}"


def sig_of(got) -> str:
    gs = set(map(tuple, got))
    return hashlib.md5(repr(sorted(map(repr, gs))).encode()).hexdigest()[:12]


def exec_questions(task):
    """(db path, [(key, gold_sql, [sql | None])], gold_seconds, cand_seconds) ->
    [(key, gold_status, gold_sig, [(runs, nonnull, ok, sig, res)])].
    gold_status: 'ok' | 'empty' | 'allnull' | 'timeout' | 'error'. Candidates are only run when the gold is 'ok'."""
    path, qs, gsec, csec = task
    try:
        g_rows = connect(path, gsec)
        c_rows = connect(path, csec)
    except Exception:
        return [(k, "error", None, []) for k, _, _ in qs]
    out = []
    for key, gold_sql, sqls in qs:
        try:
            got = g_rows(gold_sql)
            ref = set(map(tuple, got))
            if not got:
                status = "empty"
            elif not any(v is not None for x in got for v in x):
                status = "allnull"
            else:
                status = "ok"
            gsig = sig_of(got)
        except Exception as e:
            out.append((key, "timeout" if "interrupted" in str(e) else "error", None, []))
            continue
        if status != "ok":
            out.append((key, status, gsig, []))
            continue
        res, seen = [], {}
        for s in sqls:
            if s is None:
                res.append((False, False, False, None, None))
            elif s in seen:
                res.append(seen[s])
            else:
                try:
                    got = c_rows(s)
                    gs = set(map(tuple, got))
                    r = (True, any(v is not None for x in got for v in x), gs == ref, sig_of(got), result_preview(got))
                except Exception:
                    r = (False, False, False, None, None)
                seen[s] = r
                res.append(r)
        out.append((key, status, gsig, res))
    return out


def run_tasks(by_db: dict, paths: dict, gold_seconds: float, cand_seconds: float, workers: int = 6,
              chunk: int = 20) -> dict:
    """{db_id: [(key, gold_sql, [sql])]} -> {key: (gold_status, gold_sig, [(runs, nonnull, ok, sig, res)])}."""
    tasks = []
    out = {}
    for db, lst in by_db.items():
        if db not in paths:
            out.update({k: ("no-db", None, []) for k, _, _ in lst})
            continue
        tasks += [(paths[db], lst[i:i + chunk], gold_seconds, cand_seconds) for i in range(0, len(lst), chunk)]
    if workers > 1 and len(tasks) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as p:
            for part in p.imap_unordered(exec_questions, tasks):
                out.update({k: (st, sg, r) for k, st, sg, r in part})
    else:
        for t in tasks:
            out.update({k: (st, sg, r) for k, st, sg, r in exec_questions(t)})
    return out
