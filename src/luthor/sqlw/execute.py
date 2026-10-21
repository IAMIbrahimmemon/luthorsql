"""Run SQL on read-only SQLite with a time limit; compare result sets with the gold result."""
from __future__ import annotations

import hashlib
import sqlite3
import time


def connect(path: str, seconds: float = 3.0):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)

    def rows(sql: str):
        box["t"] = time.time() + seconds
        return con.execute(sql).fetchmany(100_000)
    return rows


def signature(rows) -> str:
    """Order-free fingerprint of a result set (for voting: same answer -> same signature)."""
    return hashlib.md5(repr(sorted(set(map(lambda r: tuple(map(str, r)), rows)))).encode()).hexdigest()[:16]


def check_db(task):
    """(db path, [(gold_sql, [sql | None])], seconds) -> per question None (gold fails) or
    [(runs, nonnull, ok, signature)] per candidate."""
    path, qs, seconds = task
    run = connect(path, seconds)
    out = []
    for gold_sql, sqls in qs:
        try:
            ref = set(map(tuple, run(gold_sql)))
        except Exception:
            out.append(None)
            continue
        res, seen = [], {}
        for s in sqls:
            if not s:
                res.append((False, False, False, ""))
            elif s in seen:
                res.append(seen[s])
            else:
                try:
                    got = run(s)
                    r = (True, any(v is not None for x in got for v in x), set(map(tuple, got)) == ref, signature(got))
                except Exception:
                    r = (False, False, False, "")
                seen[s] = r
                res.append(r)
        out.append(res)
    return out
