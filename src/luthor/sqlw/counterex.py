"""Counterexample databases (round 5): small mutated copies of a database on which a query that matches
the gold answer only by coincidence (e.g. COUNT(*) vs COUNT(DISTINCT x), a missing NULL check, a join
that happens not to duplicate rows) returns a different result from the gold query.

Each variant starts from a row sample of the original (at most `cap` rows per table, so large BIRD
databases stay small) and applies one mutation:
  sample  - the sample alone
  dup     - re-insert copies of some rows (new key values where a column is an INTEGER PRIMARY KEY),
            so one parent row gains several child rows
  null    - set some values of nullable, non-key columns to NULL
  shuffle - permute the values of each non-key column (breaks lucky correlations and ordering)
The gold answer may legitimately change between variants; candidates are compared with the gold query
run on the same variant. A finite set of variants cannot prove two queries equivalent.
"""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

MUTATIONS = ("sample", "dup", "null", "shuffle")


def _tables(con) -> list[str]:
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]


def _cols(con, t: str) -> list[tuple]:
    """(name, type, notnull, pk) per column."""
    return [(r[1], (r[2] or "").upper(), bool(r[3]), bool(r[5])) for r in con.execute(f'PRAGMA table_info("{t}")')]


def _fk_cols(con, t: str) -> set[str]:
    return {r[3] for r in con.execute(f'PRAGMA foreign_key_list("{t}")')}


def make_variants(src: str, out_dir: Path, seed: int = 0, cap: int = 400, rate: float = 0.25) -> list[Path]:
    """Write one SQLite file per mutation into out_dir; returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    s.text_factory = lambda b: b.decode("utf-8", "replace")
    ddl = [r[0] for r in s.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL")]
    data = {}
    for t in _tables(s):
        n = s.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        if n <= cap:
            rows = s.execute(f'SELECT * FROM "{t}"').fetchall()
        else:  # deterministic sample: every k-th rowid (tables WITHOUT ROWID: the first rows)
            k = -(-n // cap)
            try:
                rows = s.execute(f'SELECT * FROM "{t}" WHERE rowid % {k} = 0 LIMIT {cap}').fetchall()
            except sqlite3.Error:
                rows = s.execute(f'SELECT * FROM "{t}" LIMIT {cap}').fetchall()
        data[t] = (_cols(s, t), _fk_cols(s, t), [list(r) for r in rows])
    s.close()
    paths = []
    for m in MUTATIONS:
        p = out_dir / f"{m}.sqlite"
        if p.exists():
            p.unlink()
        d = sqlite3.connect(str(p))
        for q in ddl:
            try:
                d.execute(q)
            except sqlite3.Error:
                pass
        for t, (cols, fks, rows) in data.items():
            rows = [r[:] for r in rows]
            names = [c[0] for c in cols]
            keyish = {c[0] for c in cols if c[3]} | fks
            if m == "dup" and rows:
                ipk = [i for i, c in enumerate(cols) if c[3] and c[1] == "INTEGER"]
                extra = []
                top = max((r[ipk[0]] for r in rows if isinstance(r[ipk[0]], int)), default=0) if len(ipk) == 1 else 0
                for r in rnd.sample(rows, max(1, int(len(rows) * rate))):
                    c = r[:]
                    if len(ipk) == 1:
                        top += 1
                        c[ipk[0]] = top
                    extra.append(c)
                rows += extra
            elif m == "null":
                for j, c in enumerate(cols):
                    if c[0] not in keyish and not c[2]:
                        for r in rows:
                            if rnd.random() < rate:
                                r[j] = None
            elif m == "shuffle":
                for j, c in enumerate(cols):
                    if c[0] not in keyish:
                        vals = [r[j] for r in rows]
                        rnd.shuffle(vals)
                        for r, v in zip(rows, vals):
                            r[j] = v
            q = f'INSERT OR IGNORE INTO "{t}" ({", ".join(chr(34) + n + chr(34) for n in names)}) VALUES ({", ".join("?" * len(names))})'
            try:
                d.executemany(q, rows)
            except sqlite3.Error:
                pass
        d.commit()
        d.close()
        paths.append(p)
    return paths
