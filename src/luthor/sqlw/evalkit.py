"""Round-5 evaluation helpers: frozen dev list, scoring against the original gold, pick rules that do
not throw away empty answers, and paired comparisons with database-resampled ranges.

Correctness follows BIRD's execution match: the candidate's result set equals the gold result set
(rows compared as tuples, order and duplicates ignored).
"""
from __future__ import annotations

import collections
import random
from pathlib import Path

from .execute import connect, signature


def db_paths(roots) -> dict[str, str]:
    paths = {}
    for root in roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, str(p))
    return paths


def run_gold(task):
    """(db path, [gold_sql], seconds) -> per query (status, n_rows, empty, signature)."""
    path, sqls, seconds = task
    run = connect(path, seconds)
    out = []
    for s in sqls:
        try:
            rows = run(s)
        except Exception as e:
            out.append(("timeout" if "interrupt" in str(e).lower() else "error", 0, False, ""))
            continue
        empty = not any(v is not None for r in rows for v in r)
        out.append(("ok", len(rows), empty, signature(rows)))
    return out


def check_cands(task):
    """(db path, [(gold, [sql])], seconds) -> per question None (gold fails) or per candidate
    dict(runs, empty, ok, sig). `empty` means no rows or only NULLs. `gold` is either SQL (run here) or
    {"sig": ...}, the gold result's signature from the frozen list (gold run once with a longer limit);
    then a candidate is right when its signature matches."""
    path, qs, seconds = task
    run = connect(path, seconds)
    out = []
    for gold, sqls in qs:
        ref, ref_sig = None, None
        if isinstance(gold, dict):
            ref_sig = gold["sig"]
        else:
            try:
                ref = set(map(tuple, run(gold)))
            except Exception:
                out.append(None)
                continue
        res, seen = [], {}
        for s in sqls:
            if not s:
                res.append({"runs": False, "empty": True, "ok": False, "sig": ""})
                continue
            if s not in seen:
                try:
                    got = run(s)
                    sig = signature(got)
                    seen[s] = {"runs": True, "empty": not any(v is not None for r in got for v in r),
                               "ok": sig == ref_sig if ref_sig is not None else set(map(tuple, got)) == ref, "sig": sig}
                except Exception:
                    seen[s] = {"runs": False, "empty": True, "ok": False, "sig": ""}
            res.append(seen[s])
        out.append(res)
    return out


# ---- pick rules (each takes the candidate list in generation order, first = greedy) -----------------

def pick_greedy(cands):
    return cands[0] if cands else None


def pick_vote(cands, allow_empty: bool = False):
    """Largest result group. Without allow_empty (round-4 rule) empty results never win; with it they
    count like any other result, and ties go to the earliest candidate."""
    live = [c for c in cands if c["runs"] and (allow_empty or not c["empty"])]
    if not live:
        return cands[0] if cands else None
    n = collections.Counter(c["sig"] for c in live)
    best = max(n.values())
    return next(c for c in live if n[c["sig"]] == best)


def pick_vote_empty_majority(cands):
    """Empty results win only with a strict majority of the runnable candidates; otherwise round-4 vote."""
    live = [c for c in cands if c["runs"]]
    if live:
        n_empty = sum(c["empty"] for c in live)
        if n_empty * 2 > len(live):
            return next(c for c in live if c["empty"])
    return pick_vote(cands)


def oracle(cands) -> bool:
    return any(c["ok"] for c in cands)


def top2_groups_ok(cands, allow_empty: bool = True) -> bool:
    live = [c for c in cands if c["runs"] and (allow_empty or not c["empty"])]
    n = collections.Counter(c["sig"] for c in live)
    top = {s for s, _ in n.most_common(2)}
    return any(c["ok"] for c in live if c["sig"] in top)


# ---- paired comparison with database-resampled range ---------------------------------------------------

def paired(a: dict, b: dict, db_of: dict, n_boot: int = 4000, seed: int = 0) -> dict:
    """a, b: id -> 0/1 on the same ids. Mean difference a - b in points, 95% range from resampling whole
    databases (questions on one database are not independent), plus wins and losses."""
    ids = [i for i in a if i in b]
    by = collections.defaultdict(list)
    for i in ids:
        by[db_of[i]].append(a[i] - b[i])
    dbs = list(by)
    diff = sum(sum(v) for v in by.values()) / max(len(ids), 1)
    rnd, bs = random.Random(seed), []
    for _ in range(n_boot):
        s = [by[rnd.choice(dbs)] for _ in dbs]
        bs.append(sum(map(sum, s)) / max(sum(map(len, s)), 1))
    bs.sort()
    return {"n": len(ids), "diff": round(100 * diff, 2), "lo": round(100 * bs[int(0.025 * n_boot)], 2),
            "hi": round(100 * bs[int(0.975 * n_boot) - 1], 2),
            "wins": sum(x > 0 for v in by.values() for x in v), "losses": sum(x < 0 for v in by.values() for x in v)}
