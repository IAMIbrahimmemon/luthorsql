"""Execution-guided top-K decoding for LuthorSQ2 (no retraining).

The greedy decoder makes a few dozen discrete decisions (counts, columns, operators, values). Most
wrong answers differ from a right one in one or two low-confidence decisions. So:

  1. decode greedily, recording each decision's confidence margin (top-1 minus top-2 probability);
  2. re-decode with the least-confident decision(s) flipped to their runner-up, one or two at a time;
  3. compile the distinct candidates in order of confidence, run them, and return the first that
     executes and returns at least one row with a non-NULL value. If none do, return the greedy plan.

Each re-decode reuses the same model outputs (scores are cached), so it costs only plan building and
SQL execution, not another forward pass.
"""
from __future__ import annotations

import itertools
import logging
import math
import random
from collections import Counter
from typing import Callable

from ..sq.plan import Plan
from .targets import Scores, decode

_LOG = logging.getLogger("luthorsql")
SKIPPED: Counter = Counter()  # plans dropped during the search, by reason (predict.py reports it per database)


def _skip(why: str) -> None:
    """A plan that fails to decode, compile or run is dropped from the search: expected, so it is counted, with
    the exception at DEBUG level."""
    SKIPPED[why] += 1
    _LOG.debug("search: plan dropped (%s)", why, exc_info=True)


class _Steer:
    """Wraps Scores: caches each score vector, records margins, and applies forced runner-ups."""

    def __init__(self, base: Scores, cache: dict, forced: dict | None = None, rnd: random.Random | None = None,
                 temp: float = 1.0, margins: dict | None = None):
        self.base, self.cache, self.forced = base, cache, forced or {}
        self.option_ids, self.cands = base.option_ids, base.cands
        self.margins: dict = {}
        self.gaps: dict = {}  # key -> log-prob gap from the best option to ranks 1..3 (exact k-best costs)
        self.rnd, self.temp, self.ref = rnd, temp, margins  # sampling mode: draw every decision
        self.cost = 0.0

    def _get(self, key, fn):
        if key not in self.cache:
            self.cache[key] = list(fn())
        xs = list(self.cache[key])
        if len(xs) < 2:
            return xs
        order = sorted(range(len(xs)), key=lambda i: -xs[i])
        z = xs[order[0]]
        ps = [math.exp(x - z) for x in xs]
        tot = sum(ps)
        self.margins.setdefault(key, (ps[order[0]] - ps[order[1]]) / tot)
        if key not in self.gaps:
            self.gaps[key] = [z - xs[order[i]] for i in range(1, min(4, len(xs)))]
        r = self.forced.get(key)
        if r:
            xs[order[min(r, len(xs) - 1)]] = z + 100.0
        elif self.rnd is not None:
            w = [math.exp((x - z) / self.temp) for x in xs]
            i = self.rnd.choices(range(len(xs)), weights=w)[0]
            if i != order[0]:
                xs[i] = z + 100.0
                self.cost += (self.ref or self.margins).get(key, 0.0)
        return xs

    def choice(self, k):
        return self._get(("c", k), lambda: self.base.choice(k))

    def ptr(self, s):
        return self._get(("p", s), lambda: self.base.ptr(s))

    def pair(self, s, o, h):
        return self._get(("q", s, o, h), lambda: self.base.pair(s, o, h))

    def vals(self, s, o):
        return self._get(("v", s, o), lambda: self.base.vals(s, o))

    def limit(self, s):
        return self._get(("l", s), lambda: self.base.limit(s))


def candidates(sc: Scores, k: int = 8, flip_singles: int = 6, flip_pairs: int = 3, with_cost: bool = False,
               wide_singles: int = 0, wide_pairs: int = 0, runner_up2: int = 0, samples: int = 0,
               temp: float = 1.0, seed: int = 0, distinct: bool = False,
               compile_fn: Callable[[Plan], str] | None = None):
    """Greedy plan first, then plans with the least-confident decisions flipped. With with_cost, returns
    (plan, cost) pairs; cost = summed margins of the flipped decisions (0 for greedy; lower = closer call).

    Widening (round 4) only appends, so the default list is always a prefix of a wider one: flips of the
    next weakest decisions (up to `wide_singles` / pairs among the `wide_pairs` weakest), third choices for
    the `runner_up2` weakest, then plans with every decision drawn from its softmax at `temp` (cost =
    margins of the decisions that changed). Widened plans that repeat an earlier plan are dropped."""
    cache: dict = {}
    greedy = _Steer(sc, cache)
    out = [decode(greedy)]
    costs = [0.0]
    weak = sorted(greedy.margins, key=greedy.margins.get)
    if distinct:
        return _distinct(sc, cache, greedy, out[0], weak, k, temp, seed, with_cost, compile_fn)
    flips = [{key: 1} for key in weak[:flip_singles]]
    base_pairs = set(itertools.combinations(weak[:flip_pairs], 2))
    flips += [{a: 1, b: 1} for a, b in itertools.combinations(weak[:flip_pairs], 2)]
    n_base = len(flips)
    flips += [{key: 1} for key in weak[flip_singles:wide_singles]]
    flips += [{a: 1, b: 1} for a, b in itertools.combinations(weak[:wide_pairs], 2) if (a, b) not in base_pairs]
    flips += [{key: 2} for key in weak[:runner_up2]]
    seen: set = set()
    for i, f in enumerate(flips):
        if len(out) >= k:
            break
        try:
            p = decode(_Steer(sc, cache, f))
        except Exception:
            _skip("decode")
            continue
        if i >= n_base:
            seen = seen or {repr(x) for x in out}
            if repr(p) in seen:
                continue
            seen.add(repr(p))
        out.append(p)
        costs.append(sum(greedy.margins[key] for key in f))
    if samples > 0:
        seen = {repr(x) for x in out}
        rnd = random.Random(seed)
        for _ in range(samples * 3):  # duplicates are common; try a few times more
            if len(out) >= k:
                break
            st = _Steer(sc, cache, rnd=rnd, temp=temp, margins=greedy.margins)
            try:
                p = decode(st)
            except Exception:
                _skip("decode")
                continue
            if repr(p) not in seen:
                seen.add(repr(p))
                out.append(p)
                costs.append(st.cost)
    return list(zip(out, costs)) if with_cost else out


def _distinct(sc: Scores, cache: dict, greedy: _Steer, g: Plan, weak: list, k: int, temp: float, seed: int,
              with_cost: bool, compile_fn: Callable[[Plan], str] | None = None):
    """Round 7: k DIFFERENT plans. The default list repeats itself (BIRD dev top 8 held only 3.6 distinct SQL
    on average: flipping a decision the plan does not use changes nothing). Here lazy k-best (`kbest`) supplies
    plans in order of summed log-prob gaps, then softmax samples fill any shortfall; repeats are dropped and the
    rest are ranked by cost. With compile_fn, plans that do not compile are dropped and repeats are judged on
    the SQL itself."""
    def key_of(p: Plan) -> str | None:
        if compile_fn is None:
            return repr(p)
        try:
            return compile_fn(p)
        except Exception:
            _skip("compile")
            return None

    gk = key_of(g) or repr(g)  # the greedy plan always stays first, even if it does not compile
    pool = {gk: (0.0, g)}
    for p, c in kbest(sc, k, compile_fn, with_cost=True)[1:]:  # exact-style k-best first (round 7)
        r = key_of(p)
        if r is not None and r not in pool:
            pool[r] = (c, p)
    rnd = random.Random(seed)
    tries = 0
    top = max(c for c, _ in pool.values())  # samples rank after every k-best plan (their costs use another scale)
    while len(pool) < k and tries < 6 * k:  # samples only when k-best found too few
        tries += 1
        st = _Steer(sc, cache, rnd=rnd, temp=temp, margins=greedy.margins)
        try:
            p = decode(st)
        except Exception:
            _skip("decode")
            continue
        r = key_of(p)
        if r is not None:
            pool.setdefault(r, (top + 1.0 + st.cost, p))
    rest = sorted((v for r, v in pool.items() if r != gk), key=lambda v: v[0])
    out = [(g, 0.0)] + [(p, c) for c, p in rest[:k - 1]]
    return out if with_cost else [p for p, _ in out]


def kbest(sc: Scores, k: int, compile_fn: Callable[[Plan], str] | None = None, max_pops: int = 0,
          with_cost: bool = False):
    """Exact-style lazy k-best (round 7; Kool et al. 2019 / k-best of independent factors). A plan is the
    greedy decode with some decisions forced to their rank-r option; its cost is the summed log-prob gaps of
    the forced decisions. A priority queue pops the cheapest unexplored forcing; its children bump one
    decision THAT THIS PLAN USES by one rank. Decisions a plan does not use are never expanded, so no slot
    is wasted on flips that change nothing. Repeats (same compiled SQL) are dropped."""
    import heapq
    cache: dict = {}

    def run(forced):
        st = _Steer(sc, cache, forced)
        p = decode(st)
        return p, st

    def key_of(p):
        if compile_fn is None:
            return repr(p)
        try:
            return compile_fn(p)
        except Exception:
            _skip("compile")
            return None

    g, gst = run({})
    out, seen_sql, seen_f = [(g, 0.0)], {key_of(g) or repr(g)}, {frozenset()}
    heap, tie = [], 0

    def expand(forced, cost, st):
        nonlocal tie
        for key, gaps in st.gaps.items():
            r = forced.get(key, 0)
            if r < len(gaps):
                f = dict(forced)
                f[key] = r + 1
                fk = frozenset(f.items())
                if fk in seen_f:
                    continue
                seen_f.add(fk)
                c = cost + gaps[r] - (gaps[r - 1] if r else 0.0)
                tie += 1
                heapq.heappush(heap, (c, tie, f))

    expand({}, 0.0, gst)
    pops, limit = 0, max_pops or 8 * k
    while heap and len(out) < k and pops < limit:
        c, _, f = heapq.heappop(heap)
        pops += 1
        try:
            p, st = run(f)
        except Exception:
            _skip("decode")
            continue
        expand(f, c, st)
        kk = key_of(p)
        if kk is None or kk in seen_sql:
            continue
        seen_sql.add(kk)
        out.append((p, c))
    return out if with_cost else [p for p, _ in out]


def choose(plans: list[Plan], compile_fn: Callable[[Plan], str], run_fn: Callable[[str], list],
           costs: list[float] | None = None, max_cost: float = 1.0) -> tuple[Plan, str | None, int]:
    """Keep the greedy plan if it runs. Otherwise (or if it runs but returns nothing / only NULLs, and an
    alternative within `max_cost` returns rows), take the first such alternative. Returns (plan, sql, rank)."""
    seen = set()
    first_sql = None
    for rank, p in enumerate(plans):
        try:
            sql = compile_fn(p)
        except Exception:
            _skip("compile")
            continue
        if sql in seen:
            continue
        seen.add(sql)
        first_sql = first_sql or sql
        try:
            rows = run_fn(sql)
        except Exception:
            _skip("run")
            continue
        if rank == 0:
            greedy_ran = (p, sql)
        if rows and any(v is not None for r in rows for v in r):
            if rank == 0 or costs is None or costs[rank] <= max_cost:
                return p, sql, rank
    if "greedy_ran" in locals():
        return greedy_ran[0], greedy_ran[1], 0
    return plans[0], first_sql, -1
