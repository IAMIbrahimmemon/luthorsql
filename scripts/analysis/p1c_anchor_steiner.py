"""Phase 1, test 1c — research problem 03-03 (same-named columns across tables, anchor-table errors).

PRE-DECLARED VARIANT (the only one; written before any run):
  A decode-time join-coherence term on the dumped pointer scores, applied only WITHIN a name class, so it changes
  which table copy of a column is taken, never which column name is taken.

  1. Context. Replay the unchanged greedy decoder once (targets._TRACE via headdump.decisions) and read, per pointer
     slot s, the options it chose. C_s = the anchor tables of every option chosen by the OTHER pointer decisions
     (column options give their table; require / via row options give theirs).
  2. Name classes. N(o) = the options of the question whose column name equals o's (case-insensitive), row
     pseudo-options excluded. Only classes with >= 2 members are touched.
  3. Term. For option o=(t,c) in such a class at slot s:
        b(o) = -lambda * [ST(C_s + {t}) - ST(C_s)] + gamma * 1[t named in question or evidence]
     ST = greedy Steiner weight over the FK graph of the dump's schema card (Graph.path costs: declared FK 1,
     inferred 3; an unreachable table costs 10). Table "named": its name (underscores -> spaces, optional plural s)
     occurs as a whole word in the lower-cased question + evidence.
  4. Within-class renormalisation: x'_o = x_o + b(o) - max_{o' in N}(x_o' + b(o')) + max_{o' in N} x_o', so each
     class keeps its best raw score (the name ranking across classes is unchanged) and the copy order inside the
     class follows x + b.
  5. Slots touched: every column pointer slot (projections, filter, group, order, time, conditions, having, x)
     of every block; not plan / require / via / join. The modified scores feed the unchanged decoders (greedy, the
     flip pool, lazy k-best-8), so the whole list is coherent, not just the pick.
  Parameters: lambda in {0, 0.5, 1, 2, 4, 8}, gamma in {0, 0.5, 1, 2, 4} (pointer-logit units; median top-1 gap
  of a pointer slot is ~2.5). Chosen by leave-one-database-out over the 8 DECISION DBs on greedy accuracy of the
  same model; ties to the smallest lambda+gamma. (0, 0) is the baseline itself.
  Gate (frozen): keep if R2 first pick or oracle@8 improves >= +0.5 with DB sign-flip p < 0.10; else drop, and no
  second variant on this split.

  .venv/bin/python scripts/analysis/p1c_anchor_steiner.py      -> scripts/analysis/p1c_anchor_steiner.out
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1c_ae_common as C  # noqa: E402
from luthor.sq.plan import ROWS  # noqa: E402

LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0)
GRID = sorted(((l, g) for l in LAMBDAS for g in GAMMAS), key=lambda p: (p[0] + p[1], p[0]))
UNREACH = 10.0
SKIP = re.compile(r"\.(plan|require|via|join|j\d+)$")
PROJ = re.compile(r"\.p\d+$")


class Steiner:
    def __init__(self):
        self.cache = {}

    def weight(self, db, graph, tables) -> float:
        key = (db, frozenset(tables))
        if key in self.cache:
            return self.cache[key]
        need = sorted(t for t in tables if t in graph.columns)
        w = 0.0
        if len(need) > 1:
            tree, todo = [need[0]], need[1:]
            while todo:
                best = None
                for t in todo:
                    for s in tree:
                        r = graph.path(s, t)
                        if r is not None and (best is None or r[0] < best[1]):
                            best = (t, r[0], r[1])
                if best is None:
                    w += UNREACH * len(todo)
                    break
                t, c, path = best
                w += c
                for e in path:
                    if e[2] not in tree:
                        tree.append(e[2])
                todo.remove(t)
        self.cache[key] = w
        return w


ST = Steiner()


def named_tables(text: str, tables) -> set:
    low = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    out = set()
    for t in tables:
        n = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
        forms = {n, n + "s", n[:-1] if n.endswith("s") else n}
        if any(f and f" {f} " in low for f in forms):
            out.add(t)
    return out


class Prep:
    """Per question: name classes, per-slot context tables and named tables (computed once from the baseline)."""

    def __init__(self, qd):
        ids = qd.option_ids
        self.tab = [o.split(".", 1)[0] for o in ids]
        cls = {}
        for j, o in enumerate(ids):
            if o.endswith("." + ROWS):
                continue
            cls.setdefault(o.split(".", 1)[1].lower(), []).append(j)
        self.classes = [c for c in cls.values() if len(c) >= 2]
        chosen = {}
        for key, order, _allowed, _ap in C.H.decisions(qd, recorded=True):
            if key[0] != "p":
                continue
            s = key[1]
            picks = [order[-1]] if PROJ.search(s) else list(order)
            chosen.setdefault(s, set()).update(picks)
        self.ctx = {}
        for s in chosen:
            self.ctx[s] = {self.tab[j] for s2, js in chosen.items() if s2 != s for j in js}
        self.all_ctx = {self.tab[j] for js in chosen.values() for j in js}
        text = (qd.meta.get("question") or "") + " " + (qd.meta.get("context") or "")
        self.named = named_tables(text, set(self.tab))
        self.db, self.graph = qd.db_id, qd.dump.graph(qd.db_id)


_PREP: dict = {}


def make_arm(qd, p):
    base = qd.scores(recorded=True, strict=False)
    if p is None or (p[0] == 0 and p[1] == 0):
        return base
    lam, gam = p
    pr = _PREP.get(qd.id)
    if pr is None:
        pr = _PREP[qd.id] = Prep(qd)
    if not pr.classes:
        return base
    memo = {}

    def ptr_fn(s, x):
        if SKIP.search(s):
            return x
        if s in memo:
            return list(memo[s])
        ctx = pr.ctx.get(s, pr.all_ctx)  # a slot greedy never visited: context = all greedy tables
        st0 = ST.weight(pr.db, pr.graph, ctx) if lam else 0.0
        y = list(x)
        for cl in pr.classes:
            b = {}
            for j in cl:
                t = pr.tab[j]
                v = 0.0
                if lam and ctx:
                    v -= lam * (ST.weight(pr.db, pr.graph, ctx | {t}) - st0)
                if gam and t in pr.named:
                    v += gam
                b[j] = v
            if all(v == 0.0 for v in b.values()):
                continue
            raw = max(x[j] for j in cl)
            adj = max(x[j] + b[j] for j in cl)
            for j in cl:
                y[j] = x[j] + b[j] - adj + raw
        memo[s] = y
        return list(y)

    return C.BiasScores(base, ptr_fn=ptr_fn)


def fmt(p):
    return f"lambda={p[0]:g} gamma={p[1]:g}"


def main():
    out = Path(__file__).with_suffix(".out")
    log = C.Log(out)
    log(__doc__.split("  .venv")[0].rstrip())
    log(f"\nrun {time.strftime('%Y-%m-%d %H:%M:%S')}; Mac CPU; grid {len(GRID)} points; exec time limit {C.SECONDS}s")
    ex = C.Executor("exec")
    outs = {}
    t0 = time.time()
    for m in ("r2", "r1"):
        _PREP.clear()
        res = C.run_experiment(m, make_arm, GRID, ex, log)
        outs[m] = C.report(res, fmt, log)
    C.gate(outs, log)
    log(f"total runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
