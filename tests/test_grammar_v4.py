"""Grammar v4 / v4.1: each new construct converts, compiles to SQL that returns the gold rows, and survives the
slot round trip (plan -> sq2 targets -> decode with perfect scores -> same rows); with the flag off (the default)
everything is grammar v3.

The flag is LUTHOR_GRAMMAR, read once at import. Converter / compiler tests switch the v4 knobs on at run time
(`v4_on`); the slot tests need the v4 layout, so under the default (v3) this file is re-run in a subprocess with
LUTHOR_GRAMMAR=v4 (test_v4_suite_in_subprocess)."""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import luthor.sq.convert as cv
import luthor.sq.plan as pl
from luthor.sq.convert import convert
from luthor.sq.plan import ROWS, Col, Filter, Graph, Plan, Proj, compile_plan
from luthor.sq2 import targets as T
from luthor.sq2.serialize import GRAMMAR_V4, SLOTS

G = Graph({"customers": ["id", "name", "country", "signup", "referrer"],
           "orders": ["id", "customer_id", "status", "amount", "budget"],
           "notes": ["ref", "text"]},
          [("orders", "customer_id", "customers", "id")])


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    p = tmp_path_factory.mktemp("v4") / "shop.sqlite"
    con = sqlite3.connect(p)
    con.executescript("""
    CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT, signup TEXT, referrer INT);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INT, status TEXT, amount REAL, budget REAL);
    CREATE TABLE notes (ref INT, text TEXT);
    INSERT INTO customers VALUES (1,'Ann','Egypt','2023-02-01',NULL),(2,'Bo','Kenya','2024-05-03',1),
                                 (3,'Cy','Egypt','2024-07-09',1),(4,'Di','Chad','2022-01-01',2);
    INSERT INTO orders VALUES (1,1,'shipped',10,8),(2,2,'cancelled',5,9),(3,1,'shipped',7,7),(4,3,'shipped',3,1),
                              (5,4,'shipped',12,20),(6,4,'open',2,2);
    INSERT INTO notes VALUES (1,'vip'),(3,'late');
    """)
    con.commit()
    return con


def rows(con, sql):
    return sorted(map(tuple, con.execute(sql).fetchall()))


_KNOBS = ((cv, "V4"), (cv, "JOIN_PAIRS"), (pl, "JOIN_HINT_EDGES"), (pl, "JOIN_HINT_MODE"), (pl, "TEXT_FN_WHOLE"))


@pytest.fixture
def v4_on():
    """The converter / compiler side of grammar v4 at run time (what LUTHOR_GRAMMAR=v4 sets at import)."""
    old = [(m, k, getattr(m, k)) for m, k in _KNOBS]
    cv.V4 = set(cv.V4_ALL)
    cv.JOIN_PAIRS, pl.JOIN_HINT_EDGES, pl.JOIN_HINT_MODE, pl.TEXT_FN_WHOLE = True, True, "pairs", True
    try:
        yield
    finally:
        for m, k, v in old:
            setattr(m, k, v)


V4_SQL = [
    # flatten: outer ORDER BY / LIMIT over a grouped derived table
    ("SELECT T.country FROM (SELECT T2.country, SUM(T1.amount) AS num FROM orders T1 JOIN customers T2 "
     "ON T1.customer_id = T2.id GROUP BY T2.country) T ORDER BY T.num DESC LIMIT 1", "flatten"),
    # flatten: BIRD's CASE ... WHERE T IS NOT NULL idiom
    ("SELECT DISTINCT T FROM (SELECT CASE WHEN status = 'shipped' THEN customer_id ELSE NULL END AS T FROM orders) "
     "WHERE T IS NOT NULL", "flatten"),
    # COUNT(*) over a grouped derived table -> wrap=count
    ("SELECT COUNT(*) FROM (SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(id) > 1)", "wrap_count"),
    # CTE inlined, then flattened
    ("WITH s AS (SELECT customer_id, SUM(amount) AS tot FROM orders GROUP BY customer_id) "
     "SELECT customer_id FROM s ORDER BY tot DESC LIMIT 2", "cte"),
    ("SELECT name FROM customers ORDER BY signup DESC LIMIT 1 OFFSET 1", "offset"),
    ("SELECT name FROM customers ORDER BY country ASC, name DESC", "order2"),
    ("SELECT COUNT(*) FROM orders WHERE amount > budget", "colcmp"),
    ("SELECT SUM(CASE WHEN amount = budget THEN 1 ELSE 0 END) FROM orders", "colcmp"),
    ("SELECT id, customer_id, status, amount, budget FROM orders WHERE status = 'open'", "proj5+"),
    ("SELECT (SELECT COUNT(id) FROM orders WHERE status = 'shipped') - (SELECT COUNT(id) FROM orders WHERE status = 'open')",
     "scalar"),
]


@pytest.mark.parametrize("sql,tag", V4_SQL)
def test_v4_convert_then_compile_matches_gold(db, v4_on, sql, tag):
    c = convert(sql, G)
    assert c.plan is not None, c.reason
    assert tag in cv.TAGS
    assert rows(db, compile_plan(c.plan, G)) == rows(db, sql)
    assert rows(db, compile_plan(Plan.from_dict(c.plan.to_dict()), G)) == rows(db, sql)  # serialisation


@pytest.mark.parametrize("sql,tag", V4_SQL)
def test_v4_constructs_off_give_v3_behaviour(v4_on, sql, tag):
    old = set(cv.V4)
    cv.V4.clear()
    try:
        assert convert(sql, G).plan is None
    finally:
        cv.V4.update(old)


class _Oracle:
    """targets.Scores with all mass on the gold targets (as scripts/roundtrip_v3.py)."""

    def __init__(self, t, ids, cands):
        self.t, self.option_ids, self.cands = t, ids, cands
        self.ix = {o: i for i, o in enumerate(ids)}
        self.pairs = {(s, o, h): c for s, o, h, c in t.pair}
        self.vals_ = {(s, o): idx for s, o, idx in t.vals}

    def _hot(self, n, ks):
        out = [0.0] * n
        for k in ks:
            out[k] = 20.0
        return out

    def choice(self, key):
        b, k = key.split(".", 1)
        names = (T.MAIN_CHOICES if k in T.MAIN_CHOICES and b == "m" else T.CHOICES)[k]
        return self._hot(len(names), [self.t.choice.get(key, 0)])

    def ptr(self, slot):
        n = len(self.option_ids)
        if slot in self.t.ptr:
            return self._hot(n, [self.ix[self.t.ptr[slot]]])
        out = [0.0] * n
        for r, o in enumerate(self.t.sets.get(slot, [])):
            out[self.ix[o]] = 1.0 + 0.001 * (len(self.t.sets[slot]) - r)
        return out

    def pair(self, slot, opt, head):
        return self._hot(len(T.PAIRS[head]), [self.pairs.get((slot, self.option_ids[opt], head), 0)])

    def vals(self, slot, opt):
        return self._hot(len(self.cands), self.vals_.get((slot, self.option_ids[opt]), []))

    def limit(self, slot):
        return self._hot(len(self.cands), [self.t.limit[slot]] if slot in self.t.limit else [])


@pytest.mark.skipif(not GRAMMAR_V4, reason="LUTHOR_GRAMMAR=v3")
@pytest.mark.parametrize("plan", [
    Plan([Proj("none", Col("customers", "name"))], order_dir="desc", order=Proj("none", Col("customers", "signup")),
         limit=1, offset=2),
    Plan([Proj("none", Col("customers", "name"))], order_dir="asc", order=Proj("none", Col("customers", "country")),
         order2=Proj("none", Col("customers", "name")), order2_dir="desc"),
    Plan([Proj("count", Col("orders", ROWS))], filters=[Filter(Col("orders", "amount", col2="orders.budget"), "gt_col")]),
    Plan([Proj("none", Col("orders", c)) for c in ("id", "customer_id", "status", "amount", "budget")]),
    Plan([Proj("none", Col("orders", "id"))], group_by=[Col("orders", c) for c in ("id", "customer_id", "status")]),
])
def test_v4_slot_round_trip(db, v4_on, plan):
    ids = [f"{t}.{c}" for t, cols in G.columns.items() for c in [ROWS] + cols]
    cands = []
    t = T.build_targets(plan, cands)
    back = T.decode(_Oracle(t, ids, cands))
    assert rows(db, compile_plan(back, G)) == rows(db, compile_plan(plan, G))


def test_join_hint_edges_only_bridge_disconnected_tables(db, v4_on):
    # notes has no FK: v3 cannot compile a plan touching it; v4 joins along the plan's hint columns
    p = Plan([Proj("none", Col("notes", "text"))], filters=[Filter(Col("customers", "country"), "=", ["Egypt"])],
             join_cols=[Col("customers", "id"), Col("notes", "ref")])
    gold = "SELECT T2.text FROM customers T1 JOIN notes T2 ON T1.id = T2.ref WHERE T1.country = 'Egypt'"
    assert rows(db, compile_plan(p, G)) == rows(db, gold)
    pl.JOIN_HINT_EDGES = False
    try:
        with pytest.raises(pl.CompileError):
            compile_plan(p, G)
    finally:
        pl.JOIN_HINT_EDGES = True
    # "disconnected" mode: tables already connected by a foreign key keep their route
    old, pl.JOIN_HINT_MODE = pl.JOIN_HINT_MODE, "disconnected"
    try:
        assert G.with_hint_edges([Col("orders", "budget"), Col("customers", "referrer")], pl.HINT_EDGE_COST) is G
    finally:
        pl.JOIN_HINT_MODE = old


def test_join_pairs_follow_the_gold_join_condition(db, v4_on):
    # the FK says orders.customer_id = customers.id; the gold joins orders.budget = customers.id instead
    gold = "SELECT T2.name FROM orders T1 JOIN customers T2 ON T1.budget = T2.id WHERE T1.status = 'open'"
    c = convert(gold, G)
    assert [j.id for j in c.plan.join_cols] == ["orders.budget", "customers.id"]
    assert rows(db, compile_plan(c.plan, G)) == rows(db, gold)
    old, pl.JOIN_HINT_MODE = pl.JOIN_HINT_MODE, "disconnected"
    try:
        assert rows(db, compile_plan(c.plan, G)) != rows(db, gold)  # v3 routing takes the FK
    finally:
        pl.JOIN_HINT_MODE = old


def test_multi_column_join_keeps_both_conditions(db, v4_on):
    gold = ("SELECT COUNT(*) FROM orders T1 JOIN customers T2 ON T1.customer_id = T2.id AND T1.budget = T2.referrer")
    c = convert(gold, G)
    assert len(c.plan.join_cols) == 4
    sql = compile_plan(c.plan, G)
    assert " AND " in sql.split(" ON ", 1)[1]
    assert rows(db, sql) == rows(db, gold)


# ---------------------------------------------------------------- grammar v4.1 (round 2), without the dev-derived
# value-span rules (values41), which stay out of main

V41_SQL = [
    ("SELECT CAST(SUM(CASE WHEN T2.country = 'Egypt' AND T1.status = 'shipped' THEN 1 ELSE 0 END) AS REAL) * 100 "
     "/ COUNT(*) FROM orders T1 JOIN customers T2 ON T1.customer_id = T2.id", "cond2"),
    ("SELECT SUM(IIF(status = 'shipped' AND amount > 5, amount, 0)) FROM orders", "cond2"),
    ("SELECT COUNT(CASE WHEN T2.country = 'Egypt' AND T1.amount >= 7 THEN T1.id ELSE NULL END) FROM orders T1 "
     "JOIN customers T2 ON T1.customer_id = T2.id", "cond2"),
    ("SELECT SUM(status = 'shipped') - SUM(status = 'open') FROM orders", "boolsum"),
    ("SELECT CAST(SUM(status = 'shipped' AND amount < 10) AS REAL) / COUNT(id) FROM orders", "boolsum"),
    ("SELECT CAST((SUM(IIF(status = 'shipped', amount, 0)) - SUM(IIF(status = 'open', amount, 0))) AS REAL) * 100 "
     "/ SUM(IIF(status = 'open', amount, 0)) FROM orders", "dedup"),
    ("SELECT name FROM customers WHERE STRFTIME('%Y', DATE('now')) - STRFTIME('%Y', signup) > 1", "now"),
    ("SELECT name FROM customers WHERE STRFTIME('%Y', 'now') - STRFTIME('%Y', signup) > 1", "now"),
]


@pytest.mark.parametrize("sql,tag", V41_SQL)
def test_v41_convert_then_compile_matches_gold(db, v4_on, sql, tag):
    c = convert(sql, G)
    assert c.plan is not None, c.reason
    assert tag in cv.TAGS
    assert rows(db, compile_plan(c.plan, G)) == rows(db, sql)
    assert rows(db, compile_plan(Plan.from_dict(c.plan.to_dict()), G)) == rows(db, sql)


@pytest.mark.parametrize("sql,tag", V41_SQL)
def test_v41_switch_off(v4_on, sql, tag):
    cv.V4.discard(tag)
    if tag == "boolsum":
        cv.V4.discard("cond2")
    c = convert(sql, G)
    assert c.plan is None or tag not in cv.TAGS


@pytest.mark.skipif(not GRAMMAR_V4, reason="LUTHOR_GRAMMAR=v3")
@pytest.mark.parametrize("sql,tag", [x for x in V41_SQL if x[1] in ("cond2", "boolsum", "dedup")])
def test_v41_slot_round_trip(db, v4_on, sql, tag):
    assert "m.d0" in SLOTS and "cond2_mask" in T.MAIN_CHOICES
    plan = convert(sql, G).plan
    old = dict(T.VOCAB), dict(T.PAIRS), dict(T.MAIN_CHOICES)
    tm = [plan.compose] if plan.compose not in T.COMPOSE else []
    T.set_vocab(tm, [100])
    try:
        from luthor.sq.candidates import Candidate
        ids = [f"{t}.{c}" for t, cols in G.columns.items() for c in ["*"] + cols]
        lits = [v for pr in plan.projections for f in (pr.cond, pr.cond2) if f is not None for v in f.values]
        cands = [Candidate("span", v, None, (0, 1)) for v in dict.fromkeys(lits)]
        t = T.build_targets(Plan.from_dict(plan.to_dict()), cands)
        back = T.decode(_Oracle(t, ids, cands))
        assert rows(db, compile_plan(back, G)) == rows(db, sql)
    finally:
        T.VOCAB.clear(), T.VOCAB.update(old[0])
        T.PAIRS.clear(), T.PAIRS.update(old[1])
        T.MAIN_CHOICES.clear(), T.MAIN_CHOICES.update(old[2])


# ---------------------------------------------------------------- the flag


@pytest.mark.skipif(GRAMMAR_V4, reason="checks the default (v3) process")
def test_default_is_v3():
    assert pl.GRAMMAR == "v3" and not cv.V4 and not cv.JOIN_PAIRS and not pl.JOIN_HINT_EDGES and not pl.TEXT_FN_WHOLE
    assert not set(pl.COL_OPS) & set(pl.OPS) and T.PAIRS["op"] == pl.OPS
    assert len(pl.OPS) == 19  # the v1 / v3 op heads keep their size
    assert "m.j0" not in SLOTS and "m.d0" not in SLOTS and "offset" not in T.CHOICES
    assert T.CHOICES["n_proj"] == tuple("1234") and T.MAIN_CHOICES["n_join"] == tuple("0123456")
    # a v3 plan serialises without the v4 fields (byte-identical plans.jsonl rows)
    d = Plan([Proj("count", Col("orders", ROWS))], filters=[Filter(Col("orders", "status"), "=", ["open"])]).to_dict()
    assert not {"offset", "order2", "order2_dir"} & set(d) and "cond2" not in d["projections"][0]
    # v4 plans are rejected by the v3 heads, not mis-encoded
    for p in (Plan([Proj("none", Col("customers", "name"))], limit=1, offset=1),
              Plan([Proj("count", Col("orders", ROWS))], filters=[Filter(Col("orders", "amount", col2="orders.budget"),
                                                                         "gt_col")])):
        with pytest.raises(T.Unsupported):
            T.build_targets(p, [])


def test_v4_fields_round_trip_through_dict():
    p = Plan([Proj("none", Col("customers", "name"))], order_dir="asc", order=Proj("none", Col("customers", "country")),
             order2=Proj("max", Col("customers", "signup")), order2_dir="desc", limit=3, offset=2)
    assert Plan.from_dict(p.to_dict()) == p


def test_bad_flag_value_fails_loudly():
    env = {**os.environ, "LUTHOR_GRAMMAR": "v5"}
    r = subprocess.run([sys.executable, "-c", "import luthor.sq.plan"], env=env, capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode != 0 and "LUTHOR_GRAMMAR" in r.stderr


@pytest.mark.skipif(GRAMMAR_V4 or os.environ.get("_LUTHOR_V4_SUB"), reason="already the v4 run")
def test_v4_suite_in_subprocess():
    """The slot-layout tests of this file under LUTHOR_GRAMMAR=v4 (they need the flag at import time)."""
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "LUTHOR_GRAMMAR": "v4", "_LUTHOR_V4_SUB": "1"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(Path(__file__))],
                       env=env, capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert " passed, 2 skipped" in r.stdout.splitlines()[-1], r.stdout.splitlines()[-1]  # only the two v3-only tests
