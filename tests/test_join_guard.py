"""Join-route guard (LUTHOR_JOIN_GUARD, src/luthor/sq/plan.py), the bird_desc description modes and train_sq2's
--oom-drop-desc fallback (grammar v4 fix, docs/paper_notes/research/grammar_v4_fix_2026-10-11.md). CPU, no model.

The guard is read once at import (default "off"); these tests switch plan.JOIN_GUARD / the v4 knobs at run time."""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import luthor.sq.plan as pl
from luthor.sq import bird_desc
from luthor.sq.plan import Col, Graph, Plan, Proj, compile_plan

ROOT = Path(__file__).resolve().parents[1]

# customers <- orders (declared FK); orders <- items (declared FK); tags shares a column name with customers;
# regions is not connected to anything.
G = Graph({"customers": ["id", "name", "country", "referrer", "region_code"],
           "orders": ["id", "customer_id", "status", "amount", "rep_id"],
           "items": ["id", "order_id", "sku", "qty"],
           "tags": ["id", "customer_id", "tag"],
           "regions": ["code", "label"]},
          [("orders", "customer_id", "customers", "id"), ("items", "order_id", "orders", "id"),
           ("tags", "customer_id", "customers", "id")])


@pytest.fixture
def v4_pairs():
    old = (pl.JOIN_HINT_EDGES, pl.JOIN_HINT_MODE, pl.JOIN_GUARD)
    pl.JOIN_HINT_EDGES, pl.JOIN_HINT_MODE = True, "pairs"
    try:
        yield
    finally:
        pl.JOIN_HINT_EDGES, pl.JOIN_HINT_MODE, pl.JOIN_GUARD = old


def plan(proj, join_cols, filters=()):
    return Plan(projections=[Proj("none", c) for c in proj], join_cols=list(join_cols), filters=list(filters))


def sql_with(guard, p, g=G):
    pl.JOIN_GUARD = guard
    try:
        return compile_plan(p, g)
    finally:
        pl.JOIN_GUARD = "off"


# ---------------------------------------------------------------------------------------------- flag / defaults
def test_default_is_off_and_bad_value_fails():
    env = {k: v for k, v in os.environ.items() if k != "LUTHOR_JOIN_GUARD"}
    out = subprocess.run([sys.executable, "-c", "import luthor.sq.plan as p; print(p.JOIN_GUARD)"],
                         capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.stdout.strip() == "off"
    bad = subprocess.run([sys.executable, "-c", "import luthor.sq.plan"], capture_output=True, text=True,
                         env={**env, "LUTHOR_JOIN_GUARD": "maybe"}, cwd=ROOT)
    assert bad.returncode != 0 and "LUTHOR_JOIN_GUARD" in bad.stderr


def test_v3_routing_is_identical_under_every_guard():
    """v3 ("disconnected" hint mode) never pairs hints, so no guard may change a v3 compile."""
    assert pl.JOIN_HINT_MODE == "disconnected" or os.environ.get("LUTHOR_GRAMMAR") == "v4"
    old = (pl.JOIN_HINT_EDGES, pl.JOIN_HINT_MODE)
    pl.JOIN_HINT_EDGES, pl.JOIN_HINT_MODE = False, "disconnected"
    try:
        for jc in ([], [Col("orders", "id"), Col("customers", "referrer")],
                   [Col("customers", "region_code"), Col("regions", "code")]):
            p = plan([Col("customers", "name"), Col("orders", "status")], jc)
            ref = sql_with("off", p)
            assert all(sql_with(g, p) == ref for g in pl.JOIN_GUARDS)
    finally:
        pl.JOIN_HINT_EDGES, pl.JOIN_HINT_MODE = old


# ---------------------------------------------------------------------------------------------- route edges
def test_off_keeps_the_v4_behaviour(v4_pairs):
    """Guard off: a predicted non-FK pair beats the declared FK (the v4desc regression mechanism)."""
    p = plan([Col("customers", "name"), Col("orders", "status")], [Col("orders", "id"), Col("customers", "referrer")])
    assert 'T1."referrer" = T2."id"' in sql_with("off", p) or 'T1."id" = T2."referrer"' in sql_with("off", p)


@pytest.mark.parametrize("guard", ["same", "name", "direct"])
def test_wrong_pair_falls_back_to_the_fk(v4_pairs, guard):
    p = plan([Col("customers", "name"), Col("orders", "status")], [Col("orders", "id"), Col("customers", "referrer")])
    s = sql_with(guard, p)
    assert "referrer" not in s and '"customer_id"' in s


@pytest.mark.parametrize("guard", ["same", "name", "direct"])
def test_fk_pair_and_disconnected_pair_are_kept(v4_pairs, guard):
    fk = plan([Col("customers", "name"), Col("orders", "status")], [Col("orders", "customer_id"), Col("customers", "id")])
    assert sql_with(guard, fk) == sql_with("off", fk)
    far = plan([Col("customers", "name"), Col("regions", "label")], [Col("customers", "region_code"), Col("regions", "code")])
    s = sql_with(guard, far)
    assert s == sql_with("off", far) and '"region_code" = ' in s


def test_same_name_pair_beats_a_two_hop_route(v4_pairs):
    """tags.customer_id = orders.customer_id (same name, no FK between the two tables) replaces tags-customers-orders."""
    p = plan([Col("tags", "tag"), Col("orders", "status")], [Col("tags", "customer_id"), Col("orders", "customer_id")])
    for g in ("same", "name", "direct"):
        s = sql_with(g, p)
        assert s == sql_with("off", p) and '"customers"' not in s


def test_direct_allows_a_non_name_shortcut_name_does_not(v4_pairs):
    """items.qty = customers.referrer: not an FK, no name link; the tables are two hops apart (items-orders-customers),
    so "direct" lets the pair replace the multi-hop route while "name" / "same" keep the FK route."""
    p = plan([Col("items", "sku"), Col("customers", "name")], [Col("items", "qty"), Col("customers", "referrer")])
    assert '"orders"' not in sql_with("direct", p) and '"referrer"' in sql_with("direct", p)
    for g in ("same", "name"):
        assert '"orders"' in sql_with(g, p) and '"referrer"' not in sql_with(g, p)


def test_extra_on_conditions_are_guarded(v4_pairs):
    """A wrong second pair between two tables the FK already joins must not become an extra ON condition."""
    p = plan([Col("customers", "name"), Col("orders", "status")],
             [Col("orders", "customer_id"), Col("customers", "id"), Col("orders", "rep_id"), Col("customers", "referrer")])
    assert "AND" in sql_with("off", p)
    for g in ("same", "name", "direct"):
        s = sql_with(g, p)
        assert "AND" not in s and '"customer_id"' in s


def test_guarded_sql_runs(v4_pairs, tmp_path):
    con = sqlite3.connect(tmp_path / "x.sqlite")
    con.executescript("""
    CREATE TABLE customers (id INT, name TEXT, country TEXT, referrer INT, region_code TEXT);
    CREATE TABLE orders (id INT, customer_id INT, status TEXT, amount REAL, rep_id INT);
    INSERT INTO customers VALUES (1,'Ann','EG',2,'A'),(2,'Bo','KE',NULL,'B');
    INSERT INTO orders VALUES (1,1,'open',5,1),(2,2,'shipped',7,2);
    """)
    p = plan([Col("customers", "name"), Col("orders", "status")], [Col("orders", "id"), Col("customers", "referrer")])
    assert sorted(con.execute(sql_with("direct", p)).fetchall()) == [("Ann", "open"), ("Bo", "shipped")]


# ---------------------------------------------------------------------------------------------- name links
@pytest.mark.parametrize("a,b,ok", [
    (("country", "Code"), ("economy", "Country"), True),          # column named after the other table
    (("university", "id"), ("university_year", "university_id"), True),  # <table><column>
    (("address", "address_id"), ("cust_order", "dest_address_id"), True),  # key suffix
    (("part", "p_partkey"), ("lineitem", "l_partkey"), True),     # TPC-H one-letter prefixes
    (("disabled", "name"), ("unemployed", "name"), True),         # same name
    (("university", "id"), ("university_ranking_year", "ranking_criteria_id"), False),
    (("registration", "course_id"), ("student", "student_id"), False),
    (("orderdetails", "orderNumber"), ("orders", "customerNumber"), False),
    (("characters", "ActorID"), ("movie", "MovieID"), False),
    (("Credit", "episode_id"), ("Person", "person_id"), False),
])
def test_name_link(a, b, ok):
    assert pl._name_link(a[0], a[1], b[0], b[1]) is ok
    assert pl._name_link(b[0], b[1], a[0], a[1]) is ok


# ---------------------------------------------------------------------------------------------- descriptions
def test_bird_desc_modes():
    assert not bird_desc.informative("orderNumber", ("order number", "unique order number", ""))
    assert bird_desc.informative("GP", ("game presentations", "", "attendance"))
    assert bird_desc.cryptic("GP") and bird_desc.cryptic("lis_id") and bird_desc.cryptic("ft_pct")
    assert not bird_desc.cryptic("customerName") and not bird_desc.cryptic("orderNumber")

    class C:
        def __init__(self, name):
            self.name, self.description = name, ""

    class T:
        def __init__(self, name, cols):
            self.name, self.columns = name, [C(c) for c in cols]

    class Card:
        db_id = "shop"
        tables = [T("orders", ["orderNumber", "GP", "status"])]

    descs = {"shop": {"orders": {"ordernumber": ("order number", "unique order number", ""),
                                 "gp": ("game presentations", "", "attendance"),
                                 "status": ("status", "order status", "Shipped / Cancelled / On Hold")}}}
    for mode, want in (("full", {"orderNumber", "GP", "status"}), ("info", {"GP", "status"}), ("cryptic", {"GP"})):
        card = Card()
        card.tables = [T("orders", ["orderNumber", "GP", "status"])]
        bird_desc.apply(card, descs, cap=60, mode=mode)
        assert {c.name for c in card.tables[0].columns if c.description} == want, mode
        assert all(len(c.description) <= 60 for c in card.tables[0].columns)
    with pytest.raises(ValueError):
        bird_desc.apply(Card(), descs, mode="some")


# ---------------------------------------------------------------------------------------------- --oom-drop-desc
def test_oom_drop_desc_retries_one_question_without_descriptions(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_sq2 as ts
    calls = []

    def fake(model, card, es, budget_chars, rnd, scale=1.0, drop_desc=False):
        calls.append((len(es), drop_desc))
        if len(es) > 1 or not drop_desc:
            raise torch.OutOfMemoryError("fake")
        return 1.0, {"ptr": 1.0}
    monkeypatch.setattr(ts, "train_db", fake)
    monkeypatch.setattr(ts, "OOM_DROP_DESC", [True, 0])
    led = ts.OOMLedger(1.0)
    assert ts.run_db(None, "db", "db", [1, 2], 9000, None, 1.0, led)[2] == 2
    assert led.skipped == 0 and ts.OOM_DROP_DESC[1] == 2
    assert calls == [(2, False), (1, False), (1, True), (1, False), (1, True)]
    monkeypatch.setattr(ts, "OOM_DROP_DESC", [False, 0])  # off (default): skipped as before, no retry
    calls.clear()
    led = ts.OOMLedger(1.0)
    assert ts.run_db(None, "db", "db", [1], 9000, None, 1.0, led)[2] == 0 and led.skipped == 1
    assert calls == [(1, False)]
