"""Core System-1 tests: plan compiler, SQL->plan converter, and the bi-encoder + losses on a
tiny randomly initialised decoder (no download, no GPU)."""
import glob
import sqlite3
from pathlib import Path

import pytest
import torch

from luthor.sq.convert import convert
from luthor.sq.plan import Col, Filter, Graph, Plan, Proj, compile_plan

G = Graph({"customers": ["id", "name", "country", "signup"],
           "orders": ["id", "customer_id", "status", "amount"],
           "items": ["order_id", "product_id", "qty"],
           "products": ["id", "category"]},
          [("orders", "customer_id", "customers", "id"), ("items", "order_id", "orders", "id"),
           ("items", "product_id", "products", "id")])


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    p = tmp_path_factory.mktemp("sq") / "shop.sqlite"
    con = sqlite3.connect(p)
    con.executescript("""
    CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT, signup TEXT);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INT REFERENCES customers(id), status TEXT, amount REAL);
    CREATE TABLE items (order_id INT REFERENCES orders(id), product_id INT REFERENCES products(id), qty INT);
    CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT);
    INSERT INTO customers VALUES (1,'Ann','Egypt','2023-02-01'),(2,'Bo','Kenya','2024-05-03'),(3,'Cy','Egypt','2024-07-09');
    INSERT INTO orders VALUES (1,1,'shipped',10),(2,2,'cancelled',5),(3,1,'shipped',7),(4,3,'shipped',3);
    INSERT INTO products VALUES (1,'toys'),(2,'books');
    INSERT INTO items VALUES (1,1,2),(1,2,1),(2,2,5),(3,1,1),(4,2,4);
    """)
    con.commit()
    return con


def rows(con, sql):
    return sorted(map(tuple, con.execute(sql).fetchall()))


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM orders WHERE status = 'shipped'",
    "SELECT T2.country, SUM(T1.amount) FROM orders AS T1 JOIN customers AS T2 ON T1.customer_id = T2.id GROUP BY T2.country",
    "SELECT T3.category, SUM(T1.qty) FROM items T1 JOIN orders T2 ON T1.order_id=T2.id JOIN products T3 ON T1.product_id=T3.id "
    "WHERE T2.status = 'shipped' GROUP BY T3.category",
    "SELECT name FROM customers WHERE strftime('%Y', signup) = '2024' ORDER BY name DESC LIMIT 1",
    "SELECT CAST(SUM(CASE WHEN status = 'shipped' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM orders",
    "SELECT COUNT(DISTINCT country) FROM customers WHERE name LIKE '%n%'",
    "SELECT name FROM customers WHERE country = 'Egypt' OR country = 'Kenya'",
    "SELECT T1.name FROM customers T1, orders T2 WHERE T1.id = T2.customer_id AND T2.amount BETWEEN 4 AND 8",
])
def test_convert_then_compile_matches_gold(db, sql):
    c = convert(sql, G)
    assert c.plan is not None, c.reason
    assert rows(db, compile_plan(c.plan, G)) == rows(db, sql)


@pytest.mark.parametrize("sql", [
    # grammar v2: subqueries (one level), HAVING, OR groups, set operations, LIKE prefix/suffix
    "SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders WHERE status = 'shipped')",
    "SELECT name FROM customers WHERE id NOT IN (SELECT customer_id FROM orders)",
    "SELECT id FROM orders WHERE amount > (SELECT AVG(amount) FROM orders)",
    "SELECT country FROM customers GROUP BY country HAVING COUNT(*) > 1",
    "SELECT T2.name FROM orders T1 JOIN customers T2 ON T1.customer_id = T2.id GROUP BY T2.name HAVING SUM(T1.amount) >= 10",
    "SELECT name FROM customers WHERE country = 'Egypt' UNION SELECT name FROM customers WHERE signup LIKE '2024%'",
    "SELECT id FROM customers INTERSECT SELECT customer_id FROM orders",
    "SELECT id FROM customers EXCEPT SELECT customer_id FROM orders WHERE status = 'shipped'",
    "SELECT id FROM orders WHERE (status = 'cancelled' OR amount > 8) AND customer_id = 1",
    "SELECT name FROM customers WHERE name LIKE 'A%' OR country = 'Kenya'",
    "SELECT name FROM customers WHERE name LIKE '%y'",
])
def test_grammar_v2_convert_then_compile_matches_gold(db, sql):
    c = convert(sql, G)
    assert c.plan is not None, c.reason
    assert c.plan.uses_v2()
    assert rows(db, compile_plan(c.plan, G)) == rows(db, sql)
    assert Plan.from_dict(c.plan.to_dict()) == c.plan


@pytest.mark.parametrize("sql,reason", [
    ("SELECT RANK() OVER (ORDER BY amount) FROM orders", "window"),
    ("SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders WHERE id IN (SELECT order_id FROM items))", "subquery"),
    ("SELECT T1.name FROM customers T1 WHERE T1.id > (SELECT COUNT(*) FROM orders T2 WHERE T2.customer_id = T1.id)", "alias"),
    ("SELECT name FROM customers UNION ALL SELECT status FROM orders", "union all"),
])
def test_convert_unsupported(sql, reason):
    c = convert(sql, G)
    assert c.plan is None and reason in c.reason


def test_compiler_builds_bridge_joins(db):
    # customers -> products has no direct edge: the compiler must go through orders and items
    p = Plan(projections=[Proj("none", Col("customers", "name"))],
             filters=[Filter(Col("products", "category"), "=", ["books"])], distinct=True)
    sql = compile_plan(p, G)
    assert "orders" in sql and "items" in sql
    assert rows(db, sql) == [("Ann",), ("Bo",), ("Cy",)]


def test_plan_roundtrip():
    p = Plan(projections=[Proj("count", Col("orders", "*"), Filter(Col("orders", "status"), "=", ["shipped"]))],
             filters=[Filter(Col("customers", "signup"), "=", ["2024"], "year")])
    assert Plan.from_dict(p.to_dict()) == p


# ---- model ------------------------------------------------------------------------------
def _tiny_model():
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model

    from luthor.sq.model import Backbone, LuthorSQ

    tok_dir = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/*"))
    if not tok_dir:
        pytest.skip("no local tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir[0])
    cfg = Qwen2Config(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    torch.manual_seed(0)
    return LuthorSQ(Backbone(Qwen2Model(cfg), tok), d_proj=32)


def _card():
    from luthor.sq.schema_card import ColumnCard, SchemaCard, TableCard

    return SchemaCard("shop", "sqlite", [
        TableCard("customers", "people who buy", [ColumnCard("id", "int", pk=True), ColumnCard("name", "text"),
                                                  ColumnCard("country", "text", "country of residence")]),
        TableCard("orders", "purchases", [ColumnCard("id", "int", pk=True), ColumnCard("customer_id", "int"),
                                          ColumnCard("status", "text", "shipped or cancelled"), ColumnCard("amount", "real")]),
    ], [("orders", "customer_id", "customers", "id")],
        values={"customers.country": ["Egypt", "Kenya"], "orders.status": ["shipped", "cancelled"]})


def _rows():
    base = {"source": "t", "split": "train", "db_id": "shop", "context": "", "outcome": "plan"}
    mk = lambda i, q, plan, **kw: {**base, "id": i, "question": q, "plan": plan.to_dict(), **kw}
    p1 = Plan([Proj("sum", Col("orders", "amount"))], filters=[Filter(Col("customers", "country"), "=", ["Egypt"])])
    p2 = Plan([Proj("sum", Col("orders", "amount"))], filters=[Filter(Col("customers", "country"), "=", ["Kenya"])])
    p3 = Plan([Proj("none", Col("customers", "name"))], order_dir="desc", order=Proj("sum", Col("orders", "amount")), limit=1)
    return [mk("a", "total amount of orders from customers in Egypt", p1),
            mk("a_tw", "total amount of orders from customers in Kenya", p2, twin_of="a", twin_slot="value"),
            mk("b", "which customer spent the most", p3),
            {**base, "id": "u", "question": "list customers who never ordered", "outcome": "unsupported"}]


def test_model_loss_decreases_and_decodes():
    import json
    import tempfile

    from luthor.sq.data import batches, load_examples
    from luthor.sq.train import decode, forward_db, train_step

    card = _card()
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in _rows():
            f.write(json.dumps(r) + "\n")
    exs = load_examples(Path(f.name), {"shop": card})
    assert len(exs) == 4
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for step in range(60):
        for batch in batches(exs, {"shop": card}, dbs_per_batch=1, q_per_db=8, seed=step):
            loss, parts = train_step(model, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else float(loss.detach())
    assert float(loss) < 0.5 * first, (first, float(loss), parts)
    model.eval()
    with torch.no_grad():
        b = next(batches(exs, {"shop": card}, dbs_per_batch=1, q_per_db=8))[0]
        outs, _ = forward_db(model, b)
        plans = {e.id: decode(model, o, b.option_ids, e.cands) for o, e in outs}
    assert plans["u"] is None
    assert plans["a"].filters[0].values == ["Egypt"] and plans["a_tw"].filters[0].values == ["Kenya"]
    assert plans["b"].order_dir == "desc" and plans["b"].limit == 1
    sql = compile_plan(plans["a"], card.graph())
    assert "JOIN" in sql and "Egypt" in sql


def test_runtime_uses_cached_schema_vectors():
    from luthor.sq.runtime import Runtime, TorchEncoder

    model = _tiny_model().eval()
    rt = Runtime(model, TorchEncoder(model))
    card = _card()
    rt.index(card)
    assert rt.cache["shop"]["O"].shape[0] == len(card.options())
    plan, sql, t = rt.plan("shop", "total amount of orders from customers in Egypt")
    assert (plan is None) == (sql is None) and t["total_ms"] > 0


def test_candidates_normalise_literals():
    from luthor.sq.candidates import find_candidates

    card = _card()
    card.values["customers.country"] = ["United States", "Egypt"]
    card.values["orders.status"] = ["shipped", "refund"]
    text = "orders over 1,500 or 10k in States United since June 29, 2011, about 15% refunds, top twelve"
    vals = {(c.kind, c.value) for c in find_candidates(text, card)}
    for v in [1500, 10000, "2011-06-29", 0.15, 12]:
        assert ("span", v) in vals, v
    assert ("value", "United States") in vals and ("value", "refund") in vals


def test_double_quoted_strings_are_literals(db):
    c = convert('SELECT name FROM customers WHERE country = "Egypt"', G)
    assert c.plan is not None and c.plan.filters[0].values == ["Egypt"]


def test_join_prefers_declared_edges_and_follows_hints():
    from luthor.sq.plan import join_tree

    # a-b-d and a-c-d are both two hops; a-d is a direct but name-inferred edge
    g = Graph({"a": ["id", "b_id", "c_id", "d_id"], "b": ["id", "d_id"], "c": ["id", "d_id"], "d": ["id"]},
              [("a", "b_id", "b", "id"), ("b", "d_id", "d", "id"), ("a", "c_id", "c", "id"),
               ("c", "d_id", "d", "id"), ("a", "d_id", "d", "id")],
              inferred={("a", "d_id", "d", "id")})
    tables, _ = join_tree(g, ["a", "d"])
    assert tables == ["a", "b", "d"]                       # 2 declared hops (cost 2) beat 1 inferred (cost 3)
    tables, _ = join_tree(g, ["a", "d"], via=["c"])
    assert set(tables) == {"a", "c", "d"}                 # a via hint routes through c
    tables, _ = join_tree(g, ["a", "d"], hints={("c", "d_id")})
    assert "c" in tables                                   # a join-column hint makes that route cheaper


def test_converter_records_join_hints():
    c = convert("SELECT T3.category FROM customers T1 JOIN orders T2 ON T1.id = T2.customer_id "
                "JOIN items T4 ON T4.order_id = T2.id JOIN products T3 ON T3.id = T4.product_id WHERE T1.name = 'Ann'", G)
    assert c.plan is not None and set(c.plan.require) == {"orders", "items"}
    assert Col("items", "order_id") in c.plan.join_cols


def test_cached_option_phase_trains_question_side_only():
    import json
    import tempfile

    from luthor.sq.data import batches, load_examples
    from luthor.sq.train import TextCache, train_step

    card = _card()
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in _rows():
            f.write(json.dumps(r) + "\n")
    exs = load_examples(Path(f.name), {"shop": card})
    model = _tiny_model()
    cache = TextCache(refresh=5)
    b = next(batches(exs, {"shop": card}, dbs_per_batch=1, q_per_db=8))
    loss, _ = train_step(model, b, cache=cache, step=0)
    loss.backward()
    assert len(cache.vecs) > 0 and all(not v.requires_grad for _, v in cache.vecs.values())
    assert any(p.grad is not None for p in model.q_proj.parameters())   # question side still learns


def test_via_hint_cannot_add_a_dead_end_join():
    from luthor.sq.plan import join_tree

    tables, edges = join_tree(G, ["customers"], via=["orders"])   # orders would be a leaf: pruned
    assert tables == ["customers"] and edges == []
    tables, _ = join_tree(G, ["customers", "products"], via=["orders"])  # a real bridge stays
    assert "orders" in tables and "items" in tables


def test_existence_join_is_kept(db):
    sql = ("SELECT DISTINCT T1.name FROM customers T1 JOIN orders T2 ON T1.id = T2.customer_id "
           "WHERE T2.status = 'cancelled'")
    c = convert(sql, G)
    assert rows(db, compile_plan(c.plan, G)) == rows(db, sql)
    sql2 = "SELECT DISTINCT T1.name FROM customers T1 JOIN orders T2 ON T1.id = T2.customer_id"
    c2 = convert(sql2, G)
    assert c2.plan.require == ["orders"] and rows(db, compile_plan(c2.plan, G)) == rows(db, sql2)
