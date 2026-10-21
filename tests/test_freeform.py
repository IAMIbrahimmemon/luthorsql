"""scripts/freeform: OmniSQL prompt, SQL extraction, gen-time labelling and the merge into a grammar pool (CPU only)."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/freeform"))
import merge  # noqa: E402
import scoreset  # noqa: E402,F401
import prompt as P  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts/val2"))
import execq  # noqa: E402


def make_db(tmp_path) -> str:
    d = tmp_path / "shop"
    d.mkdir()
    p = d / "shop.sqlite"
    con = sqlite3.connect(p)
    con.executescript("""
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, "home city" TEXT);
        CREATE TABLE orders (oid INTEGER PRIMARY KEY, cid INTEGER REFERENCES customer(id), amount REAL);
        INSERT INTO customer VALUES (1, 'Ann', 'Oslo'), (2, 'Bob', 'Rome'), (3, 'Cy', 'Oslo');
        INSERT INTO orders VALUES (10, 1, 5.0), (11, 1, 7.5), (12, 2, 1.0);
    """)
    con.commit()
    con.close()
    return str(p)


def test_prompt_matches_omnisql_layout(tmp_path):
    db = make_db(tmp_path)
    text = P.build(db, "How many customers live in Oslo?", "Oslo refers to `home city` = 'Oslo'")
    assert text.startswith("Task Overview:\nYou are a data science expert.")
    assert "Database Engine:\nSQLite" in text
    assert "CREATE TABLE customer (" in text and "`home city` text" in text
    assert "FOREIGN KEY (cid) REFERENCES customer (id)" in text
    assert "PRIMARY KEY (id)" in text
    assert "Question:\nOslo refers to `home city` = 'Oslo'\nHow many customers live in Oslo?" in text
    assert "example: ['Oslo'" in text  # the quoted literal of the evidence is matched first
    assert text.rstrip().endswith("Take a deep breath and think step by step to find the correct SQL query.")
    assert ",\n);" not in text and ", --" not in text.split("PRIMARY KEY")[0].splitlines()[-1]


def test_extract_sql():
    out = "Steps...\n```sql\n-- count\nSELECT 1;\n```\nthen\n```sql\nSELECT COUNT(*) FROM customer;\n```"
    assert P.extract_sql(out) == "SELECT COUNT(*) FROM customer"
    assert P.extract_sql("no block") is None
    assert P.extract_sql("```sql\nSELECT 1; SELECT 2\n```") is None
    assert P.extract_sql("```sql\nSELECT ';' FROM t\n```") == "SELECT ';' FROM t"


def ns(**kw):
    d = dict(normalize=True, no_gold=False, gold_seconds=5.0, cand_seconds=3.0, workers=1)
    d.update(kw)
    return argparse.Namespace(**d)


GOLD = "SELECT name FROM customer WHERE \"home city\" = 'Oslo'"


def ff_rec(sqls):
    return {"id": "q1", "db_id": "shop", "question": "q", "context": "", "gold_sql": GOLD,
            "cands": [{"rank": i, "sql": s, "src": "ff", "temp": 0.0 if i == 0 else 0.8} for i, s in enumerate(sqls)]}


def test_label_and_merge(tmp_path):
    db = make_db(tmp_path)
    paths = {"shop": db}
    right_grammar = 'SELECT "name" FROM "customer" WHERE "home city" = \'Oslo\''
    wrong_grammar = 'SELECT "name" FROM "customer"'
    pool_lab = execq.run_tasks({"shop": [("q1", GOLD, [right_grammar, wrong_grammar])]}, paths, 5.0, 3.0, workers=1)
    _, _, res = pool_lab["q1"]
    pool = {"id": "q1", "db_id": "shop", "question": "q", "context": "", "gold_sql": GOLD, "in_slots": True,
            "covered": True, "cands": []}
    for i, (s, (runs, nn, ok, sig, prev)) in enumerate(zip([right_grammar, wrong_grammar], res)):
        pool["cands"].append({"rank": i, "cost": float(i), "sql": s, "runs": runs, "nonnull": nn, "ok": ok, "sig": sig,
                              "res": prev, "src": "both"})
    ff = ff_rec([
        right_grammar,                                                 # same text as a grammar cand -> ff_n
        "select name from customer where `home city` = 'Oslo'",       # normalises to the grammar text -> ff_n
        "SELECT name FROM customer WHERE id = 2",                      # new wrong sig
        "SELECT name FROM customer WHERE id = 2",
        "SELECT nope FROM customer",                                   # dead
        None,
    ])
    lab = merge.label([ff], paths, ns())
    st, cl = lab["q1"]
    assert st == "ok" and len(cl) == 6
    assert cl[1]["ok"] and cl[1]["sig"] == pool["cands"][0]["sig"]
    assert cl[1]["sql_raw"].startswith("select") and cl[1]["sql"].startswith('SELECT "name"')  # normalised, same sig
    assert not cl[4]["runs"] and cl[5]["runs"] is False
    m = merge.merge_record(pool, ff, cl, st, max_ff=8, same_sig="drop")
    assert [c["sql"] for c in m["cands"][:2]] == [right_grammar, wrong_grammar]  # pool order kept
    assert m["cands"][0]["ff_n"] == 2 and m["cands"][0]["ff_sig_n"] == 2
    assert len(m["cands"]) == 3
    new = m["cands"][2]
    assert new["src"] == "ff" and new["ff_n"] == 2 and new["ff_sig_n"] == 2 and new["rank"] == 1002
    assert "cost" not in new and new["ok"] is False
    assert m["ff"]["live"] == 4 and m["ff"]["no_sql"] == 1 and m["ff"]["greedy_sig"] == pool["cands"][0]["sig"]
    assert m["ff"]["vote_sig"] == pool["cands"][0]["sig"]  # 2 right vs 2 wrong: tie -> first seen (rank 0)
    other = merge.label([ff_rec([right_grammar, "SELECT name FROM customer WHERE \"home city\" LIKE 'Oslo'"])], paths,
                        ns())["q1"][1]
    assert len(merge.merge_record(pool, ff, other, st, same_sig="drop")["cands"]) == 2  # same sig, other text: dropped
    assert len(merge.merge_record(pool, ff, other, st, same_sig="keep")["cands"]) == 3
    s = merge.summary([m], lab)
    assert s["pool_oracle"] == 100.0 and s["ff_vote"] == 100.0 and s["newly_reachable"] == 0


def test_uncovered_question_becomes_ff_only_record(tmp_path):
    db = make_db(tmp_path)
    ff = ff_rec(["SELECT name FROM customer WHERE \"home city\" = 'Oslo'", "SELECT 1"])
    lab = merge.label([ff], {"shop": db}, ns())
    st, cl = lab["q1"]
    m = merge.merge_record(None, ff, cl, st)
    assert m["in_slots"] is False and m["covered"] is False
    assert [c["src"] for c in m["cands"]] == ["ff", "ff"] and m["cands"][0]["ok"]
    s = merge.summary([m], lab)
    assert s["pool_oracle"] == 0.0 and s["union_oracle"] == 100.0 and s["newly_reachable"] == 1


def test_blind_labelling_has_no_ok(tmp_path):
    db = make_db(tmp_path)
    ff = ff_rec(["SELECT name FROM customer"])
    st, cl = merge.label([ff], {"shop": db}, ns(no_gold=True))["q1"]
    assert st == "blind" and cl[0]["runs"] and cl[0]["ok"] is None and cl[0]["sig"]


def test_gen_prompt_only_cpu(tmp_path):
    import subprocess
    db = make_db(tmp_path)
    qf = tmp_path / "qs.json"
    qf.write_text(json.dumps([{"db_id": "shop", "question": "Who lives in Oslo?", "evidence": "", "SQL": GOLD}]))
    out = tmp_path / "p.jsonl"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/freeform/gen.py"), "--questions", str(qf), "--id-prefix", "t_",
                        "--db-roots", str(Path(db).parent), "--prompt-only", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    row = json.loads(out.read_text().splitlines()[0])
    assert row["id"] == "t_0" and "CREATE TABLE orders" in row["prompt"]


def test_extra_mutants_vote_fields_and_scoreset(tmp_path):
    import scoreset
    db = make_db(tmp_path)
    paths = {"shop": db}
    g = 'SELECT "name" FROM "customer" WHERE "home city" = \'Oslo\''
    pool_lab = execq.run_tasks({"shop": [("q1", GOLD, [g])]}, paths, 5.0, 3.0, workers=1)["q1"][2][0]
    pool = {"id": "q1", "db_id": "shop", "question": "q", "context": "", "gold_sql": GOLD, "in_slots": True,
            "covered": True, "cands": [{"rank": 0, "cost": 0.0, "sql": g, "runs": pool_lab[0], "nonnull": pool_lab[1],
                                        "ok": pool_lab[2], "sig": pool_lab[3], "res": pool_lab[4], "src": "both"}]}
    ff = ff_rec(["SELECT name FROM customer WHERE id = 2", "SELECT name FROM customer WHERE id = 2", g])
    st, cl = merge.label([ff], paths, ns())["q1"]
    m = merge.merge_record(pool, ff, cl, st)
    assert m["ff"]["vote_ok"] is False and m["ff"]["greedy_ok"] is False and m["ff"]["vote_n"] == 2
    mut = {"id": "q1", "db_id": "shop", "gold_sql": GOLD, "cands": [
        {"rank": 200, "sql": 'SELECT "name" FROM "customer"', "src": "mut"},
        {"rank": 201, "sql": g, "src": "mut"},                                  # same text as grammar: nothing added
        {"rank": 202, "sql": "SELECT name FROM customer WHERE id = 2", "src": "mut"}]}  # sig live via ff: dropped
    _, ml = merge.label([mut], paths, ns(normalize=False))["q1"]
    m2 = merge.add_extra(m, ml, cl, "mut", max_extra=0)
    added = [c for c in m2["cands"] if c["src"] == "mut"]
    assert len(added) == 1 and added[0]["rank"] == 2200 and added[0]["ff_sig_n"] == 0 and "cost" not in added[0]
    assert m2["mut"]["added"] == 1 and m2["ff"] == m["ff"]  # the vote never sees mutants
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(m2) + "\n")
    b.write_text(json.dumps(m) + "\n")
    sup = tmp_path / "sup.jsonl"
    assert scoreset.build([b, a], sup)["added_from_other_pools"] == 1
    r = json.loads(sup.read_text())
    for i, c in enumerate(r["cands"]):
        c["rr"] = float(i)
    sup.write_text(json.dumps(r) + "\n")
    st = scoreset.attach(sup, [a, b], "_by_r1", tmp_path / "s")
    assert st["a_by_r1.jsonl"]["missing"] == 0
    got = json.loads((tmp_path / "s/a_by_r1.jsonl").read_text())
    assert [c["rr"] for c in got["cands"]] == [{c["sql"]: c["rr"] for c in r["cands"]}[c["sql"]] for c in got["cands"]]
