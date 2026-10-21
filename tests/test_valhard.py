"""VALHARD (scripts/valhard): clause-diff classifier, synthetic edit generator, hard-pair miner, list mixer, group loss,
and build_lists --pool. CPU, no model, a few seconds."""
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/valhard"))
sys.path.insert(0, str(ROOT / "scripts/val2"))
import build_lists as bl  # noqa: E402
import edits as ed  # noqa: E402
import execq  # noqa: E402
import mine  # noqa: E402
import mix  # noqa: E402
import sqldiff as sd  # noqa: E402

P = 'SELECT COUNT(T1."patient") FROM "patients" AS T1 INNER JOIN "meds" AS T2 ON T1."patient" = T2."patient" ' \
    "WHERE T2.\"descr\" = 'Nitro' AND T1.\"gender\" = 'F'"
SCHEMA = {"patients": {"cols": ["patient", "first", "last", "gender"], "pk": ["patient"], "fks": []},
          "meds": {"cols": ["patient", "code", "descr"], "pk": [], "fks": [("patient", "patients", "patient")]},
          "claims": {"cols": ["id", "patient", "amount"], "pk": ["id"], "fks": [("patient", "patients", "patient")]}}


# ---- sqldiff.classify ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("right,wrong,kind", [
    (P, P.replace('COUNT(T1."patient")', 'COUNT(DISTINCT T1."patient")'), "count_distinct"),
    (P, P.replace('COUNT(T1."patient")', 'COUNT(T2."code")'), "count_key"),
    (P, P.replace(" AND T1.\"gender\" = 'F'", ""), "where_drop"),
    (P.replace(" AND T1.\"gender\" = 'F'", ""), P, "where_extra"),
    (P, P.replace("'Nitro'", "'Aspirin'"), "literal"),
    ('SELECT "a" FROM "t" WHERE "x" > 5', 'SELECT "a" FROM "t" WHERE "x" >= 5', "where_op"),
    ('SELECT "a" FROM "t" ORDER BY "b" DESC LIMIT 1', 'SELECT "a" FROM "t" ORDER BY "b" ASC LIMIT 1', "order_dir"),
    ('SELECT "a" FROM "t" ORDER BY "b" DESC LIMIT 1', 'SELECT "a" FROM "t" ORDER BY "b" DESC', "ties"),
    ('SELECT "a" FROM "t" WHERE "b" = (SELECT MAX("b") FROM "t")', 'SELECT "a" FROM "t" ORDER BY "b" DESC LIMIT 1', "ties"),
    ('SELECT "a", "b" FROM "t"', 'SELECT "a" FROM "t"', "proj_drop"),
    ('SELECT "a" FROM "t"', 'SELECT "a", "b" FROM "t"', "proj_extra"),
    ('SELECT "a", "b" FROM "t"', 'SELECT "b", "a" FROM "t"', "proj_order"),
    ('SELECT T1."name" FROM "p" AS T1', 'SELECT T1."person_id" FROM "p" AS T1', "id_name"),
    ('SELECT MAX("a") FROM "t"', 'SELECT MIN("a") FROM "t"', "agg_func"),
    (P, P.replace(' WHERE', ' INNER JOIN "claims" AS T3 ON T1."patient" = T3."patient" WHERE'), "join_fanout"),
    ('SELECT T1."a" FROM "x" AS T1 INNER JOIN "y" AS T2 ON T1."k" = T2."k" WHERE T2."v" = 1',
     'SELECT T1."a" FROM "x" AS T1', "join_filter"),
    ('SELECT "a" FROM "t" WHERE "x" = 1', 'SELECT "c" FROM "t" WHERE "x" = 2', "multi"),
])
def test_classify(right, wrong, kind):
    assert sd.classify(right, wrong)[0] == kind


def test_classify_alias_renumbering_is_not_a_difference():
    a = 'SELECT T1."a" FROM "x" AS T1 INNER JOIN "y" AS T2 ON T1."k" = T2."k"'
    b = 'SELECT T2."a" FROM "x" AS T2 INNER JOIN "y" AS T1 ON T2."k" = T1."k"'
    assert sd.classify(a, b)[0] == "same"
    assert sd.classify("not sql at all (", a)[0] == "unparsed"
    assert sd.is_hard("count_distinct") and not sd.is_hard("ties") and not sd.is_hard("multi")


# ---- edits ---------------------------------------------------------------------------------------------------------

def test_edits_kinds_and_identity_rule():
    out = dict()
    for k, s in ed.edits(P, SCHEMA, seed="t"):
        out.setdefault(k, []).append(s)
    assert out["count_distinct"] == [P.replace('COUNT(T1."patient")', 'COUNT(DISTINCT T1."patient")')]
    assert sd.classify(P, out["count_key"][0])[0] == "count_key"
    assert sd.classify(P, out["add_join"][0])[0] == "join_fanout"
    assert len(out["drop_where"]) == 2 and len(set(out["drop_where"])) == 2
    assert all(sd.classify(P, s)[0] == "where_drop" for s in out["drop_where"])
    assert sd.classify(P, out["drop_join_filter"][0])[0] == "join_filter"
    assert "LIMIT" not in " ".join(s for v in out.values() for s in v)  # LIMIT is never added or removed
    # not a sqlglot identity round trip (lower-case keywords) -> no edits at all
    assert ed.edits(P.replace("SELECT", "select"), SCHEMA) == []


def test_edits_deterministic_and_seeded():
    assert ed.edits(P, SCHEMA, seed="a") == ed.edits(P, SCHEMA, seed="a")


def test_flip_order_prints_no_nulls_clause():
    for q in ('SELECT T1."a" FROM "t" AS T1 ORDER BY T1."b" ASC', 'SELECT T1."a" FROM "t" AS T1 ORDER BY T1."b" DESC LIMIT 1'):
        (k, s), = ed.edits(q, {}, kinds=("flip_order",))
        assert "NULLS" not in s and sd.classify(q, s)[0] in ("order_dir",)


def test_limit_without_order_is_not_edited():
    assert ed.edits('SELECT T1."a", T1."b" FROM "t" AS T1 WHERE T1."c" = 1 LIMIT 3', {}) == []


def test_projection_edits():
    q = 'SELECT T1."patient", T1."gender" FROM "patients" AS T1 ORDER BY T1."last" ASC'
    got = dict(ed.edits(q, SCHEMA, seed="p", kinds=("proj_drop", "proj_add", "proj_reorder", "id_name")))
    assert sd.classify(q, got["proj_drop"])[0] == "proj_drop"
    assert got["proj_add"].startswith('SELECT T1."patient", T1."gender", T1."last" FROM')
    assert got["proj_reorder"].startswith('SELECT T1."gender", T1."patient" FROM')
    assert got["id_name"].startswith('SELECT T1."first", T1."gender"')  # pk -> name-like column of the same table


def _db(tmp_path):
    p = tmp_path / "d.sqlite"
    con = sqlite3.connect(p)
    con.executescript("""
      CREATE TABLE patients (patient TEXT PRIMARY KEY, first TEXT, last TEXT, gender TEXT);
      CREATE TABLE meds (patient TEXT REFERENCES patients(patient), code INT, descr TEXT);
      INSERT INTO patients VALUES ('p1','A','X','F'),('p2','B','Y','F'),('p3','C','Z','M');
      INSERT INTO meds VALUES ('p1',1,'Nitro'),('p1',2,'Nitro'),('p2',3,'Nitro'),('p3',4,'Nitro');""")
    con.commit()
    con.close()
    return str(p)


def test_edits_execute_and_keep(tmp_path):
    path = _db(tmp_path)
    sch = ed.schema_of(path)
    assert sch["meds"]["fks"] == [("patient", "patients", "patient")] and sch["patients"]["pk"] == ["patient"]
    gold = 'SELECT COUNT(DISTINCT T1."patient") FROM "patients" AS T1 INNER JOIN "meds" AS T2 ON T1."patient" = T2."patient" ' \
           "WHERE T2.\"descr\" = 'Nitro' AND T1.\"gender\" = 'F'"
    es = ed.edits(gold, sch, seed="x")
    res = execq.run_tasks({"d": [("q", gold, [s for _, s in es])]}, {"d": path}, 5, 5, workers=1)
    status, gsig, rr = res["q"]
    assert status == "ok"
    kept, st = mine.keep_edits([(k, s, x) for (k, s), x in zip(es, rr)], {gsig})
    kinds = {e["kind"] for e in kept}
    assert "count_distinct" in kinds  # COUNT(patient) = 3 rows (p1 twice) vs 2 distinct
    assert all(not e["ok"] and e["sig"] != gsig for e in kept)
    assert len({e["sig"] for e in kept}) == len(kept)  # one per new result
    assert sum(v for k, v in st.items() if k.endswith(":kept")) == len(kept)


# ---- miner ---------------------------------------------------------------------------------------------------------

def c(rank, sql, sig, ok=False, runs=True, nonnull=True):
    return {"rank": rank, "cost": float(rank), "sql": sql, "runs": runs, "nonnull": nonnull, "ok": ok, "sig": sig,
            "res": "1 row"}


R0 = 'SELECT COUNT(DISTINCT "a") FROM "t" WHERE "x" = 1'
W1 = 'SELECT COUNT("a") FROM "t" WHERE "x" = 1'                  # count_distinct
W2 = 'SELECT COUNT(DISTINCT "a") FROM "t"'                        # where_drop
W3 = 'SELECT "b" FROM "u"'                                         # multi
W4 = 'SELECT COUNT(DISTINCT "a") FROM "t" WHERE "x" = 2'          # literal, cut by the cap (not in the list)


def test_mine_list_labels_and_cap_cut():
    union = [c(0, R0, "r", True), c(1, W1, "w1"), c(2, W2, "w2"), c(3, W3, "w3"), c(4, W4, "w4"),
             c(5, 'SELECT COUNT(DISTINCT "a") FROM "t" WHERE "x" = 1 AND 1 = 1', "r", True),
             c(6, W1 + " AND 1 = 1", "w1"), c(7, 'SELECT "zz" FROM "t"', None, runs=False)]
    rec = {"id": "bird_train_train_2", "gold_corrected": False,
           "cands": [c(0, R0, "r", True), c(1, W1, "w1"), c(2, W2, "w2"), c(3, W3, "w3"),
                     dict(c(100, 'SELECT 1', "m"), mutant=True)]}
    got = {p["wrong_sig"]: p for p in mine.mine_list(rec, union, {})}
    assert set(got) == {"w1", "w2", "w3", "w4"}
    assert got["w1"]["type"] == "count_distinct" and got["w1"]["right_sql"] == R0 and got["w1"]["wrong_sql"] == W1
    assert got["w2"]["type"] == "where_drop" and got["w3"]["type"] == "multi"
    assert got["w4"]["type"] == "literal" and got["w4"]["in_list"] is False and got["w1"]["in_list"] is True
    # corrected gold: a class cut by the cap has only a stale gen-time label -> skipped
    got2 = {p["wrong_sig"] for p in mine.mine_list(dict(rec, gold_corrected=True), union, {})}
    assert got2 == {"w1", "w2", "w3"}
    # the list's (relabelled) label wins over the raw pool's: w2 right in the list -> no pair for it
    rec3 = dict(rec, cands=[dict(x, ok=True) if x["sig"] == "w2" else x for x in rec["cands"]])
    assert "w2" not in {p["wrong_sig"] for p in mine.mine_list(rec3, union, {})}


def test_edit_source_falls_back_to_a_round_tripping_member():
    rec = {"cands": [c(0, R0.replace("SELECT", "select"), "r", True), c(1, W1, "w1")]}
    union = [c(0, R0.replace("SELECT", "select"), "r", True), c(3, R0, "r", True)]
    assert mine.edit_source(rec, union) == R0
    assert mine.edit_source({"cands": [c(1, W1, "w1")]}, union) is None


# ---- mixer ---------------------------------------------------------------------------------------------------------

def _args(**kw):
    d = dict(use_hard=True, max_edits=3, max_mined_add=2, pair_members=True, max_pair_members=2, drop_mutations=["limit"])
    d.update(kw)
    return SimpleNamespace(**d)


def test_mix_one():
    rec = {"id": "q", "db_id": "d", "weight": 1.0,
           "cands": [c(0, R0, "r", True), c(1, W1 + " ", "w1"), dict(c(100, R0 + " LIMIT 1", "m1"), mutant=True, mutation="limit"),
                     dict(c(101, R0 + " * 100", "m2"), mutant=True, mutation="x100")]}
    hard = {"mined": [
        {"type": "count_distinct", "ncomp": 1, "right_rank": 3, "wrong_rank": 6, "right_sql": "R_ALT", "right_sig": "r",
         "wrong_sql": "W1_ALT", "wrong_sig": "w1", "in_list": True},
        {"type": "literal", "ncomp": 1, "right_rank": 0, "wrong_rank": 4, "right_sql": R0, "right_sig": "r",
         "wrong_sql": W4, "wrong_sig": "w4", "in_list": False, "wrong_res": "1 row"},
        {"type": "multi", "ncomp": 2, "right_rank": 0, "wrong_rank": 9, "right_sql": R0, "right_sig": "r",
         "wrong_sql": W3, "wrong_sig": "w3", "in_list": False},
        {"type": "ties", "ncomp": 1, "right_rank": 0, "wrong_rank": 9, "right_sql": R0, "right_sig": "r",
         "wrong_sql": "T", "wrong_sig": "wt", "in_list": False}],
        "edits": [{"kind": k, "bucket": b, "sql": f"E_{k}", "sig": f"e_{k}", "ok": False, "res": "r"}
                  for k, b in (("proj_add", "proj"), ("drop_where", "sem"), ("flip_order", "sem"),
                               ("count_distinct", "grain"), ("count_key", "grain"))]
        + [{"kind": "proj_drop", "bucket": "proj", "sql": "E_dup", "sig": "w1", "ok": False}]}
    out, st = mix.mix_one(rec, hard, _args())
    by = {x["sql"]: x for x in out["cands"]}
    assert R0 + " LIMIT 1" not in by and R0 + " * 100" in by                      # limit mutant dropped, x100 kept
    assert by[W4]["prov"] == "mined" and by[W4]["ok"] is False and "T" not in by and W3 not in by  # hard only, no ties
    assert by["W1_ALT"]["prov"] == "pair" and by["W1_ALT"]["sig"] == "w1" and by["W1_ALT"]["ok"] is False
    assert by["R_ALT"]["prov"] == "pair" and by["R_ALT"]["sig"] == "r" and by["R_ALT"]["ok"] is True
    edits = [x["mutation"] for x in out["cands"] if x["prov"] == "edit"]
    assert edits == ["count_distinct", "drop_where", "proj_add"]                    # one per bucket, KINDS order
    assert "E_dup" not in by                                                         # an edit with a list result: never
    # default (no --hard): only the mutant filter applies
    out0, _ = mix.mix_one(rec, hard, _args(use_hard=False))
    assert [x["sql"] for x in out0["cands"]] == [R0, W1 + " ", R0 + " * 100"]


# ---- group loss ----------------------------------------------------------------------------------------------------

def test_group_loss():
    torch = pytest.importorskip("torch")
    from luthor.sq2.rerank import group_loss, multipos_loss
    g = torch.Generator().manual_seed(0)
    s = torch.randn(6, generator=g)
    ok = [True, False, False, True, False, False]
    # singleton groups == multipos (with and without smoothing)
    for sm in (0.0, 0.05):
        assert float(group_loss(s, ok, list(range(6)), sm)) == pytest.approx(float(multipos_loss(s, ok, sm)), abs=1e-6)
    # a duplicated wrong member with the same score adds no mass (log-mean-exp), unlike multipos
    s2 = torch.tensor([1.0, 2.0, 2.0])
    assert float(group_loss(s2, [True, False, False], ["r", "w", "w"])) == pytest.approx(
        float(multipos_loss(s2[:2], [True, False])), abs=1e-6)
    assert float(multipos_loss(s2, [True, False, False])) > float(multipos_loss(s2[:2], [True, False]))
    # gradient reaches every member
    s3 = torch.tensor([0.5, 1.0, 0.2, -0.3], requires_grad=True)
    group_loss(s3, [True, True, False, False], ["r", "r", "w", "w"]).backward()
    assert all(abs(float(x)) > 0 for x in s3.grad)


def test_trainer_question_loss_group():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "scripts"))
    import rerank_sq2 as R
    s = torch.tensor([1.0, 2.0, 2.0])
    a = SimpleNamespace(loss="group", label_smooth=0.0)
    assert float(R.question_loss(a, s, [True, False, False], ["r", "w", "w"])) == pytest.approx(
        float(R.question_loss(SimpleNamespace(loss="multipos", label_smooth=0.0), s[:2], [True, False])), abs=1e-6)


# ---- build_lists --pool --------------------------------------------------------------------------------------------

def test_load_pools_specs(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"id": "bird_train_train_1", "cands": []}) + "\n")
    got = bl.load_pools([f"r1={p}"])
    assert list(got) == ["r1"] and list(got["r1"]) == ["bird_train_train_1"]


def test_db_step_v2_group_loss_gradcache_matches_one_pass(monkeypatch):
    """--loss group through the real trainer step (tiny Qwen2 + LoRA): chunked gradient cache == one pass, and with
    shared result signatures it differs from multipos."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import rerank_sq2 as R
    import test_rerank_gradcache as T
    from test_sq2_r1_levers import _card
    recs = T._recs()
    for r in recs:  # result signatures: q2's two right COUNTs share one; q3's two wrong ones share one
        for k, c_ in enumerate(r["cands"]):
            c_["sig"] = f"{r['id']}_{k}"
    recs[1]["cands"][1]["sig"] = recs[1]["cands"][0]["sig"]
    recs[2]["cands"][2]["sig"] = recs[2]["cands"][1]["sig"]
    monkeypatch.setattr(T, "_recs", lambda: recs)

    def step(rr, budget, loss):
        a = SimpleNamespace(budget_chars=9000, tok_budget=budget, loss=loss, label_smooth=0.05)
        rr.zero_grad(set_to_none=True)
        stats = R.new_stats()
        l, n = R.db_step_v2(rr, _card(), recs, a, 1.0 / 3, 1.5, stats)
        return l, {k: p.grad.clone() for k, p in rr.named_parameters() if p.grad is not None}, stats

    rr = T._rr()
    l0, g0, s0 = step(rr, 10**9, "group")
    l1, g1, s1 = step(rr, 1, "group")
    assert s0["direct"] == 1 and s1["gradcache"] == 1
    assert l1 == pytest.approx(l0, rel=1e-5)
    T._same(g0, g1)
    lm, _, _ = step(rr, 10**9, "multipos")
    assert abs(lm - l0) > 1e-4
