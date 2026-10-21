"""Repair stage (scripts/repair): the CPU evidence parser / coverage check, pool triggers, failing-candidate bans, the
constrained Scores wrapper, and constrained lazy k-best on a fake score model (no torch model, no database). Seconds."""
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "repair"))
sys.path.insert(0, str(ROOT / "src"))

import evidence as E  # noqa: E402
import repair_gen as RG  # noqa: E402
from luthor.sq.candidates import Candidate  # noqa: E402
from luthor.sq2 import targets as T  # noqa: E402
from luthor.sq2.topk import candidates, kbest  # noqa: E402


def _kinds(atoms):
    return [(a.kind, a.col, a.op, a.values) for a in atoms]


# ---------------------------------------------------------------------------------------------------- parser
def test_refers_to_equality_and_spaced_operators():
    at = E.parse_evidence("canned refers to ContainerType = 'Can'; profit of no less than 1,000 refers to Profit > = 1000")
    assert ("cmp", "containertype", "=", ("can",)) in _kinds(at)
    assert ("cmp", "profit", ">=", ("1000",)) in _kinds(at)


def test_in_null_between():
    at = E.parse_evidence("organ IN ('army','peace_corps','foreign_legion'); no catch refers to shootCatch IS NULL; "
                          "released refers to release_date BETWEEN '1990-01-01' AND '1995-12-31'; "
                          "alive refers to deathYear IS NOT NULL")
    k = _kinds(at)
    assert ("in", "organ", "in", ("army", "peace_corps", "foreign_legion")) in k
    assert ("null", "shootcatch", "is_null", ()) in k
    assert ("null", "deathyear", "not_null", ()) in k
    assert ("between", "release_date", "between", ("1990-01-01", "1995-12-31")) in k


def test_formulas_divide_percentage_multiply_constant():
    at = E.parse_evidence("percentage = (divide(count(paper_id where class_label = 'Agents')), (count(paper_id)))*100")
    k = _kinds(at)
    assert ("arith", None, "/", ()) in k and ("const", None, None, ("100",)) in k
    assert ("cmp", "class_label", "=", ("agents",)) in k
    at = E.parse_evidence("percentage = MULTIPLY(DIVIDE(SUM(x), COUNT(y)), 100)")
    assert ("const", None, None, ("100",)) in _kinds(at)
    at = E.parse_evidence("average per month = DIVIDE(SUM(amount), 9)")
    assert ("const", None, None, ("9",)) in _kinds(at)
    at = E.parse_evidence("old refers to SUBTRACT(2008, strftime('%Y', DOB))")
    assert ("arith", None, "-", ()) in _kinds(at) and ("const", None, None, ("2008",)) in _kinds(at)


def test_function_left_side_and_numbers_normalised():
    at = E.parse_evidence("shipment in 2017 refers to Cast(ship_date AS DATE) = 2017; in 2007 refers to year = '2007'")
    k = _kinds(at)
    assert ("cmp", "ship_date", "=", ("2017",)) in k
    assert ("cmp", "year", "=", ("2007",)) in k  # '2007' and 2007 normalise alike
    assert E.norm_lit("2007") == E.norm_lit(2007) == E.norm_lit(2007.0) == "2007"


def test_question_marks_unused_definitions_optional():
    ev = "male refers to Gender = 'M'; female refers to Gender = 'F'"
    at = E.parse_evidence(ev, question="How many female students are there?")
    req = {a.values[0]: a.required for a in at}
    assert req == {"m": False, "f": True}  # 'female' does not count as a use of 'male'


def test_schema_columns_and_multiword_names():
    cols = {"content rating", "app", "rating"}
    at = E.parse_evidence("percentage = DIVIDE(SUM(Content Rating = 'Teen'), COUNT(App))", cols)
    k = _kinds(at)
    assert ("cmp", "content rating", "=", ("teen",)) in k
    assert not any(a.kind == "col" and a.col == "app" for a in at)  # COUNT(App) == COUNT(*): not required
    # with a schema, an unknown left side over a bare number is skipped (a computed quantity, not a column)
    assert E.parse_evidence("old refers to age > 65", {"birthdate"}) == []


# ---------------------------------------------------------------------------------------------------- coverage
def test_coverage_equality_and_year_like():
    at = E.parse_evidence("in 2006 refers to release_date LIKE '2006%'; American refers to country = 'USA'")
    good = "SELECT AVG(revenue) FROM movie WHERE STRFTIME('%Y', release_date) = '2006' AND country = 'USA'"
    bad = "SELECT AVG(revenue) FROM movie WHERE country = 'USA'"
    assert E.coverage(at, good)["covers"]
    c = E.coverage(at, bad)
    assert not c["covers"] and c["frac"] == 0.5


def test_coverage_null_in_and_ops():
    at = E.parse_evidence("shootCatch IS NULL")
    assert E.coverage(at, 'SELECT 1 FROM Master WHERE "shootCatch" IS NULL')["covers"]
    assert not E.coverage(at, 'SELECT 1 FROM Master WHERE NOT "shootCatch" IS NULL')["covers"]
    at = E.parse_evidence("organ IN ('army','peace_corps')")
    assert E.coverage(at, "SELECT 1 FROM e WHERE organ IN ('army', 'peace_corps')")["covers"]
    assert E.coverage(at, "SELECT 1 FROM e WHERE organ = 'army' OR organ = 'peace_corps'")["covers"]
    assert not E.coverage(at, "SELECT 1 FROM e WHERE organ IN ('army')")["covers"]
    at = E.parse_evidence("after May refers to month > 5")
    assert E.coverage(at, "SELECT 1 FROM t WHERE month > 5")["covers"]
    assert E.coverage(at, "SELECT 1 FROM t WHERE month >= 6")["covers"]  # integer shift
    assert not E.coverage(at, "SELECT 1 FROM t WHERE month >= 5")["covers"]  # the 4435 miss
    assert E.coverage(at, "SELECT 1 FROM t WHERE 5 < month")["covers"]  # flipped sides


def test_coverage_arithmetic():
    at = E.parse_evidence("percentage = DIVIDE(SUM(num_students * pct_female), 100)")
    assert E.coverage(at, "SELECT SUM(num_students * pct_female) / 100 FROM u")["covers"]
    assert not E.coverage(at, "SELECT SUM(num_students * pct_female) FROM u")["covers"]  # the /100 blind spot
    assert E.coverage(at, "SELECT SUM(num_students * pct_female) * 0.01 / 1 FROM u")["covers"]
    at = E.parse_evidence("average = DIVIDE(SUM(Extra_Runs), COUNT(Extra_Runs))")
    assert E.coverage(at, 'SELECT AVG("Extra_Runs") FROM t')["covers"]  # AVG implements SUM / COUNT


def test_coverage_aliases_and_unparseable_sql():
    at = E.parse_evidence("Zimbabwea refers to Country_Name = 'Zimbabwea'")
    sql = 'SELECT T1."x" FROM "A" AS T1 INNER JOIN "Country" AS T3 ON 1 = 1 WHERE T3."Country_Name" = \'Zimbabwea\''
    assert E.coverage(at, sql)["covers"]
    f = E.sql_facts(sql)
    assert ("country.country_name", "=", ("zimbabwea",)) in f.qpreds and f.select_tables == {"a"}
    assert E.coverage(at, None) == {"covered": [False], "frac": 0.0, "covers": False, "missing": [0]}
    assert E.coverage([], "SELECT 1")["frac"] is None


def test_auc():
    assert E.auc([1, 1, 0, 0], [True, True, False, False]) == 1.0
    assert E.auc([0, 0, 1, 1], [True, True, False, False]) == 0.0
    assert E.auc([1, 1], [True, False]) == 0.5
    assert E.auc([1], [True]) is None


def test_guard_refuses_dev_and_sealed():
    for bad in ("runs/x/cands_bird_dev.jsonl", "data/gate.json", "data/sealed/x.json", "data/eval/sel_cands/sel/score_a"):
        with pytest.raises(SystemExit):
            E.guard(bad)


# ---------------------------------------------------------------------------------------------------- triggers
def _c(sql, runs=True, nonnull=True, res="3 rows x 1 col (text); first: ('a')", sig="s1", ok=False, rank=0):
    return {"sql": sql, "runs": runs, "nonnull": nonnull, "res": res, "sig": sig, "ok": ok, "rank": rank}


def test_triggers_degenerate_all_same_no_live_evidence():
    zero = dict(res="1 row x 1 col (integer); first: (0)")
    rec = {"id": "q", "question": "how many", "context": "", "cands": [
        _c("SELECT COUNT(*) FROM a WHERE x = 'v'", sig="z", **zero),
        _c("SELECT COUNT(*) FROM a WHERE y = 'v'", sig="e", nonnull=False, res="0 rows"),
        _c("SELECT oops", runs=False, nonnull=False, res=None, sig=None)]}
    assert RG.pool_state(rec)["degenerate"] and RG.triggers(rec) == ["degenerate"]
    rec["cands"][1] = _c("SELECT COUNT(*) FROM a", sig="n", res="1 row x 1 col (integer); first: (12)")
    assert RG.triggers(rec) == []
    same = {"id": "q", "question": "q", "context": "", "cands": [_c("SELECT a FROM t"), _c("SELECT DISTINCT a FROM t")]}
    assert RG.triggers(same) == ["all_same"]
    dead = {"id": "q", "question": "q", "context": "", "cands": [_c("SELECT", runs=False, nonnull=False, sig=None)]}
    assert RG.triggers(dead) == ["no_live"]
    ev = {"id": "q", "question": "how many goalies have no catch", "context": "no catch refers to shootCatch IS NULL",
          "cands": [_c("SELECT a FROM t", sig="1"), _c("SELECT b FROM t", sig="2")]}
    assert RG.triggers(ev) == ["evidence"]
    ev["cands"][1]["sql"] = "SELECT b FROM t WHERE shootCatch IS NULL"
    assert RG.triggers(ev) == []
    ev["cands"][1]["rank"] = 9  # only a candidate past the top ranks covers it: still triggers
    assert RG.triggers(ev, ev_top=8) == ["evidence"]


def test_failing_constraints_map_to_options():
    ids = ["patients.*", "patients.first", "patients.birthdate", "allergies.*", "allergies.DESCRIPTION",
           "careplans.*", "careplans.ID"]
    rec = {"cands": [
        _c('SELECT T1."first" FROM "patients" AS T1 INNER JOIN "allergies" AS T2 ON 1 = 1 WHERE T2."DESCRIPTION" = \'Influenza\'',
           nonnull=False, res="0 rows"),
        _c('SELECT T1."first" FROM "patients" AS T1 INNER JOIN "allergies" AS T2 ON 1 = 1 WHERE T1."birthdate" = \'Major\'',
           nonnull=False, res="0 rows")]}
    fc = RG.failing_constraints(rec, ids)
    assert fc["vals"] == {"influenza", "major"}
    assert {ids[i] for i in fc["cols"]} == {"allergies.DESCRIPTION", "patients.birthdate"}
    assert {ids[i] for i in fc["tables"]} == {"allergies.*"}  # joined by every failing candidate, never projected


# ---------------------------------------------------------------------------------------------------- decoding
class FakeScores:
    """Deterministic pseudo-random score vectors of the right sizes for targets.decode (no model)."""

    def __init__(self, option_ids, cands, seed=0):
        self.option_ids, self.cands, self.seed = option_ids, cands, seed

    def _v(self, key, n):
        r = random.Random(f"{self.seed}|{key}")
        return [r.gauss(0, 2) for _ in range(n)]

    def choice(self, key):
        name = key.split(".", 1)[1]
        names = T.MAIN_CHOICES.get(name) if key.startswith("m.") and name in T.MAIN_CHOICES else T.CHOICES[name]
        xs = self._v(("c", key), len(names))
        if name == "n_filter":
            xs[1] += 6  # mostly one filter, so the tests are about which one
        if name in ("set_op", "n_having", "grain", "compose", "n_via", "n_join", "n_require", "n_group"):
            xs[0] += 8
        return xs

    def ptr(self, slot):
        return self._v(("p", slot), len(self.option_ids))

    def pair(self, slot, opt, head):
        xs = self._v(("q", slot, opt, head), len(T.PAIRS[head]))
        if head in ("sub", "or", "part"):
            xs[0] += 8
        if head == "op":
            xs[0] += 4  # '=' usually
        return xs

    def vals(self, slot, opt):
        return self._v(("v", slot, opt), len(self.cands))

    def limit(self, slot):
        return self._v(("l", slot), len(self.cands))


IDS = ["t.*", "t.a", "t.b", "t.c", "u.*", "u.d", "u.e"]
CANDS = [Candidate("value", v, "t.a", None) for v in ("x", "y", "z", "Influenza")]


def _filters(p):
    return [(f.col.id, tuple(f.values), f.op) for f in p.filters]


def test_constrained_scores_ban_and_boost_vectors():
    sc = FakeScores(IDS, CANDS)
    cs = RG.ConstrainedScores(sc, ban_vals={"influenza"}, ban_ptr={RG.FILTER_SLOT.pattern: {1}},
                              boost_ptr={RG.FILTER_SLOT.pattern: {5}}, boost_ops={5: "is_null"},
                              min_choice={"m.n_filter": 2}, boost=30)
    assert cs.ptr("m.filter")[1] < -1000 and cs.ptr("m.p0")[1] == sc.ptr("m.p0")[1]  # bans only filter slots
    assert cs.ptr("m.filter")[5] == pytest.approx(sc.ptr("m.filter")[5] + 30)
    assert cs.vals("m.filter", 2)[3] < -1000 and cs.vals("m.filter", 2)[0] == sc.vals("m.filter", 2)[0]
    xs = cs.choice("m.n_filter")
    assert max(range(len(xs)), key=xs.__getitem__) >= 2
    op = cs.pair("m.filter", 5, "op")
    assert max(range(len(op)), key=op.__getitem__) == T.OPS.index("is_null")


def test_constrained_kbest_respects_bans_and_forces():
    base = None
    for seed in range(50):  # a seed whose greedy plan filters on t.a with a literal
        sc = FakeScores(IDS, CANDS, seed)
        g = T.decode(sc)
        if any(c == "t.a" and v for c, v, _ in _filters(g)):
            base = sc
            break
    assert base is not None
    lit = next(v[0] for c, v, _ in _filters(g) if c == "t.a" and v)
    banned = RG.ConstrainedScores(base, ban_vals={lit})
    for p, _ in kbest(banned, 8, repr, with_cost=True):
        assert all(lit not in v for _, v, _ in _filters(p))
    nocol = RG.ConstrainedScores(base, ban_ptr={RG.FILTER_SLOT.pattern: {IDS.index("t.a")}})
    for p, _ in kbest(nocol, 8, repr, with_cost=True):
        assert all(c != "t.a" for c, _, _ in _filters(p))
    forced = RG.ConstrainedScores(base, boost_ptr={RG.FILTER_SLOT.pattern: {IDS.index("u.e")}},
                                  boost_ops={IDS.index("u.e"): "is_null"})
    g2 = T.decode(forced)
    assert ("u.e", (), "is_null") in _filters(g2)


def test_repair_plans_dedupes_against_pool_and_tags_variants():
    sc = FakeScores(IDS, CANDS, 3)
    g = T.decode(sc)
    rec = {"id": "q", "question": "how many have no e", "context": "no e refers to e IS NULL",
           "cands": [_c(repr(g), nonnull=False, res="0 rows", sig="0")]}
    trig = RG.triggers(rec)
    assert trig == ["degenerate", "evidence"]
    atoms = E.parse_evidence(rec["context"], None, rec["question"])
    missing = RG._missing_in_top(atoms, rec, 8)
    out = RG.repair_plans(rec, trig, sc, repr, k_rep=4, hot_samples=6, atoms=atoms, missing=missing, max_extra=40)
    sqls = [s for _, s, _ in out]
    assert repr(g) not in sqls and len(sqls) == len(set(sqls)) and sqls
    names = {v for v, _, _ in out}
    assert names <= {"F1:ban_vals", "F2:ban_cols", "F3:ban_tables", "F4:hot", "E1:force", "E2:force+count", "E3:arith"}
    assert any(v.startswith("E1") for v in names)
    assert any("'e'" in s and "is_null" in s for v, s, _ in out if v.startswith("E"))  # the clause was forced in
    assert len(RG.repair_plans(rec, trig, sc, repr, k_rep=4, hot_samples=6, atoms=atoms, missing=missing,
                               max_extra=3)) == 3


def test_k32_distinct_list_is_a_prefix_of_k64():
    """(a): --k 64 --distinct extends the k=32 list (same lazy k-best order) whenever k-best alone fills k=32."""
    n_checked = 0
    for seed in range(12):
        sc = FakeScores(IDS, CANDS, seed)
        k32 = [repr(p) for p in candidates(sc, k=32, distinct=True, compile_fn=repr)]
        k64 = [repr(p) for p in candidates(sc, k=64, distinct=True, compile_fn=repr)]
        kb = kbest(sc, 32, repr)
        if len(kb) < 32:
            continue  # sampled fill-in: order may differ (documented)
        n_checked += 1
        assert k64[:32] == k32
    assert n_checked > 0


def test_merge_appends_repair_candidates(tmp_path):
    import json
    pool = tmp_path / "pool.jsonl"
    extra = tmp_path / "extra.jsonl"
    out = tmp_path / "out.jsonl"
    pool.write_text(json.dumps({"id": "a", "cands": [{"sql": "S1", "rank": 0}]}) + "\n" +
                    json.dumps({"id": "b", "cands": [{"sql": "S2", "rank": 0}]}) + "\n")
    extra.write_text(json.dumps({"id": "a", "cands": [{"sql": "S1"}, {"sql": "S3", "repair": "F1:ban_vals"}]}) + "\n")
    RG.main(["merge", "--pool", str(pool), "--extra", str(extra), "--out", str(out)])
    rows = [json.loads(l) for l in open(out)]
    assert [c["sql"] for c in rows[0]["cands"]] == ["S1", "S3"]
    assert rows[0]["cands"][1]["src"] == "repair" and rows[0]["cands"][1]["rank"] == 1
    assert rows[1]["cands"] == [{"sql": "S2", "rank": 0}]


def test_merge_add_missing_keeps_k32_only_questions(tmp_path):
    """k64 pool step: a question the k=64 run dropped (gold timeout) is kept whole from the k=32 pool, src=k32."""
    import json
    k64 = tmp_path / "k64.jsonl"
    k32 = tmp_path / "k32.jsonl"
    out = tmp_path / "out.jsonl"
    k64.write_text(json.dumps({"id": "a", "cands": [{"sql": "S1", "rank": 0}]}) + "\n")
    k32.write_text(json.dumps({"id": "a", "cands": [{"sql": "S1", "src": "r1"}, {"sql": "S4", "src": "both"}]}) + "\n" +
                   json.dumps({"id": "b", "db_id": "x", "cands": [{"sql": "S2", "src": "r1"}, {"sql": "S2"}]}) + "\n")
    RG.main(["merge", "--pool", str(k64), "--extra", str(k32), "--src", "k32", "--add-missing", "--out", str(out)])
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(out)}
    assert [(c["sql"], c.get("src")) for c in rows["a"]["cands"]] == [("S1", None), ("S4", "k32")]
    assert rows["b"]["db_id"] == "x" and [(c["sql"], c["src"], c["rank"]) for c in rows["b"]["cands"]] == [("S2", "k32", 0)]
    RG.main(["merge", "--pool", str(k64), "--extra", str(k32), "--src", "k32", "--out", str(out)])
    assert [json.loads(l)["id"] for l in open(out)] == ["a"]  # without the flag: pool questions only (old behaviour)


def test_gen_cli_has_ids_and_blind():
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        RG.main(["gen", "--help"])
    assert "--ids" in buf.getvalue() and "--blind" in buf.getvalue()


def test_tiny_slot_model_repair_compiles_real_sql():
    """The box path on CPU: a tiny random Qwen2 slot model's TorchScores, wrapped and decoded by repair_plans, gives
    compiled SQL over a real schema card that avoids the banned literal and is new to the pool."""
    import torch
    from _global_state import isolated
    from luthor.sq.plan import compile_plan
    from luthor.sq2.model import TorchScores
    import luthor.sq2.train as tr
    from test_sq2_r1_levers import _card, _rows, _tiny
    with isolated(seed=0):
        from luthor.sq.schema_card import SchemaCard
        full = _card()  # without the FK-less `codes` table, so the random model's plans compile
        card = SchemaCard(full.db_id, full.dialect, [t for t in full.tables if t.name != "codes"], full.fks,
                          {k: v for k, v in full.values.items() if not k.startswith("codes.")})
        exs = [tr.make_example(r, card, random.Random(0), order="v2") for r in _rows() if r["id"] != "b"]
        m = _tiny(value_side=True)
        m.eval()
        g = card.graph()
        with torch.no_grad():
            st, _, enc = tr.encode_db(m, card, exs)
            greedy = sc = None
            for x, e in zip(enc, exs):  # a question whose random-model plans compile (the codes table has no FK)
                # a random model's plans are garbage (compose, grain, join hints...): pin those heads to "none" with
                # the wrapper itself, so the repair below runs on plans that compile
                simple = {f"m.{k}": {0} for k in ("compose", "grain", "set_op", "n_group", "n_via", "n_join", "n_require",
                                                  "cond_mask", "order_dir", "n_having", "limit_kind", "distinct",
                                                  "wrap", "cond2_mask", "offset", "order2_dir")}
                simple["m.n_filter"] = {1}  # one filter with a literal, for F1 to ban
                s = RG.ConstrainedScores(TorchScores(m, x, st.option_ids, e.cands), boost_choice=simple, boost=50)
                for p in [T.decode(s)] + kbest(s, 8):
                    try:
                        greedy, sc, ex = compile_plan(p, g), s, e
                        break
                    except Exception:
                        continue
                if greedy:
                    break
            assert greedy
            rec = {"id": "a", "question": ex.question, "context": "Egypt refers to country = 'Egypt'",
                   "cands": [_c(greedy, nonnull=False, res="0 rows", sig="0")]}
            trig = RG.triggers(rec, {"country", "amount", "status", "name", "id", "customer_id"})
            assert "degenerate" in trig
            atoms = E.parse_evidence(rec["context"], None, rec["question"])
            out = RG.repair_plans(rec, trig, sc, lambda p: compile_plan(p, g), k_rep=4, hot_samples=4, atoms=atoms,
                                  missing=RG._missing_in_top(atoms, rec, 8))
    assert out and all(s != greedy for _, s, _ in out)
    banned = {v for _, _, vs in E.sql_facts(greedy).preds for v in vs}
    f1 = [s for v, s, _ in out if v.startswith("F1")]
    assert f1 and banned
    for s in f1:  # F1 never compares against a literal the failing candidate used
        assert not ({v for _, _, vs in E.sql_facts(s).preds for v in vs} & banned), (s, banned)
    for v, s, _ in out:
        assert s.upper().startswith("SELECT"), (v, s)


def test_decoding_under_grammar_v4():
    """The repair wrapper on a grammar-v4 decoder (LUTHOR_GRAMMAR is read once at import: a child process)."""
    import os
    import subprocess
    if os.environ.get("LUTHOR_GRAMMAR") == "v4":
        pytest.skip("already running under v4")
    env = dict(os.environ, LUTHOR_GRAMMAR="v4")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__, "-k",
                        "constrained or repair_plans or k32"], env=env, capture_output=True, text=True, cwd=ROOT,
                       timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert " passed" in r.stdout
