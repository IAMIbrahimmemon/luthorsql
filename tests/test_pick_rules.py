import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis/validator"))
import pick_rules as pr  # noqa: E402


def C(sql, rr, cost=0.0, res="2 rows x 1 col (text); first: ('a',)", src=None, runs=True, nonnull=True, **kw):
    return {"sql": sql, "rr": rr, "cost": cost, "res": res, "src": src, "runs": runs, "nonnull": nonnull, **kw}


ZERO = "1 row x 1 col (integer); first: (0,)"


def test_gamma_frozen():
    assert pr.GAMMA == 0.2


def test_degenerate_regex():
    assert pr.degenerate({"res": "1 row x 1 col (integer); first: (0,)"})
    assert pr.degenerate({"res": "1 row x 1 col (real); first: (0.0,)"})
    assert pr.degenerate({"res": "1 row x 1 col (null); first: (None,)"})
    assert not pr.degenerate({"res": "1 row x 1 col (integer); first: (10,)"})
    assert not pr.degenerate({"res": "2 rows x 1 col (integer); first: (0,)"})
    assert not pr.degenerate({"res": "1 row x 2 col (integer); first: (0, 1)"})
    assert not pr.degenerate({"res": None})
    assert not pr.degenerate({})


def test_max_uses_live_only_and_falls_back():
    cs = [C("a", 5.0, runs=False), C("b", 1.0)]
    assert pr.pick_max(cs)["sql"] == "b"
    dead = [C("a", 5.0, runs=False), C("b", 1.0, nonnull=False)]
    assert pr.pick_max(dead)["sql"] == "a"
    assert pr.pick_max([]) is None
    assert pr.pick_max([C("", 9.0), C("x", 0.0)])["sql"] == "x"


def test_missing_rr_is_minus_inf_and_no_mutation():
    cs = [C("a", None), C("b", -3.0)]
    before = copy.deepcopy(cs)
    assert pr.pick_max(cs)["sql"] == "b"
    assert cs == before


def test_r2_drops_degenerate_unless_all():
    cs = [C("z", 3.0, res=ZERO), C("b", 1.0)]
    assert pr.pick_max(cs)["sql"] == "z"
    assert pr.pick_r2(cs)["sql"] == "b"
    alld = [C("z", 3.0, res=ZERO), C("y", 1.0, res="1 row x 1 col (null); first: (None,)")]
    assert pr.pick_r2(alld)["sql"] == "z"


def test_r1_cost_penalty():
    cs = [C("a", 2.0, cost=10.0), C("b", 1.5, cost=0.0)]
    assert pr.pick_max(cs)["sql"] == "a"
    assert pr.pick_r1(cs)["sql"] == "b"  # 2.0 - 2.0 = 0 < 1.5
    assert pr.pick_r1(cs, gamma=0.0)["sql"] == "a"


def test_repair_cost_capped_at_max_nonrepair():
    cs = [C("a", 1.0, cost=0.0), C("b", 0.0, cost=5.0), C("r", 1.5, cost=0.0, src="repair")]
    assert pr.gen_cost(cs[2], cs) == 5.0
    # repair: 1.5 - 1.0 = 0.5 < a: 1.0
    assert pr.pick_r1(cs)["sql"] == "a"
    only_rep = [C("r", 1.0, cost=3.0, src="repair")]
    assert pr.gen_cost(only_rep[0], only_rep) == 0.0


def test_r1r2_combines_and_ties_keep_list_order():
    cs = [C("z", 9.0, res=ZERO), C("a", 2.0, cost=10.0), C("b", 1.5, cost=2.0), C("c", 1.0, cost=0.0)]
    assert pr.pick_r1r2(cs)["sql"] == "b"  # z dropped (R2); a 0.0, b 1.1, c 1.0 (R1)
    assert pr.pick_r1(cs)["sql"] == "z"  # R1 alone keeps the degenerate candidate
    tie = [C("a", 1.0), C("b", 1.0)]
    assert pr.pick_r1r2(tie)["sql"] == "a"


def test_pick_by_name():
    row = {"id": "x", "cands": [C("z", 3.0, res=ZERO), C("b", 1.0)]}
    assert pr.pick(row, "max")["sql"] == "z"
    assert pr.pick(row, "r2")["sql"] == "b"
    assert pr.pick({"id": "y"}, "r1r2") is None
