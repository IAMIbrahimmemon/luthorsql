"""teacher.py label --s1-cands (System 1 candidate block, opt-in) and the deepseek_s1_report math.
   pytest tests/test_deepseek_s1.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/teacher"))
sys.path.insert(0, str(REPO / "scripts/analysis/hybrid"))

import deepseek_s1_report as R  # noqa: E402
import teacher as T  # noqa: E402

CARDS = REPO / "data/sys1/bird_train/cards"
needs_card = pytest.mark.skipif(not (CARDS / "trains.json").exists(), reason="needs the BIRD-train trains card")
ROW = {"id": "q1", "db_id": "trains", "question": "How many trains run east?", "context": "east refers to direction = 'east'"}
CANDS = [{"label": "A", "sql": "SELECT COUNT(*) FROM trains WHERE direction = 'east'", "result": "1 row x 1 col (int); first: (10)",
          "n_of_32": 20},
         {"label": "B", "sql": "SELECT COUNT(id) FROM trains", "result": "1 row x 1 col (int); first: (20)", "n_of_32": 3}]


@needs_card
def test_flag_off_prompt_byte_identical():
    from luthor.sqlw.prompt6 import think_messages
    cards = T.Cards([CARDS])
    before = think_messages(cards.get("trains"), ROW["question"], ROW["context"])
    assert T.label_builder(cards)(ROW) == before
    assert T.label_builder(cards, None)(ROW) == before


def test_cli_default_off_and_prompt_version(tmp_path):
    import argparse
    ns = argparse.Namespace(model="m", temp=0.0, n=1, think=False, input="x", max_tokens=1, eval_only=True, s1_cands=None)
    assert T.write_run_json(tmp_path / "r", ns, "label", 1)["prompt_version"] == T.LABEL_PROMPT
    ns.s1_cands = "c.jsonl"
    m = T.write_run_json(tmp_path / "r2", ns, "label", 1)
    assert m["prompt_version"] == T.LABEL_PROMPT + T.S1_PROMPT_SUFFIX and m["s1_cands"] == "c.jsonl"


@needs_card
def test_s1_block_inserted_before_final_line():
    cards = T.Cards([CARDS])
    base = T.label_builder(cards)(ROW)[0]["content"]
    got = T.label_builder(cards, {"q1": CANDS})(ROW)[0]["content"]
    assert got.endswith(T.S1_FINAL)
    assert got.startswith(base[:-len(T.S1_FINAL)])  # everything before the final line is unchanged
    blk = got[len(base) - len(T.S1_FINAL):-len(T.S1_FINAL)]
    assert "[A] " + CANDS[0]["sql"] in blk and "(n_of_32=20)" in blk and "[B] " in blk
    assert "output that SQL exactly" in blk
    none = T.label_builder(cards, {"q1": []})(ROW)[0]["content"]
    assert "found no candidate queries" in none and "[A]" not in none
    with pytest.raises(KeyError):
        T.label_builder(cards, {})(ROW)


def test_report_math():
    assert R.norm_sql(" SELECT 1 ;\n") == "SELECT 1"
    assert R.dollars(1_000_000, 1_000_000) == pytest.approx(0.75)
    assert R.dollars(1_000_000, 0, peak=True) == pytest.approx(0.30)
    m = R.mcnemar_exact([True, True, False, False, True], [True, False, True, True, True])
    assert (m["a_only"], m["b_only"]) == (1, 2) and m["p"] == pytest.approx(1.0)
    m = R.mcnemar_exact([False] * 10, [True] * 10)  # 10 vs 0 discordant: p = 2 / 2**10
    assert m["p"] == pytest.approx(2 / 1024)
    assert R.mcnemar_exact([True], [True])["p"] == 1.0


def test_pick_margin_and_skip_curve():
    c = [{"sql": "a", "sig": "x", "rr": 5.0, "runs": True, "nonnull": True, "ok": True, "src": "r1"},
         {"sql": "b", "sig": "y", "rr": 2.0, "runs": True, "nonnull": True, "ok": False, "src": "both"},
         {"sql": "c", "sig": "x", "rr": 9.0, "runs": False, "nonnull": True, "ok": False, "src": "r1"}]  # dead
    assert R.s1_pick_and_margin(c) == (True, 3.0)
    assert R.s1_pick_and_margin(c[:1])[1] == math.inf
    assert R.s1_pick_and_margin([]) is None
    ids = ["a", "b", "c", "d"]
    ok = {"a": False, "b": True, "c": True, "d": False}
    n_in = {i: 1000 for i in ids}
    n_out = {i: 100 for i in ids}
    s1 = {"a": (True, 9.0), "b": (False, 1.0), "c": (True, 5.0)}  # d has no candidates
    cur = R.skip_curve(ids, ok, n_in, n_out, s1, shares=(0.0, 0.5, 1.0))
    assert [x["skipped"] for x in cur] == [0, 2, 3]
    assert cur[0]["correct"] == 2 and cur[0]["tokens_in"] == 4000
    assert cur[1]["correct"] == 3 and cur[1]["tokens_in"] == 2000  # a, c skipped (S1 right on both); b, d by LLM
    assert cur[2]["correct"] == 2 and cur[2]["llm_calls"] == 1
    assert cur[1]["usd_with_s1"] == pytest.approx(R.dollars(2000, 200) + 4 * R.S1_COST_PER_Q, abs=1e-4)
