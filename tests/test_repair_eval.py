"""scripts/analysis/repair_eval.py: verdict bars and the pick rule (CPU, no data)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import repair_eval as RE  # noqa: E402


def test_verdict_bars():
    assert RE.verdict(1.5, 0.09) == "PASS"
    assert RE.verdict(1.5, 0.10) == "NO PASS"
    assert RE.verdict(3.0, 0.5) == "NO PASS"
    assert RE.verdict(0.5, 0.01) == "NO PASS"
    assert RE.verdict(0.49, 0.01) == "KILL"


def test_pick_prefers_live_and_max_rr():
    c = [{"sql": "a", "rr": 5.0, "runs": True, "nonnull": False},
         {"sql": "b", "rr": 1.0, "runs": True, "nonnull": True},
         {"sql": "c", "rr": 2.0, "runs": True, "nonnull": True, "src": "repair"},
         {"sql": None, "rr": 9.0}]
    assert RE.pick(c)["sql"] == "c"
    assert RE.pick([{"sql": "x", "rr": 0.0}, {"sql": "y", "rr": 1.0}])["sql"] == "y"  # none live: all
    assert RE.pick([]) is None


def test_guard_refuses_dev_and_gate():
    import pytest
    for p in ("runs/x/cands_bird_dev.jsonl", "runs/r1r2/r1/gate.json", "data/sys1/sealed/a.json"):
        with pytest.raises(SystemExit):
            RE.guard(Path(p))
