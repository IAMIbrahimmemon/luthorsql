"""v4guard chain + eval: decision thresholds and DRY runs of the box script (no GPU, no network, no data read)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))


def _eval():
    import importlib
    try:
        return importlib.import_module("v4guard_eval")
    except Exception as e:  # phase1_eval reads data files at import
        import pytest
        pytest.skip(f"v4guard_eval import needs local data: {e}")


def test_thresholds_in_hits():
    m = _eval()
    assert m.min_hits(53.5) == 426  # 425/796 = 53.39, 426/796 = 53.52
    assert m.min_hits(59.0) == 470  # 469/796 = 58.92, 470/796 = 59.05
    assert m.min_hits(52.51) == 418


def test_decide():
    m = _eval()
    assert m.decide(53.52, None).startswith("GO")
    assert m.decide(53.39, 58.92).startswith("NO-GO")
    assert m.decide(52.51, 59.05).startswith("GO")
    assert m.decide(52.51, None).startswith("NO-GO")


def test_box_script_syntax_and_dry(tmp_path):
    for s in ("v4guard_box.sh", "run_v4guard_box.sh"):
        assert subprocess.run(["bash", "-n", str(ROOT / "scripts/chains" / s)]).returncode == 0
    env = {**os.environ, "DRY": "1", "REMOTE_ROOT": str(tmp_path), "REMOTE_D": str(tmp_path / "d"), "REPO": str(ROOT),
           "PY": sys.executable}
    r = subprocess.run(["bash", str(ROOT / "scripts/chains/v4guard_box.sh")], env=env, capture_output=True, text=True)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    assert "V4GUARD_DONE" in out and "V4GUARD_FAIL" not in out
    lines = [l for l in out.splitlines() if "DRY:" in l]
    names = [l.rsplit("/", 1)[-1] for l in lines]
    assert names == ["k8_off_gen.log", "k8_guard.log", "score_k8_guard_own.log", "score_k8_off_own.log",
                     "k32_guard.log", "score_k32_guard_own.log", "score_k32_guard_valhard.log"], names
    gens = [l for l in lines if " gen " in l]
    assert all("--bird-desc" in l and "--split dev" in l and "plans_v4/bird_train" in l for l in gens)
    assert "LUTHOR_JOIN_GUARD=off" in gens[0] and all("LUTHOR_JOIN_GUARD=direct" in l for l in gens[1:])
    assert "--k 32 --distinct" in gens[2] and "--k 8" in gens[1]
    assert not any("bird_dev" in l or "dev_databases" in l or "sealed" in l for l in lines)
    scores = [l for l in lines if " score " in l]
    assert all("--lora-r 16" in l and "--bird-desc" in l for l in scores)


def test_preflight_before_setup_uses_system_python():
    """Everything before setup6.sh must not touch the venv ($V) -- it does not exist yet on a fresh box."""
    txt = (ROOT / "scripts/chains/v4guard_box.sh").read_text()
    pre = txt[txt.index("preflight BEFORE setup"):txt.index("scripts/vast/setup6.sh")]
    assert '"$V"' not in pre and "$V " not in pre and "venv6" not in pre
    assert "python3 -c" in pre
