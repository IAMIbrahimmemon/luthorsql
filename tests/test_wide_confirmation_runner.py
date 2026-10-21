"""Pipeline failures must not silently start analysis on incomplete scores."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/analysis/run_wide_confirmation.py"
    spec = importlib.util.spec_from_file_location("confirmation_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_inputs(path):
    for name in ("confirmation_manifest.json", "frozen_retention_model.json"):
        (path / name).write_text("{}")


def test_generation_failure_stops_pipeline_and_records_failure(tmp_path, monkeypatch):
    m = load_runner()
    setup_inputs(tmp_path)
    monkeypatch.setattr(m.sys, "argv", ["runner", "--run-dir", str(tmp_path)])
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(m.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        m.main()
    status = json.loads((tmp_path / "confirmation_status.json").read_text())
    assert status["state"] == "failed" and status["stage"] == "generate"
    assert len(calls) == 1
    assert not (tmp_path / "confirmation_results.json").exists()


def test_success_runs_stages_in_order_and_persists_results(tmp_path, monkeypatch):
    m = load_runner()
    setup_inputs(tmp_path)
    monkeypatch.setattr(m.sys, "argv", ["runner", "--run-dir", str(tmp_path)])
    calls = []

    def run(command, **kwargs):
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        calls.append(Path(command[2]).name)

    monkeypatch.setattr(m.subprocess, "run", run)
    monkeypatch.setattr(m, "analyze", lambda base: {"verified": True})
    m.main()
    assert calls == ["confirm_wide_pool.py", "score_wide_pool_mps.py"]
    assert json.loads((tmp_path / "confirmation_status.json").read_text())["state"] == "complete"
    assert json.loads((tmp_path / "confirmation_results.json").read_text()) == {"verified": True}
