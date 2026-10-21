"""Run local 01-01 confirmation stages sequentially, with durable status and logs.

The cohort and prediction model must already be frozen. No wall-clock watcher
timeout: generation completes before scoring starts, then analysis runs directly.
Restarting resumes the existing per-question generation and scoring checkpoints.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).parent)]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def analyze(base):
    from luthor.retention import nested_curve, predict_retention
    from luthor.stats import cluster_signflip, question_signflip
    from wide_pool_retention import validate_scored

    model_path = base / "frozen_retention_model.json"
    scored_path = base / "confirmation_scored.jsonl"
    reference_path = base / "confirmation_candidates.jsonl"
    allowed = json.loads((ROOT / "data/sys1/dev_dbs_bird.json").read_text())
    records = read_rows(scored_path)
    validate_scored(records, read_rows(reference_path), set(allowed))
    model = json.loads(model_path.read_text())
    predictions = predict_retention(model, records, allowed_dbs=allowed, allow_same_dbs=True)
    observed = nested_curve(records, tuple(model["ks"]))
    outcomes = observed.pop("rows")
    for k, point in observed["curve"].items():
        point["predicted_accuracy"] = predictions["predicted_accuracy"][k]
        point["prediction_error"] = point["predicted_accuracy"] - point["accuracy"]
        point["within_one_point_observed"] = abs(point["prediction_error"]) <= 0.01
    a = {r["id"]: r["pick"] for r in outcomes if r["k"] == 8}
    b = {r["id"]: r["pick"] for r in outcomes if r["k"] == 32}
    db = {r["id"]: r["db_id"] for r in outcomes if r["k"] == 8}
    return {
        "scope": predictions["scope"],
        "primary": {"comparison": "fixed width32 vs width8", "cluster": cluster_signflip(a, b, db),
                    "question": question_signflip(a, b)},
        "observed": observed,
        "secondary_predicted_width": predictions["chosen_k"],
        "prediction_model_refitted": False,
        "note": "Fresh questions on the same databases; not independent new-database evidence.",
        "provenance": {p.name: digest(p) for p in (model_path, scored_path, reference_path,
                                                 base / "confirmation_manifest.json")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/research/01-01")
    args = parser.parse_args()
    base = args.run_dir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    lock = (base / "confirmation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = base / "confirmation_manifest.json"
    model = base / "frozen_retention_model.json"
    for required in (manifest, model):
        if not required.exists():
            raise FileNotFoundError(f"Freeze this input before launching: {required}")
    provenance = {p.name: digest(p) for p in (manifest, model)}

    def status(state, stage, **extra):
        atomic_json(base / "confirmation_status.json", {
            "state": state, "stage": stage, "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "frozen_inputs": provenance, **extra,
        })

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    python = str(ROOT / ".venv/bin/python")
    stages = [
        ("generate", [python, "-I", str(ROOT / "scripts/analysis/confirm_wide_pool.py"),
                      "--generate", "--manifest", str(manifest),
                      "--out", str(base / "confirmation_candidates.jsonl")]),
        ("score", [python, "-I", str(ROOT / "scripts/analysis/score_wide_pool_mps.py"),
                   "--input", str(base / "confirmation_candidates.jsonl"),
                   "--output", str(base / "confirmation_scored.jsonl")]),
    ]
    stage = "preflight"
    try:
        for stage, command in stages:
            if any(digest(base / name) != value for name, value in provenance.items()):
                raise ValueError("Frozen inputs changed while running")
            status("running", stage)
            with (base / f"confirmation_{stage}.log").open("a") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        stage = "analyze"
        status("running", stage)
        result = analyze(base)
        atomic_json(base / "confirmation_results.json", result)
        status("complete", stage, result="confirmation_results.json")
    except BaseException as exc:
        status("failed", stage, error=str(exc))
        raise


if __name__ == "__main__":
    main()
