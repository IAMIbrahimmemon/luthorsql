"""train_sq2.py's out-of-memory handling, with train_db replaced by a fake that runs out of memory above a
question count: forward OOMs are retried in halves with the right weights, questions that never fit are
counted, a backward OOM is raised, and the run stops once too many questions are lost. CPU, no model."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_sq2 as ts  # noqa: E402
from luthor.sq2.train import BackwardOOM  # noqa: E402


def _fake(fits: int, calls: list, backward_oom_db: str | None = None):
    def train_db(model, card, es, budget_chars, rnd, scale=1.0):
        calls.append((len(es), scale))
        if card == backward_oom_db:
            raise BackwardOOM(card)
        if len(es) > fits:
            raise torch.OutOfMemoryError("fake")
        return float(sum(es)) / len(es), {"ptr": 1.0}
    return train_db


def test_forward_oom_splits_and_weights(monkeypatch):
    calls = []
    monkeypatch.setattr(ts, "train_db", _fake(2, calls))
    led = ts.OOMLedger(1.0)
    loss, parts, n = ts.run_db(None, "db", "db", [1, 2, 3, 4, 5, 6, 7, 8], 9000, None, 0.25, led)
    assert n == 8 and led.skipped == 0 and led.retries == 3  # 8 -> 4+4 -> 2+2+2+2
    assert loss == pytest.approx(4.5) and parts == {"ptr": pytest.approx(1.0)}
    done = [(k, s) for k, s in calls if k <= 2]
    assert [k for k, _ in done] == [2, 2, 2, 2]
    # each part's mean loss is scaled by its share: sum(scale) = the database's scale, so the gradient equals the
    # unsplit mean over all 8 questions
    assert [s for _, s in done] == [pytest.approx(0.25 / 4)] * 4


def test_question_that_never_fits_is_counted(monkeypatch):
    monkeypatch.setattr(ts, "train_db", _fake(0, []))
    led = ts.OOMLedger(1.0)
    led.seen = 4
    assert ts.run_db(None, "big", "big", [1, 2, 3, 4], 9000, None, 1.0, led)[2] == 0
    assert led.skipped == 4 and led.by_db == {"big": 4}


def test_backward_oom_is_raised(monkeypatch):
    monkeypatch.setattr(ts, "train_db", _fake(8, [], backward_oom_db="db"))
    with pytest.raises(BackwardOOM):
        ts.run_db(None, "db", "db", [1, 2], 9000, None, 1.0, ts.OOMLedger(1.0))


def test_too_many_skips_stop_the_run():
    led = ts.OOMLedger(0.002, min_seen=2000)
    led.seen = 1000
    led.skip("works_cycles", 500)  # too early to judge
    led.seen = 300_000
    led.skip("works_cycles", 100)  # 600 / 300k = 0.2%: at the limit, still fine
    with pytest.raises(SystemExit, match="works_cycles"):
        led.skip("works_cycles", 8)
