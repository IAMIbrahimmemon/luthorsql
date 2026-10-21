"""RL draws scored by the decisions decode actually made (rl_core.sample_draws / trace_logprob), on a tiny random
Qwen2 slot model: every draw is kept (the old plan_targets path dropped ~25% of sampled plans), sampled
trajectory frequencies match exp(log-prob), and the log-prob is differentiable. CPU, under a minute."""
import collections
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rl"))
import rl_core as rc  # noqa: E402
import luthor.sq2.train as tr  # noqa: E402
from luthor.sq2.model import TorchScores  # noqa: E402
from test_sq2_r1_levers import _card, _rows, _tiny  # noqa: E402
import pytest  # noqa: E402
from _global_state import isolated  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate():
    """Order independence: an earlier test (e.g. test_rl_pilot's end-to-end rl_train.main) sets R2's template
    vocabulary process-wide, which changes the tiny model's head sizes and so its draws. Each test here runs with the
    pristine vocabulary / candidate order and a fixed seed."""
    with isolated(seed=0):
        yield


def _setup():
    card = _card()
    exs = [tr.make_example(r, card, random.Random(0), order="v2") for r in _rows()]
    m = _tiny(value_side=True)
    m.eval()
    st, oix, enc = tr.encode_db(m, card, exs)
    return m, st, exs, enc


def test_every_draw_is_kept_and_scored():
    m, st, exs, enc = _setup()
    with torch.no_grad():
        ds = rc.sample_draws(TorchScores(m, enc[0], st.option_ids, exs[0].cands), 32, 1.0, seed=0)
    assert len(ds) == 32 and all(d.trace for d in ds)
    assert all(rc.trace_conflicts(d.trace) == 0 for d in ds)
    lp = rc.trace_logprobs(m, enc[0], ds, 1.0)
    assert lp.shape == (32,) and torch.isfinite(lp).all() and (lp < 0).all()
    lp.sum().backward()
    assert m.W_q.weight.grad is not None and float(m.W_q.weight.grad.abs().sum()) > 0


def test_trajectory_frequencies_match_logprob():
    """At a low temperature the tiny model's draws concentrate on a few trajectories; their sampled frequencies
    must match exp(trace_logprob) (the whole point: the gradient is taken on the policy's real probabilities)."""
    m, st, exs, enc = _setup()
    N = 3000
    with torch.no_grad():
        for T in (0.002, 0.001, 0.0005):  # the first (temperature, question) whose draws concentrate
            for qi in range(len(exs)):
                ds = rc.sample_draws(TorchScores(m, enc[qi], st.option_ids, exs[qi].cands), N, T, seed=0)
                top = collections.Counter(tuple(d.trace) for d in ds).most_common(5)
                if top[0][1] / N > 0.05:
                    break
            else:
                continue
            break
        assert top[0][1] / N > 0.05, "draws did not concentrate: the check below would be meaningless"
        for tr_, c in top:
            assert not any(ap for *_, ap in tr_)
            p = math.exp(float(rc.trace_logprob(m, enc[qi], list(tr_), T)))
            se = math.sqrt(p * (1 - p) / N)
            assert abs(c / N - p) < 4 * se + 1e-3, (c / N, p)
