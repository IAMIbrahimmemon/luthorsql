"""Process-wide state the System-1 code keeps in module globals, so tests can isolate themselves from each other.

luthor.sq2.targets.set_vocab (called by train_sq2._init_worker, e.g. from rl_train.main / eval) rewrites the head
vocabularies (VOCAB, PAIRS, MAIN_CHOICES) in place, and luthor.sq2.train / train_sq2 keep the candidate order and
augmentation knobs as globals. A test that builds a tiny model after one of those ran gets different head sizes
(and so a different random model). `snapshot()` at import (collection, before any test runs) records the pristine
values; `restore(s)` puts them back; `isolated()` starts from the pristine vocabulary (restore_vocab) with a fixed Python /
torch random state and afterwards restores everything it found."""
from __future__ import annotations

import contextlib
import random
import sys

import torch

import luthor.sq2.targets as T
import luthor.sq2.train as tr

_DICTS = ("VOCAB", "PAIRS", "MAIN_CHOICES", "CHOICES")
_SCALARS_TR = ("CAND_ORDER", "AUG_P", "DESC_DROP")


def snapshot() -> dict:
    s = {"T": {k: dict(getattr(T, k)) for k in _DICTS if hasattr(T, k)},
         "tr": {k: getattr(tr, k) for k in _SCALARS_TR if hasattr(tr, k)}}
    ts = sys.modules.get("train_sq2")
    if ts is not None:
        s["train_sq2"] = {k: getattr(ts, k) for k in ("VOCAB_ARGS", "CAND_ORDER") if hasattr(ts, k)}
    return s


def restore(s: dict) -> None:
    for k, v in s["T"].items():
        d = getattr(T, k)
        d.clear()
        d.update(v)
    for k, v in s["tr"].items():
        setattr(tr, k, v)
    ts = sys.modules.get("train_sq2")
    for k, v in s.get("train_sq2", {}).items():
        if ts is not None:
            setattr(ts, k, v)


# what targets.set_vocab writes (and nothing else: other tests switch grammar knobs at run time together with
# serialize.SLOTS, and a partial reset of those would make the two disagree)
_VOCAB_KEYS = {"VOCAB": ("templates", "compose", "nums"), "PAIRS": ("fn", "n0", "n1", "n2"),
               "MAIN_CHOICES": ("cn0", "cn1", "cn2", "compose", "compose_pos", "wrap", "wrap_idx")}


def restore_vocab(s: dict) -> None:
    """Put back only the set_vocab entries (and the candidate order) of snapshot s."""
    for name, keys in _VOCAB_KEYS.items():
        d = getattr(T, name)
        for k in keys:
            if k in s["T"][name]:
                d[k] = s["T"][name][k]
            else:
                d.pop(k, None)
    for k, v in s["tr"].items():
        setattr(tr, k, v)


PRISTINE = snapshot()


@contextlib.contextmanager
def isolated(seed: int = 0, pristine: bool = True):
    """Run with the pristine module globals and a fixed random state; afterwards put back whatever was there."""
    before = snapshot()
    rs, ts = random.getstate(), torch.get_rng_state()
    if pristine:
        restore_vocab(PRISTINE)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        restore(before)
        random.setstate(rs)
        torch.set_rng_state(ts)
