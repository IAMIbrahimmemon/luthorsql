"""scripts/analysis/gold_noise.py: drop categories from gold execution records, the BIRD-Verified join key, and the
written drop list (no held-out database, every category from the allowed set)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/analysis"))
spec = importlib.util.spec_from_file_location("gold_noise", REPO / "scripts/analysis/gold_noise.py")
gn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gn)


def test_category():
    assert gn.category({"status": "error"}) == "error"
    assert gn.category({"status": "timeout"}) == "timeout"
    assert gn.category({"status": "ok", "pathology": "empty"}) == "empty"
    assert gn.category({"status": "ok", "pathology": "single_zero"}) == "single_zero"
    assert gn.category({"status": "ok", "pathology": "single_null"}) == "single_null"
    assert gn.category({"status": "ok", "pathology": "all_null"}) is None  # reported, not dropped
    assert gn.category({"status": "ok", "pathology": None}) is None


def test_verified_key(tmp_path, monkeypatch):
    cmp = {"db|how many users are there": {"changed": True}, "db|other": {"changed": False}}
    p = tmp_path / "cmp.json"
    p.write_text(json.dumps(cmp))
    monkeypatch.setattr(gn, "CMP", p)
    T = [{"db_id": "db", "question": "How many users are there?"}, {"db_id": "db", "question": "Other."},
         {"db_id": "x", "question": "How many users are there?"}]
    assert gn.verified_changed(T) == {0: True, 1: False}


@pytest.mark.skipif(not gn.OUT.exists(), reason="drop list not built")
def test_drop_list_excludes_heldout():
    d = json.load(open(gn.OUT))
    T = json.load(open(gn.hv.TRAIN_JSON))
    held = set(json.load(open(gn.hv.HELDOUT_DBS)))
    assert d["n"] == len(d["ids"]) and d["ids_sha256"] == gn.hv.ids_sha256(d["ids"])
    assert set(d["ids"].values()) <= set(gn.DROP_CATS)
    assert not any(T[gn.hv._idnum(x)]["db_id"] in held for x in d["ids"])
