"""Training-question filters and the off-policy buffer for rl_train.py.

  load_drop_ids   --drop-ids PATH: question ids to leave out (e.g. data/sys1/gold_noise_drop.json from
                  scripts/analysis/gold_noise.py: ORIGINAL gold errors, returns nothing, or returns a single 0).
                  Accepts a JSON list, a JSON object ({"ids": [...]}, {"drop": [...]} or {id: reason}), or JSONL rows
                  with an "id" field. BIRD-Verified ids (bv_<split>_<n>) are mapped to the plan-set id.
  load_buffer     --buffer PATH...: off-policy SQL candidates for 04-01 (default files: OmniSQL-32B x 5 candidates,
                  data/eval/sel_cands/sel/{train,val}_cands.jsonl). Questions on held-out databases are removed and
                  the loader asserts none remain. Their `ok` flags were computed against BIRD-Verified (corrected)
                  gold, not the ORIGINAL gold the RL reward uses: every record says so (gold_source).

Sealed sets and score_* files are never opened here (load_buffer refuses them by name).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[2]
HELDOUT_DBS = R / "data/sys1/dev_dbs_bird.json"
BUFFER_FILES = [R / "data/eval/sel_cands/sel/train_cands.jsonl", R / "data/eval/sel_cands/sel/val_cands.jsonl"]
BUFFER_GOLD_SOURCE = "bird_verified"  # the buffer's ok flags / gold_sql: BIRD-Verified (expert-corrected) labels

_BV = re.compile(r"^bv_(?:train|val)_(\d+)$")


def plan_id(qid: str) -> str:
    """BIRD-Verified id -> the bird_train plan-set id (both keep BIRD train's question_id): bv_val_12 ->
    bird_train_train_12. Other ids are returned unchanged."""
    m = _BV.match(qid)
    return f"bird_train_train_{m.group(1)}" if m else qid


def heldout_dbs(path: Path | None = None) -> set[str]:
    return set(json.loads(Path(path or HELDOUT_DBS).read_text()))


def load_drop_ids(path: Path) -> set[str]:
    text = Path(path).read_text()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:  # JSONL
        obj = [json.loads(l) for l in text.splitlines() if l.strip()]
    if isinstance(obj, dict):
        for k in ("ids", "drop", "drop_ids"):
            if isinstance(obj.get(k), list):
                obj = obj[k]
                break
        else:
            obj = list(obj)
    ids = set()
    for x in obj:
        x = x.get("id") if isinstance(x, dict) else x
        if isinstance(x, str):
            ids.add(plan_id(x))
    return ids


def _refuse(p: Path) -> None:
    n = p.name.lower()
    if n.startswith("score_") or "sealed" in str(p).lower():
        raise ValueError(f"refusing to read {p}: scoring / sealed files are not training data")


def load_buffer(paths: list[Path], heldout: set[str]) -> tuple[dict[str, dict], dict]:
    """plan-set id -> {"bv_id", "db_id", "sqls": distinct candidate SQL, "ok_verified": per SQL (any candidate
    with that SQL right against BIRD-Verified gold), "gold_sql_verified", "gold_source": "bird_verified"}.
    Held-out databases are dropped, then asserted absent. Returns (buffer, stats)."""
    out, st = {}, {"rows": 0, "dropped_heldout": 0, "kept": 0, "with_right": 0, "gold_source": BUFFER_GOLD_SOURCE}
    for p in paths:
        p = Path(p)
        _refuse(p)
        for line in open(p):
            r = json.loads(line)
            st["rows"] += 1
            if r["db_id"] in heldout:
                st["dropped_heldout"] += 1
                continue
            sqls, ok = [], {}
            for c in r.get("cands") or []:
                s = (c.get("sql") or "").strip()
                if not s:
                    continue
                if s not in ok:
                    sqls.append(s)
                    ok[s] = False
                ok[s] = ok[s] or bool(c.get("ok"))
            out[plan_id(r["id"])] = {"bv_id": r["id"], "db_id": r["db_id"], "sqls": sqls,
                                     "ok_verified": [ok[s] for s in sqls], "gold_sql_verified": r.get("gold_sql"),
                                     "gold_source": BUFFER_GOLD_SOURCE}
    assert not any(v["db_id"] in heldout for v in out.values()), "held-out database left in the off-policy buffer"
    st["kept"] = len(out)
    st["with_right"] = sum(any(v["ok_verified"]) for v in out.values())
    return out, st
