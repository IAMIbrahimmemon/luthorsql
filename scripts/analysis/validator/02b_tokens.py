"""Token cost of result-aware validator inputs (note 02). Mac only, DECISION only, no model run.

Re-executes every live R1 candidate of N DECISION questions, renders the candidate text in four formats and counts
OmniSQL-7B tokens:
  F0  current (question + context + SQL + 'Does this SQL answer ...?')
  F1  + thin preview (round 7, KILLed: shape, types, first row)
  F2  + rich result block (column names, types, NULL/duplicate counts, up to 3 rows, grain probe)
  F2s F2 with 2 rows and 16-char values (budget variant)
Out: runs/analysis/validator/02b_tokens.out (stdout), runs/analysis/validator/02b_examples.txt
"""
from __future__ import annotations

import importlib.util
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from luthor.sq2.rerank import candidate_text, result_preview  # noqa: E402

spec = importlib.util.spec_from_file_location("rf", Path(__file__).with_name("02_result_features.py"))
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)


def short(v, w):
    if v is None:
        return "NULL"
    if isinstance(v, float):
        v = round(v, 4)
    s = str(v)
    return s if len(s) <= w else s[: w - 1] + "…"


def colname(n, w=24):
    n = n.replace('"', "")
    return n if len(n) <= w else n[: w - 1] + "…"


def rich(names, rows, probe=None, k_rows=3, w=24):
    """Rich result block. Example:
    Result: 1 row x 1 col
    Columns: COUNT(T1.playerID) int
    Rows: (2945)
    Counted: 2945 rows, 36 distinct T1.playerID
    """
    f = rf.features(names, rows)
    if f["n"] == 0:
        return "0 rows\nColumns: " + ", ".join(colname(c) for c in names)
    lines = [f"{f['n']} row{'s' * (f['n'] != 1)} x {f['m']} col{'s' * (f['m'] != 1)}"]
    lines.append("Columns: " + ", ".join(f"{colname(c['name'])} {c['type']}" for c in f["cols"]))
    notes = []
    nul = [f"{colname(c['name'], 16)} {c['nulls']}" for c in f["cols"] if c["nulls"]]
    if nul:
        notes.append("NULLs: " + ", ".join(nul))
    if f["dup"]:
        notes.append(f"duplicate rows: {f['dup']}")
    if notes:
        lines.append("; ".join(notes))
    shown = rows[:k_rows]
    lines.append("Rows: " + " | ".join("(" + ", ".join(short(v, w) for v in r) + ")" for r in shown)
                 + (f" | … {f['n'] - k_rows} more" if f["n"] > k_rows else ""))
    if probe is not None and f["scalar"] is not None and probe != f["scalar"]:
        lines.append(f"Counted: {f['scalar']} rows, {probe} distinct")
    return "\n".join(lines)


def main(n_q=160, seed=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("seeklhy/OmniSQL-7B")
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    ids = set(json.load(open(rf.DEC))["ids"])
    rows_ = [json.loads(l) for l in open(rf.POOL)]
    rows_ = [r for r in rows_ if r["id"] in ids]
    random.Random(seed).shuffle(rows_)
    runs, L = {}, {k: [] for k in ("F0", "F1", "F2", "F2s")}
    per_list = {k: [] for k in L}
    ex = []
    for r in rows_[:n_q]:
        run = runs.get(r["db_id"]) or runs.setdefault(r["db_id"], rf.connect(r["db_id"]))
        live = [c for c in r["cands"] if c["src"] in ("r1", "both") and c["runs"] and c["nonnull"]]
        tot = {k: 0 for k in L}
        for c in live:
            try:
                names, got = run(c["sql"])
            except Exception:
                continue
            probe = rf.grain_probe(run, c["sql"])
            texts = {"F0": candidate_text(r["question"], r["context"], c["sql"]),
                     "F1": candidate_text(r["question"], r["context"], c["sql"], result_preview(got)),
                     "F2": candidate_text(r["question"], r["context"], c["sql"], rich(names, got, probe)),
                     "F2s": candidate_text(r["question"], r["context"], c["sql"], rich(names, got, probe, 2, 16))}
            for k, t in texts.items():
                n = ntok(t)
                L[k].append(n)
                tot[k] += n
            if len(ex) < 6 and (probe is not None or len(names) > 1):
                ex.append(texts["F2"])
        for k in L:
            per_list[k].append(tot[k])
    print(f"{n_q} DECISION questions, {len(L['F0'])} live candidates")
    for k, v in L.items():
        v = sorted(v)
        q = lambda p: v[int(p * (len(v) - 1))]
        print(f"{k:4s} tokens/candidate mean {st.mean(v):6.1f}  p50 {q(.5):4d}  p90 {q(.9):4d}  p99 {q(.99):4d}  max {v[-1]:4d}"
              f"  >384 (left-truncated today) {sum(x > 384 for x in v) / len(v) * 100:5.1f}%"
              f"  list total mean {st.mean(per_list[k]):7.0f}")
    (rf.OUT / "02b_examples.txt").write_text("\n\n=====\n\n".join(ex))


if __name__ == "__main__":
    main()
