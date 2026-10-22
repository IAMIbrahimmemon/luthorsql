"""Score the Mini-Dev memorisation probe (experiments/memorisation_probe/minidev/PREREG.md).

P1 (schema-free SQL) and P2 (guided vs general completion) as in ../score_probe.py, plus gold-string reproduction,
execution errors and, descriptively, a per-question cross-tab against third-party prediction files
(schemagate, bird-bench.github.io #251), re-scored here with the same executor.

  .venv/bin/python experiments/memorisation_probe/minidev/score_md.py --box <memprobe_md> --tasks <dir> \
      --data data/raw/ext/minidev/mini_dev_sqlite.json --ext <dir with schemagate *_plain_sqlite*.json> --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/memorisation_probe"))
import score_probe as sp  # noqa: E402

DBS = ROOT / "data/bench/dev_20240627/dev_databases"


def norm(x):
    x = x.lower().replace('"', "'").replace("`", "")
    x = re.sub(r"\s*([(),=<>])\s*", r"\1", x)
    return re.sub(r"\s+", " ", x).strip().rstrip(";")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ext", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = json.load(open(a.data))
    items = [(f"md_{q['question_id']}", q["db_id"], q["SQL"], q["difficulty"]) for q in d]
    out = {"id_mismatches": []}
    out["compliance"] = sp.compliance(a.tasks, a.box)
    p1 = sp.load(a.box, "P1", out["id_mismatches"])
    gres = {i: sp.execute(DBS, db, g) for i, db, g, _ in items}
    blind = {}
    errs = strs = 0
    for i, db, g, _ in items:
        sql = (p1.get(i) or {}).get("sql") or ""
        r = sp.execute(DBS, db, sql) if sql else None
        errs += r is None
        strs += bool(sql) and norm(sql) == norm(g)
        blind[i] = gres[i] is not None and r == gres[i]
    n = len(items)
    out["gold_failed_to_execute"] = sum(v is None for v in gres.values())
    out["P1"] = {"n": n, "ex": round(100 * sum(blind.values()) / n, 2), "exec_errors": errs, "gold_string": strs,
                 "by_difficulty": {}}
    for lvl in ("simple", "moderate", "challenging"):
        sel = [i for i, _, _, dl in items if dl == lvl]
        out["P1"]["by_difficulty"][lvl] = {"n": len(sel), "ex": round(100 * sum(blind[i] for i in sel) / len(sel), 2)}
    pg, pn = sp.load(a.box, "P2g", out["id_mismatches"]), sp.load(a.box, "P2n", out["id_mismatches"])
    truth = json.load(open(a.box / "truth_minidev.json"))
    rows = []
    for i, q in truth.items():
        w = q.split()
        ref = " ".join(w[len(w) // 2:])
        g, nn = (pg.get(i) or {}).get("completion", ""), (pn.get(i) or {}).get("completion", "")
        rows.append((sp.rouge_l(g, ref), sp.rouge_l(nn, ref), sp.words(g) == sp.words(ref), sp.words(nn) == sp.words(ref)))
    k = len(rows)
    out["P2"] = {"n": k, "rougeL_guided": round(sum(r[0] for r in rows) / k, 4), "rougeL_general": round(sum(r[1] for r in rows) / k, 4),
                 "exact_guided": sum(r[2] for r in rows), "exact_general": sum(r[3] for r in rows),
                 "guided_minus_general": sp.signflip([r[0] - r[1] for r in rows])}
    if a.ext:
        out["crosstab"] = {}
        for f in sorted(a.ext.glob("*_plain_sqlite*.json")):
            preds = json.load(open(f))
            ok = {}
            for k_, (i, db, g, _) in enumerate(items):
                sql = str(preds.get(str(k_), "")).split("\t----- bird -----\t")[0]
                r = sp.execute(DBS, db, sql) if sql.strip() else None
                ok[i] = gres[i] is not None and r == gres[i]
            both = sum(ok[i] and blind[i] for i in ok)
            out["crosstab"][f.stem] = {"ex_rescored": round(100 * sum(ok.values()) / n, 2),
                                       "their_correct": sum(ok.values()), "also_correct_blind": both,
                                       "blind_correct_they_miss": sum(blind[i] and not ok[i] for i in ok),
                                       "share_of_their_correct_reachable_blind": round(both / max(1, sum(ok.values())), 4)}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
