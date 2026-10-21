"""System 1 + System 2 (LLM) hybrids on DECISION: accuracy vs cost per question.

System 1 = frozen R1 generator k=32 + VALHARD validator (59.05). System 2 = DeepSeek V4.1 Flash (thinking off, one-shot,
teacher bake-off run data/teacher/bakeoff/deepseek_t0) or Opus 5.5 (agent with read-only SQL, opus_bird_2026-10-10.md).
Strict EX vs the original BIRD-train gold (result sets executed here for the LLM answers and the agreement checks).

Policies
- route x%: System 1 answers; the x% least confident (no live candidate first, then smallest VALHARD top-2 class
  margin) go to the LLM.
- agree: System 1 and DeepSeek both answer; equal result sets are accepted; on disagreement DeepSeek's answer (or Opus's
  for the 3-tier policy) is used.
Costs per question are inputs (--s1 measured on GPU; LLM prices from the providers' published list prices).

  .venv/bin/python scripts/analysis/hybrid_router.py --opus runs/analysis/hybrid/opus_dec_answers --json runs/analysis/hybrid/router.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data/raw/bird_train/train_databases"
S1_FILE = ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl"   # VALHARD on R1 k=32 (labelled, DECISION + RESERVED)
DS_FILES = [ROOT / "data/teacher/bakeoff/deepseek_t0/verified.jsonl",            # 670 slot-covered DECISION q (bake-off)
            ROOT / "data/teacher/bakeoff/deepseek_t0_uncovered/verified.jsonl"]  # the other 126, same settings
FORBIDDEN = ("bird_dev", "gate.json", "sealed", "reserved")


def run(db, sql, t=30):
    con = sqlite3.connect(f"file:{DB}/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    tm = threading.Timer(t, con.interrupt)
    tm.start()
    try:
        return frozenset(con.execute(sql).fetchall())
    except Exception:
        return None
    finally:
        tm.cancel()
        con.close()


def job(a):
    qid, db, sqls = a
    return qid, [run(db, s) if s else None for s in sqls]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--opus", type=Path, help="dir of Opus agent answers (*.jsonl: question_id, sql)")
    ap.add_argument("--s1", type=float, default=0.0006, help="System 1 $/question (GPU)")
    ap.add_argument("--ds-in", type=float, default=0.15, help="DeepSeek $/M input (off-peak list; peak 0.30)")
    ap.add_argument("--ds-out", type=float, default=0.60, help="DeepSeek $/M output (off-peak list; peak 1.20)")
    ap.add_argument("--opus-q", type=float, default=0.075, help="Opus agent $/question (estimate, $0.05-0.10)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    dec = json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))
    ids = list(dec["ids"])
    n = len(ids)
    idset = set(ids)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    row = {i: tr[int(i.rsplit("_", 1)[1])] for i in ids}

    s1_ok, s1_sql, margin = {}, {}, {}
    for ln in open(S1_FILE):
        r = json.loads(ln)
        if r["id"] not in idset:
            continue  # RESERVED rows never read past the id
        c = [x for x in r["cands"] if x.get("sql") and x.get("src") in ("r1", "both", None)]
        lv = [x for x in c if x.get("runs") and x.get("nonnull")] or c
        if not lv:
            continue
        best = {}
        for x in lv:
            s = x.get("sig") or x["sql"]
            if s not in best or x["rr"] > best[s]["rr"]:
                best[s] = x
        top = sorted(best.values(), key=lambda x: -x["rr"])
        pk = max(lv, key=lambda x: x.get("rr", -1e9))  # val3_eval.pick: first max-rr live candidate (official tie-break)
        s1_ok[r["id"]] = bool(pk.get("ok"))
        s1_sql[r["id"]] = pk["sql"]
        margin[r["id"]] = top[0]["rr"] - top[1]["rr"] if len(top) > 1 else 1e6
    ds_ok, ds_sql, ds_tok = {}, {}, {}
    for f in DS_FILES:
        for ln in open(f):
            r = json.loads(ln)
            if r["id"] in idset and r.get("samples"):
                s = r["samples"][0]
                ds_ok[r["id"]] = bool(s.get("match"))
                ds_sql[r["id"]] = s.get("sql")
                ds_tok[r["id"]] = (s.get("n_in", 0), s.get("n_out", 0))
    op_sql = {}
    if a.opus:
        for f in glob.glob(str(a.opus / "*.jsonl")):
            for ln in open(f):
                try:
                    r = json.loads(ln)
                    op_sql[str(r["question_id"])] = r["sql"]
                except Exception:
                    pass
    jobs = [(i, row[i]["db_id"], [row[i]["SQL"], s1_sql.get(i), ds_sql.get(i), op_sql.get(i)]) for i in ids]
    with ProcessPoolExecutor(6) as ex:
        res = dict(ex.map(job, jobs, chunksize=4))
    op_ok = {i: (op_sql.get(i) is not None and res[i][3] is not None and res[i][0] is not None and res[i][3] == res[i][0])
             for i in ids}
    agree = {i: (i in s1_sql and res[i][1] is not None and res[i][1] == res[i][2]) for i in ids}
    ds_cost = {i: (t[0] * a.ds_in + t[1] * a.ds_out) / 1e6 for i, t in ds_tok.items()}
    ds_mean = sum(ds_cost.values()) / max(len(ds_cost), 1)
    print(f"DECISION n={n}: covered by System 1 {len(s1_ok)}, DeepSeek answers {len(ds_ok)}, Opus answers {len(op_sql)}")
    print(f"DeepSeek tokens/question in {sum(t[0] for t in ds_tok.values()) / len(ds_tok):.0f} out "
          f"{sum(t[1] for t in ds_tok.values()) / len(ds_tok):.0f} -> ${ds_mean:.5f}/q at ${a.ds_in}/${a.ds_out} per M")
    acc = lambda h: round(100 * sum(h[i] for i in ids) / n, 2)  # noqa: E731
    S1 = {i: s1_ok.get(i, False) for i in ids}
    DS = {i: ds_ok.get(i, False) for i in ids}
    OP = op_ok
    order = sorted(ids, key=lambda i: margin.get(i, -1e9))   # least confident first; uncovered (-1e9) first
    out = {"n": n, "prices": vars(a) | {"opus": str(a.opus), "json": str(a.json)}, "ds_cost_q": ds_mean, "rows": []}

    def add(name, h, cost):
        out["rows"].append({"policy": name, "ex": acc(h), "usd_per_1k": round(1000 * cost, 3)})
        print(f"  {name:44s} EX {acc(h):6.2f}   ${1000 * cost:8.3f} per 1,000 q")

    print("\nPolicy (DECISION)                              EX      cost")
    add("System 1 alone", S1, a.s1)
    add("DeepSeek V4.1 Flash alone", DS, ds_mean)
    if op_sql:
        add("Opus 5.5 agent alone", OP, a.opus_q)
    for frac in (0.16, 0.25, 0.5, 0.75):
        k = round(frac * n)
        esc = set(order[:k])
        h = {i: (DS[i] if i in esc else S1[i]) for i in ids}
        add(f"S1, least-confident {int(frac * 100)}% -> DeepSeek", h, a.s1 + frac * ds_mean)
    if op_sql:
        for frac in (0.16, 0.25, 0.5, 0.75):
            k = round(frac * n)
            esc = set(order[:k])
            h = {i: (OP[i] if i in esc else S1[i]) for i in ids}
            add(f"S1, least-confident {int(frac * 100)}% -> Opus", h, a.s1 + frac * a.opus_q)
    dis = [i for i in ids if not agree[i]]
    h = {i: (S1[i] if agree[i] else DS[i]) for i in ids}
    add(f"S1 + DeepSeek, disagree -> DeepSeek ({len(dis)} q)", h, a.s1 + ds_mean)
    if op_sql:
        h = {i: (S1[i] if agree[i] else OP[i]) for i in ids}
        add(f"S1 + DeepSeek, disagree -> Opus ({100 * len(dis) / n:.0f}%)", h, a.s1 + ds_mean + len(dis) / n * a.opus_q)
    agr = [i for i in ids if agree[i]]
    print(f"\nagreement: {len(agr)} questions ({100 * len(agr) / n:.1f}%); System 1 right on {100 * sum(S1[i] for i in agr) / max(len(agr), 1):.1f}% "
          f"of them; on the {len(dis)} disagreements System 1 {100 * sum(S1[i] for i in dis) / max(len(dis), 1):.1f}%, "
          f"DeepSeek {100 * sum(DS[i] for i in dis) / max(len(dis), 1):.1f}%" +
          (f", Opus {100 * sum(OP[i] for i in dis) / max(len(dis), 1):.1f}%" if op_sql else ""))
    out["agreement"] = {"n": len(agr), "s1_acc_on_agree": sum(S1[i] for i in agr) / max(len(agr), 1)}
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
