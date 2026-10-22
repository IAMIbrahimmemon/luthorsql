"""Score the memorisation probe (experiments/memorisation_probe/PREREG.md).

P1: schema-free SQL, EX against gold (read-only SQLite, 30 s, set of rows), per set.
P2: guided vs general completion of the question's second half: word-level ROUGE-L F1 and normalised exact match,
guided - general per item, paired sign-flip (10,000 flips, seed 20261021) per set.

  .venv/bin/python experiments/memorisation_probe/score_probe.py --box <scratchpad>/memprobe --tasks <transcripts> --out experiments/memorisation_probe/result.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DBROOT = {"kaggledbqa_test": ROOT / "data/sys1/sealed/dbs", "spider_realistic": ROOT / "data/sys1/sealed/dbs",
          "reserved": ROOT / "data/raw/bird_train/train_databases"}


def execute(root, db, sql):
    con = sqlite3.connect(f"file:{root}/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    t = threading.Timer(30, con.interrupt)
    t.start()
    try:
        return frozenset(con.execute(sql).fetchall())
    except Exception:
        return None
    finally:
        t.cancel()
        con.close()


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def rouge_l(a, b):
    a, b = words(a), words(b)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def signflip(d, n=10000, seed=20261021):
    rng = random.Random(seed)
    obs = sum(d) / len(d)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d) / len(d)) >= abs(obs) - 1e-12 for _ in range(n))
    return {"mean": round(obs, 4), "p_two_sided": round((hits + 1) / (n + 1), 5)}


def load(box, cond, mismatches):
    """Answers aligned to the chunk by line position (agents copy item_id by hand; one copied it wrong)."""
    out = {}
    for ch in sorted((box / "chunks" / cond).glob("*.jsonl")):
        items = [json.loads(l)["item_id"] for l in open(ch)]
        f = box / "answers" / cond / ch.name
        rows = []
        if f.exists():
            for ln in open(f):
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        if len(rows) != len(items):
            mismatches.append(f"{cond}/{ch.name}: {len(rows)} lines for {len(items)} items, matched by id")
            by = {r.get("item_id"): r for r in rows}
            out.update({i: by[i] for i in items if i in by})
            continue
        for i, r in zip(items, rows):
            if r.get("item_id") != i:
                mismatches.append(f"{cond}/{ch.name}: line for {i} labelled {r.get('item_id')}")
            out[i] = r
    return out


def compliance(tasks, box):
    """Per probe transcript: tool calls other than one Read of its own chunk and one Write of its own answer file."""
    pat = re.compile(re.escape(str(box)) + r"/chunks/(P1|P2g|P2n)/((?:p|md)[12][gn]?_\d+)\.jsonl")
    bad, n = [], 0
    for f in sorted(tasks.glob("*.output")):
        m = pat.search(open(f).read(20000))
        if not m:
            continue
        n += 1
        cond, name = m.groups()
        for ln in open(f):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            msg = r.get("message") or {}
            if r.get("type") != "assistant" or not isinstance(msg, dict):
                continue
            for b in msg.get("content") or []:
                if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                    continue
                path = str((b.get("input") or {}).get("file_path", ""))
                ok = (b["name"] == "Read" and path.endswith(f"chunks/{cond}/{name}.jsonl")) or \
                     (b["name"] == "Write" and path.endswith(f"answers/{cond}/{name}.jsonl")) or b["name"] == "SubagentHandback"
                if not ok:
                    bad.append(f"{cond}/{name}: {b['name']} {json.dumps(b.get('input'))[:120]}")
    return {"transcripts": n, "violations": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True, help="subagent transcript dir")
    a = ap.parse_args()
    gold, setof = {}, {}
    for s in ("kaggledbqa_test", "spider_realistic"):
        for r in map(json.loads, open(ROOT / f"data/eval/sealed_{s}.jsonl")):
            gold[r["id"]], setof[r["id"]] = (r["db_id"], r["gold_sql"]), s
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    for i in json.load(open(ROOT / "data/sys1/heldout_v2/reserved.json"))["ids"]:
        q = tr[int(i.rsplit("_", 1)[1])]
        gold[i], setof[i] = (q["db_id"], q["SQL"]), "reserved"
    out = {"P1": {}, "P2": {}, "id_mismatches": []}
    out["compliance"] = compliance(a.tasks, a.box)
    p1 = load(a.box, "P1", out["id_mismatches"])
    per = defaultdict(lambda: [0, 0, 0])  # correct, answered, total
    for i, (db, g) in gold.items():
        s = setof[i]
        per[s][2] += 1
        sql = (p1.get(i) or {}).get("sql")
        if not sql:
            continue
        per[s][1] += 1
        gr, pr = execute(DBROOT[s], db, g), execute(DBROOT[s], db, sql)
        per[s][0] += gr is not None and pr is not None and gr == pr
    for s, (c, n_ans, n) in per.items():
        out["P1"][s] = {"n": n, "answered": n_ans, "ex": round(100 * c / n, 2)}
    pg, pn = load(a.box, "P2g", out["id_mismatches"]), load(a.box, "P2n", out["id_mismatches"])
    for s in ("kaggledbqa_test", "spider_realistic", "reserved"):
        truth = json.load(open(a.box / f"truth_{s}.json"))
        rows = []
        for i, q in truth.items():
            w = q.split()
            ref = " ".join(w[len(w) // 2:])
            g, n = (pg.get(i) or {}).get("completion", ""), (pn.get(i) or {}).get("completion", "")
            rows.append((rouge_l(g, ref), rouge_l(n, ref), words(g) == words(ref), words(n) == words(ref)))
        k = len(rows)
        out["P2"][s] = {"n": k, "rougeL_guided": round(sum(r[0] for r in rows) / k, 4),
                        "rougeL_general": round(sum(r[1] for r in rows) / k, 4),
                        "exact_guided": sum(r[2] for r in rows), "exact_general": sum(r[3] for r in rows),
                        "guided_minus_general": signflip([r[0] - r[1] for r in rows])}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
