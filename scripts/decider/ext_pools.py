"""External candidate SQL for BIRD-train questions -> candidate pools in the box-W list format (re-labelled here).

  python -I scripts/decider/ext_pools.py   # -> data/decider/ext_pools.jsonl + ext_stats.json

Sources (downloaded under data/raw/ext/, treated as untrusted data: parsed, never executed as code):
  sense   SENSE (ACL 2024) DPO pairs on BIRD-train (data/raw/ext/sense/dpo/deepseek-1.3b.bird.grouped-any-limit1.jsonl):
          chosen/rejected SQL from a 1.3B model; question = the "Question:" line of the instruction.
  merge   SynSQL-Merge-Think-310k BIRD split (data/raw/ext/synsql_merge/train_merge_bird.json, cc-by-nc-4.0): two
          draft SQL per prompt from Qwen2.5-Coder-7B samples; question = the line before the drafts.
Their labels are NOT used: every candidate is executed on our BIRD-train SQLite copy and marked ok when its result
set equals the gold's (BIRD-Verified corrected gold where one exists, else the original). Questions are matched to
BIRD-train by (db_id, question text); held-out DBs, contaminated DBs and gold-noise ids (without a correction) are
dropped. Output rows: {id, db_id, question, context, gold_sql, source, cands: [{sql, runs, nonnull, ok, sig, res}]}.
"""
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.teacher.verify import db_paths, execute  # noqa: E402

EXT = ROOT / "data/raw/ext"
OUT = ROOT / "data/decider"


def norm_q(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def sense_rows():
    for ln in open(EXT / "sense/dpo/deepseek-1.3b.bird.grouped-any-limit1.jsonl"):
        r = json.loads(ln)
        m = re.search(r"Question:\s*(.+?)\s*$", r["instruction"], re.S)
        if m:
            yield r["db_id"], m.group(1), [r["chosen"], r["rejected"]]


def merge_rows():
    for r in json.load(open(EXT / "synsql_merge/train_merge_bird.json")):
        s = r["input_seq"]
        q = re.search(r"Question:\n(.*?)\n\nHere are some corresponding draft SQL", s, re.S)
        if not q:
            continue
        lines = q.group(1).strip().split("\n")
        drafts = re.findall(r"^\d+\. (SELECT.*?)\n【Execution result】", s, re.S | re.M)
        yield r["db_id"], lines[-1], drafts


def main():
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    held = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
    cont = set(json.load(open(ROOT / "data/sys1/contaminated_dbs.json")))
    noise = set(json.load(open(ROOT / "data/sys1/gold_noise_drop.json"))["ids"])
    fixed = {}
    for f in ("bird-verified-train.json", "bird-verified-val.json"):
        for r in json.load(open(ROOT / "data/raw/bird_verified" / f)):
            fixed[int(r["question_id"])] = r
    idx = {(t["db_id"], norm_q(t["question"])): n for n, t in enumerate(tr)}
    pools, src, st = collections.defaultdict(set), collections.defaultdict(set), collections.Counter()
    for name, gen in (("sense", sense_rows()), ("merge", merge_rows())):
        for db, q, sqls in gen:
            st[f"{name}_rows"] += 1
            n = idx.get((db, norm_q(q)))
            if n is None:
                st[f"{name}_unmatched"] += 1
                continue
            qid = f"bird_train_train_{n}"
            if db in held or db in cont:
                st[f"{name}_dropped_heldout_or_contaminated"] += 1
                continue
            if qid in noise and n not in fixed:
                st[f"{name}_dropped_gold_noise"] += 1
                continue
            pools[n].update(" ".join(s.split()) for s in sqls if s and s.strip())
            src[n].add(name)
    # box W's own candidates for every question with a BIRD-Verified correction: relabelled here against the corrected
    # gold, so a merged pool never mixes labels from two different golds (dataset.py then uses this pool alone)
    for role in ("r1", "r2"):
        for ln in open(ROOT / "runs/phase1/p1w/p1w" / role / "cands_train.jsonl"):
            r = json.loads(ln)
            n = int(r["id"].rsplit("_", 1)[1])
            if n in fixed and r["db_id"] not in held and r["db_id"] not in cont:
                pools[n].update(" ".join(c["sql"].split()) for c in r["cands"] if c.get("sql"))
                src[n].add("w_relabelled")
                st["w_relabelled_questions"] += 0
    st["w_relabelled_questions"] = sum("w_relabelled" in v for v in src.values())
    print(dict(st), flush=True)
    paths = db_paths([ROOT / "data/raw/bird_train/train_databases"])
    items, gold = [], {}
    for n, sqls in pools.items():
        g = fixed[n]["SQL"] if n in fixed else tr[n]["SQL"]
        gold[n] = g
        items.append((f"{n}|gold", tr[n]["db_id"], g))
        items += [(f"{n}|{k}", tr[n]["db_id"], s) for k, s in enumerate(sorted(sqls))]
    print(f"executing {len(items)} SQL on {len({tr[n]['db_id'] for n in pools})} DBs", flush=True)
    res = execute(items, paths, seconds=10.0, workers=8)
    OUT.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(OUT / "ext_pools.jsonl", "w") as f:
        for n, sqls in pools.items():
            gs = res.get(f"{n}|gold", ("error", "", 0))
            if gs[0] != "ok":
                st["gold_not_ok"] += 1
                continue
            cands = []
            for k, s in enumerate(sorted(sqls)):
                status, sig, nrows = res.get(f"{n}|{k}", ("error", "", 0))
                ok = status == "ok" and sig == gs[1]
                cands.append({"rank": k, "cost": 0.0, "sql": s, "runs": status in ("ok", "empty"),
                              "nonnull": status == "ok", "ok": ok, "sig": sig or None, "res": f"{nrows} rows"})
            if any(c["ok"] for c in cands) and any(not c["ok"] and c["nonnull"] for c in cands):
                kept += 1
            t = tr[n]
            ev = fixed[n]["evidence"] if n in fixed else t.get("evidence", "")
            q = fixed[n]["question"] if n in fixed else t["question"]
            f.write(json.dumps({"id": f"bird_train_train_{n}", "db_id": t["db_id"], "question": q, "context": ev,
                                "gold_sql": gold[n], "gold_corrected": n in fixed, "source": sorted(src[n]),
                                "cands": cands}) + "\n")
    st["questions"] = len(pools)
    st["questions_with_right_and_wrong"] = kept
    st["corrected_gold_used"] = sum(n in fixed for n in pools)
    (OUT / "ext_stats.json").write_text(json.dumps(dict(st), indent=1))
    print(json.dumps(dict(st), indent=1))


if __name__ == "__main__":
    main()
