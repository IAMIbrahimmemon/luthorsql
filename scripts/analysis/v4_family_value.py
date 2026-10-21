"""Which grammar v4 additions earn their keep? Mac CPU, no model.

Per v4 construct family:
  A  BIRD-train coverage (reports/grammar_gap_2026-10-10/v{3,4}_bird_train.jsonl, the repo's own coverage check):
     v4-covered golds that use the family, and how many of them v3 could not cover (unlocks). Held-out DBs
     (dev_dbs_bird.json) are split out: "train DBs" is the selection side, DECISION is reported only.
  B  training rows that exercise the family's slots/heads in the v4 training mix (data/plans_v4 bird + spider + synsql
     plans; detectable from the plan dict: offset, order2, colcmp ops, wide (proj>4 / group>2 / filters>4), cond2,
     join pairs). Converter-only families (flatten, cte, scalar, boolsum, dedup, now, wrap_count) add no slot.
  C  DECISION (reported, never used to choose): the newly covered questions per family and v4desc's right picks /
     oracle@8 on them; false-positive use of each slot family in v4desc's k=8 candidates (the candidate SQL uses the
     construct, the gold does not) and how often such a candidate is the greedy or the pick and wrong.

  uv run --offline python scripts/analysis/v4_family_value.py   # -> runs/analysis/grammar_v4_fix/v4_family_value.txt
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

os.environ["LUTHOR_GRAMMAR"] = "v4"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_eval import IDS  # noqa: E402  (DECISION ids)
from luthor.sq.plan import COL_OPS  # noqa: E402

OUT = ROOT / "runs/analysis/grammar_v4_fix"
GG = ROOT / "reports/grammar_gap_2026-10-10"
HELD = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
LOG: list[str] = []
SLOT_FAMILIES = ("offset", "order2", "colcmp", "proj5+", "group3+", "filt5+", "cond2", "joinpairs")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def plan_families(p: dict) -> set:
    f = set()
    if p.get("offset"):
        f.add("offset")
    if p.get("order2"):
        f.add("order2")
    flt = list(p.get("filters") or []) + [x.get(k) for x in p.get("projections") or [] for k in ("cond", "cond2") if x.get(k)]
    if any(x and x.get("op") in COL_OPS for x in flt):
        f.add("colcmp")
    if len(p.get("projections") or []) > 4:
        f.add("proj5+")
    if len(p.get("group_by") or []) > 2:
        f.add("group3+")
    if len(p.get("filters") or []) > 4:
        f.add("filt5+")
    if any(x.get("cond2") for x in p.get("projections") or []):
        f.add("cond2")
    if p.get("join_cols"):
        f.add("joinpairs")
    return f


SEL = re.compile(r"^SELECT (.*?) FROM ", re.S)


def sql_families(sql: str) -> set:
    """Slot families visible in compiled SQL (the compiler's own output format)."""
    f = set()
    if not sql:
        return f
    if " OFFSET " in sql:
        f.add("offset")
    m = re.search(r" ORDER BY (.*?)(?: LIMIT | OFFSET |$)", sql)
    if m and re.search(r"(?:ASC|DESC), ", m.group(1)):
        f.add("order2")
    w = re.search(r" WHERE (.*?)(?: GROUP BY | ORDER BY | LIMIT |$)", sql)
    if w and re.search(r'(?:\w+\.)?"[^"]+" (?:=|!=|>|>=|<|<=) (?:\w+\.)?"[^"]+"', w.group(1)):
        f.add("colcmp")
    s = SEL.match(sql)
    if s:
        depth, n = 0, 1
        for ch in s.group(1):
            depth += ch == "("
            depth -= ch == ")"
            n += ch == "," and depth == 0
        if n > 4:
            f.add("proj5+")
    if re.search(r"CASE WHEN [^)]*? AND [^)]*? THEN", sql):
        f.add("cond2")
    g = re.search(r" GROUP BY (.*?)(?: HAVING | ORDER BY | LIMIT |$)", sql)
    if g and g.group(1).count(",") >= 2:
        f.add("group3+")
    return f


def load_ids(p, ids):
    out = {}
    with open(p) as f:
        for line in f:
            m = re.match(r'\{"id": "([^"]+)"', line)
            if m and m.group(1) not in ids:
                continue
            r = json.loads(line)
            if r["id"] in ids:
                out[r["id"]] = r
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dec = set(IDS)
    v3 = {r["i"]: r for r in map(json.loads, open(GG / "v3_bird_train.jsonl"))}
    v4 = {r["i"]: r for r in map(json.loads, open(GG / "v4_bird_train.jsonl"))}

    say("## A. BIRD-train coverage per v4 family (tags of the repo coverage check; a question may use several)")
    cov = {"train": collections.Counter(), "train_unlock": collections.Counter(), "dec": collections.Counter(),
           "dec_unlock": collections.Counter()}
    dec_unlock_ids = collections.defaultdict(list)
    tot = collections.Counter()
    for i, r in v4.items():
        side = "dec" if f"bird_train_train_{i}" in dec else ("held" if r.get("db") in HELD else "train")
        c3 = v3.get(i, {}).get("o") == "plan"
        c4 = r.get("o") == "plan"
        tot[side + "_v3"] += c3
        tot[side + "_v4"] += c4
        tot[side + "_n"] += 1
        if not c4 or side == "held":
            continue
        for t in r.get("v4") or []:
            cov[side][t] += 1
            if not c3:
                cov[side + "_unlock"][t] += 1
                if side == "dec":
                    dec_unlock_ids[t].append(f"bird_train_train_{i}")
        if not c3 and not (r.get("v4") or []):
            cov[side + "_unlock"]["(no tag: vocab)"] += 1
            if side == "dec":
                dec_unlock_ids["(no tag: vocab)"].append(f"bird_train_train_{i}")
    say(f"train DBs: n {tot['train_n']}, v3 {tot['train_v3']}, v4 {tot['train_v4']} | DECISION: n {tot['dec_n']}, "
        f"v3 {tot['dec_v3']}, v4 {tot['dec_v4']}")
    fams = sorted(set(cov["train"]) | set(cov["train_unlock"]) | set(cov["dec"]), key=lambda t: -cov["train_unlock"][t])
    say("family             train-DB golds using it | of them v3-uncovered (unlocks) | DECISION using | DECISION unlocks")
    for t in fams:
        say(f"{t:18s} {cov['train'][t]:8d} | {cov['train_unlock'][t]:8d} | {cov['dec'][t]:6d} | {cov['dec_unlock'][t]:6d}")

    say("\n## B. training rows per slot family in the v4 mix (plans with outcome=plan; twins included)")
    srcs = {"bird": "data/plans_v4/train/bird_train", "spider": "data/plans_v4/train/spider_train",
            "synsql": "data/plans_v4/train_big/synsql_train"}
    rows = {k: collections.Counter() for k in srcs}
    n = collections.Counter()
    for k, d in srcs.items():
        with open(ROOT / d / "plans.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r.get("outcome") != "plan" or r.get("db_id") in HELD:
                    continue
                n[k] += 1
                for x in plan_families(r["plan"]):
                    rows[k][x] += 1
    say("family        " + "".join(f"{k:>10s}" for k in srcs) + "     total   (share of all rows)")
    allrows = sum(n.values())
    for x in SLOT_FAMILIES:
        t = sum(rows[k][x] for k in srcs)
        say(f"{x:12s}  " + "".join(f"{rows[k][x]:10d}" for k in srcs) + f"  {t:8d}   {100 * t / allrows:5.2f}%")
    say(f"rows          " + "".join(f"{n[k]:10d}" for k in srcs) + f"  {allrows:8d}")

    say("\n## C1. DECISION questions newly covered by v4 (reported only): v4desc right picks / oracle@8 per family")
    N = load_ids(ROOT / "runs/phase1/v4desc/r1/rr/dev_scored.jsonl", dec)

    def hit(i):
        r = N.get(i)
        if not r:
            return False, False
        cs = r["cands"]
        live = [j for j, c in enumerate(cs) if c.get("runs") and c.get("nonnull")] or list(range(len(cs)))
        return bool(cs[max(live, key=lambda j: cs[j].get("rr", -1e9))].get("ok")), any(c.get("ok") for c in cs)
    allu = sorted({i for v in dec_unlock_ids.values() for i in v})
    say(f"newly covered DECISION questions: {len(allu)}; v4desc right picks {sum(hit(i)[0] for i in allu)}, "
        f"oracle@8 {sum(hit(i)[1] for i in allu)}")
    for t, ids in sorted(dec_unlock_ids.items(), key=lambda x: -len(x[1])):
        say(f"  {t:18s} {len(ids):3d} questions: right {sum(hit(i)[0] for i in ids)}, oracle@8 {sum(hit(i)[1] for i in ids)}")

    say("\n## C2. false-positive slot use on DECISION (candidate SQL uses the family, gold does not), v4desc vs R3a")
    C = load_ids(ROOT / "runs/phase1/seed/r1/rr/dev_scored.jsonl", dec)
    gold = {}  # DECISION gold families from the gold's v4 plan (questions v4 covers; others are skipped here)
    with open(ROOT / "data/plans_v4/train/bird_train/plans.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("id") in dec and r.get("outcome") == "plan":
                gold[r["id"]] = plan_families(r["plan"])
    say(f"(questions with a v4 gold plan: {len(gold)})")
    say("family       arm     cands_fp  greedy_fp(wrong)  pick_fp(wrong) | gold uses (questions)")
    for x in SLOT_FAMILIES[:-1]:
        for nm, S in (("v4desc", N), ("R3a", C)):
            a = collections.Counter()
            for i, r in S.items():
                if i not in gold:
                    continue
                g = x in gold[i]
                a["gold"] += g
                if g:
                    continue
                cs = r["cands"]
                live = [j for j, c in enumerate(cs) if c.get("runs") and c.get("nonnull")] or list(range(len(cs)))
                p = max(live, key=lambda j: cs[j].get("rr", -1e9))
                a["c"] += sum(x in sql_families(c.get("sql")) for c in cs)
                if x in sql_families(cs[0].get("sql")):
                    a["g"] += 1
                    a["gw"] += not cs[0].get("ok")
                if x in sql_families(cs[p].get("sql")):
                    a["p"] += 1
                    a["pw"] += not cs[p].get("ok")
            say(f"{x:12s} {nm:7s} {a['c']:8d}  {a['g']:6d} ({a['gw']:3d})       {a['p']:5d} ({a['pw']:3d})   | {a['gold']}")
    (OUT / "v4_family_value.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
