"""Memorisation probes for a candidates file (CPU; no model). Plan: docs/paper_notes/research/freeform_plan_2026-10-10.md.

  python scripts/freeform/probe.py bug  --cands POOL.jsonl --ids data/sys1/heldout_v2/decision.json --pick rr
  python scripts/freeform/probe.py text --gen runs/ff/omni7b/decision_ff.jsonl --ids data/sys1/heldout_v2/decision.json

bug (wrong-gold reproduction): on held-out questions whose ORIGINAL BIRD-train gold is a known bug (BIRD-Verified
changed the SQL, kept the question and evidence, grades by set, and the two golds return different row sets), how
often does the picked candidate return the ORIGINAL (buggy) result vs the VERIFIED result? A model that memorised
BIRD-train reproduces the bug; one that generalises matches the fix at least as often as the bug. BIRD annotation
conventions learned from OTHER databases also pull toward the bug, so the number is only read RELATIVE to a reference
arm (the grammar pool, whose LoRA never saw these databases; or a non-OmniSQL generator).
Reported: n bug questions, pick == orig, pick == verified, pool has orig, pool has verified, and the reproduction ratio
orig / (orig + verified).

text (gold-text recall): share of generated SQL whose canonical text (sqlglot sqlite print, identifiers unquoted,
lower case, single spaces) equals the canonical ORIGINAL gold text; overall and among samples whose result is right
(needs merge.py --pool-less output to know which are right: --labelled). Grammar SQL is compiled, so its rate is ~0
by construction; compare ff generators with each other (OmniSQL vs a model that never trained on BIRD-train).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/val2"))
import execq  # noqa: E402

TRAIN = ROOT / "data/raw/bird_train/train.json"
DBROOT = ROOT / "data/raw/bird_train/train_databases"
RESERVED = ROOT / "data/sys1/heldout_v2/reserved.json"


def ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().rstrip(";")).strip().lower()


def canon(sql: str | None) -> str:
    if not sql:
        return ""
    try:
        import sqlglot
        out = sqlglot.transpile(sql, read="sqlite", write="sqlite", identify=False)[0]
    except Exception:
        out = sql
    return ws(out.replace('"', "").replace("`", ""))


def load_ids(p: Path | None, confirm: bool) -> list[str] | None:
    if not p:
        return None
    m = json.loads(p.read_text())
    ids = list(m["ids"] if isinstance(m, dict) else m)
    if RESERVED.exists() and set(ids) & set(json.loads(RESERVED.read_text())["ids"]) and not confirm:
        sys.exit("RESERVED ids in --ids: refused without --confirm")
    return ids


def bug_questions(ids: set[str], seconds: float = 30.0) -> dict[str, dict]:
    """id -> {orig_sig, ver_sig} for questions whose ORIGINAL gold is a verified bug (see module doc)."""
    tr = json.loads(TRAIN.read_text())
    out = {}
    for f in ("bird-verified-train.json", "bird-verified-val.json"):
        for v in json.load(open(ROOT / "data/raw/bird_verified" / f)):
            qid = f"bird_train_train_{int(v['question_id'])}"
            if qid not in ids or v.get("grading_method", "set") != "set":
                continue
            o = tr[int(v["question_id"])]
            if ws(v.get("question")) != ws(o["question"]) or ws(v.get("evidence")) != ws(o.get("evidence")):
                continue
            if ws(v["SQL"]) == ws(o["SQL"]):
                continue
            db = DBROOT / o["db_id"] / f"{o['db_id']}.sqlite"
            try:
                rows = execq.connect(str(db), seconds)
                a, b = rows(o["SQL"]), rows(v["SQL"])
            except Exception:
                continue
            if not a or not b or set(map(tuple, a)) == set(map(tuple, b)):
                continue
            out[qid] = {"orig_sig": execq.sig_of(a), "ver_sig": execq.sig_of(b)}
    return out


def pick(r: dict, rule: str, src: str | None) -> dict | None:
    cs = [c for c in r["cands"] if c.get("sql") and (src is None or c.get("src") == src)]
    lv = [c for c in cs if c.get("runs") and c.get("nonnull")]
    if rule == "rr":
        lv = [c for c in lv if "rr" in c]
        return max(lv, key=lambda c: c["rr"]) if lv else None
    if rule == "first":
        return cs[0] if cs else None
    if rule == "ff_vote":
        s = (r.get("ff") or {}).get("vote_sig")
        return {"sig": s} if s else None
    if rule == "ff_greedy":
        s = (r.get("ff") or {}).get("greedy_sig")
        return {"sig": s} if s else None
    raise ValueError(rule)


def run_bug(a):
    ids = load_ids(a.ids, a.confirm)
    recs = [json.loads(l) for l in open(a.cands)]
    recs = [r for r in recs if ids is None or r["id"] in set(ids)]
    bugs = bug_questions({r["id"] for r in recs})
    n = len(bugs)
    po = pv = ao = av = 0
    for r in recs:
        if r["id"] not in bugs:
            continue
        b = bugs[r["id"]]
        p = pick(r, a.pick, a.src)
        po += bool(p and p["sig"] == b["orig_sig"])
        pv += bool(p and p["sig"] == b["ver_sig"])
        sigs = {c.get("sig") for c in r["cands"] if c.get("runs") and c.get("nonnull")
                and (a.src is None or c.get("src") == a.src)}
        ao += b["orig_sig"] in sigs
        av += b["ver_sig"] in sigs
    res = {"file": str(a.cands), "pick": a.pick, "src": a.src, "n_records": len(recs), "n_bug": n,
           "pick_orig": po, "pick_verified": pv, "pool_has_orig": ao, "pool_has_verified": av,
           "repro_ratio_pick": round(po / (po + pv), 3) if po + pv else None,
           "repro_ratio_pool": round(ao / (ao + av), 3) if ao + av else None}
    print("BUG " + json.dumps(res), flush=True)


def run_text(a):
    ids = load_ids(a.ids, a.confirm)
    keep = None if ids is None else set(ids)
    tot = em = ok_n = ok_em = 0
    for r in map(json.loads, open(a.gen)):
        if keep is not None and r["id"] not in keep:
            continue
        g = canon(r.get("gold_sql"))
        for c in r["cands"]:
            if a.src and c.get("src") != a.src:
                continue
            hit = bool(g) and canon(c.get("sql_raw") or c.get("sql")) == g
            tot += 1
            em += hit
            if c.get("ok"):
                ok_n += 1
                ok_em += hit
    print("TEXT " + json.dumps({"file": str(a.gen), "samples": tot, "gold_text_em": round(em / tot, 4) if tot else None,
                                "right_samples": ok_n,
                                "gold_text_em_among_right": round(ok_em / ok_n, 4) if ok_n else None}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["bug", "text"])
    ap.add_argument("--cands", type=Path, help="bug: pool / merged / ff-only labelled jsonl")
    ap.add_argument("--gen", type=Path, help="text: gen.py output or a labelled file")
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--pick", choices=["rr", "first", "ff_vote", "ff_greedy"], default="rr")
    ap.add_argument("--src", default=None, help="only candidates with this src (e.g. ff)")
    a = ap.parse_args()
    run_bug(a) if a.probe == "bug" else run_text(a)


if __name__ == "__main__":
    main()
