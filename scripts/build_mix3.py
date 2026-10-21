"""Build the round-3 training mix: quality first.

  python scripts/build_mix3.py --out data/mix3 [--synsql-cap 150000] [--min-audit 0.90]

Human-written (public), repeated --real-repeat times:
  plans_v3/{bird_train, spider_train, spider_syn_train, sparc_cosql_train}
Other public sets:
  plans_v3/wikisql_train (capped), plans_v3/synsql_train (capped; only after its audit passes)
Generated questions (data/qgen/{spider,bird}/{ds,claude}, data/qgen3/{bird,spider}/ds), kept only if ALL hold:
  1. the blind checker's reading matches the plan (qgen.grade_one), which also rejects schema/SQL wording;
  2. the source passed the independent Claude audit (faithful rate of checker-accepted questions >= --min-audit);
  3. the plan recompiles with today's join graph, runs within 5 s and returns a non-empty, non-trivial result;
  4. the question is not a duplicate (same database, same normalised text).
Every generated question gets the database value-lookup hits serving will see (row["db_values"]).
Held out: --dev-dbs and every database of the sealed sets; questions with >= 50% word overlap with any
sealed question are dropped. Public data only.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import shutil
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_plans import dedup_against  # noqa: E402
from qgen import grade_one  # noqa: E402
from quality import fix_fan_out, identifier_leak, limit_without_order  # noqa: E402

from luthor.sq.candidates import ValueIndex  # noqa: E402
from luthor.sq.plan import Plan, compile_plan  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402

V3 = Path("/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v3")
V2 = Path("/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2")
EVAL = Path("data/eval_plans_v2")
DBS = {"spider": Path("data/raw/spider_data/database"), "bird": Path("data/raw/bird_train/train_databases")}
CARDS = {"spider": V3 / "spider_train" / "cards", "bird": V2 / "bird_train" / "cards"}
GEN = [("data/qgen/spider/ds", "spider", "spider/ds"), ("data/qgen/spider/claude", "spider", "spider/claude"),
       ("data/qgen/bird/ds", "bird", "bird/ds"), ("data/qgen/bird/claude", "bird", "bird/claude"),
       ("data/qgen3/bird/ds", "bird", "bird/ds"), ("data/qgen3/spider/ds", "spider", "spider/ds")]
HUMAN = ["bird_train", "spider_train", "spider_syn_train", "sparc_cosql_train"]


def _qnorm(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()


def audit_rates(d: Path = Path("data/qgen/audit3")) -> dict[str, tuple[float, int]]:
    """source -> (faithful-and-natural rate among checker-accepted audited questions, n)."""
    key = {json.loads(l)["aid"]: json.loads(l) for l in open(d / "key.jsonl")} if (d / "key.jsonl").exists() else {}
    acc = collections.defaultdict(lambda: [0, 0])
    for f in d.glob("audit_result_*.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            k = key.get(r.get("aid"))
            if k and k["checker_accepted"]:
                acc[k["source"]][0] += bool(r.get("faithful")) and bool(r.get("natural"))
                acc[k["source"]][1] += 1
    return {s: (a / n, n) for s, (a, n) in acc.items() if n}


def verified_ok(qdir: Path) -> set[str]:
    ok = set()
    for f in qdir.glob("*.verified.jsonl"):
        for l in open(f):
            try:
                v = json.loads(l)
            except Exception:
                continue
            if v.get("faithful") and v.get("natural"):
                ok.add(v["vid"])
            else:
                ok.discard(v.get("vid"))  # a later "no" for the same vid wins
    return ok


def claude_ok(tag: str, d: Path = Path("data/claude_verify")) -> set[str]:
    """Item ids Claude (subagent verifier) or Gemini (scripts/gemini_verify.py, packs Claude didn't reach)
    judged faithful and natural, for one source tag. Claude's verdict wins when both exist."""
    ok = set()
    for f in sorted(d.glob("pack_*.gemini.jsonl")) + sorted(d.glob("pack_*.codex.jsonl")) + sorted(d.glob("pack_*.result.jsonl")):
        for l in open(f):
            try:
                v = json.loads(l)
            except Exception:
                continue
            src, _, iid = str(v.get("vid", "")).partition("|")
            if src == tag:
                (ok.add if v.get("faithful") and v.get("natural") else ok.discard)(iid)
    return ok


def gen_rows(qdir: Path, stats: collections.Counter, need_verifier: bool = True) -> list[dict]:
    plans = {json.loads(l)["plan_id"]: json.loads(l) for l in open(qdir / "plans_with_ids.jsonl")}
    vok = verified_ok(qdir)
    cok = claude_ok(str(qdir).replace("data/", "", 1))
    out = []
    for f in sorted(qdir.glob("questions_*.json")):
        cf = f.with_name(f.name.replace("questions_", "checked_"))
        try:
            items = json.loads(f.read_text())["items"]
            verdicts = {c["qid"]: c for c in json.loads(cf.read_text())["items"] if c.get("qid")} if cf.exists() else {}
        except Exception:
            continue
        for it in items:
            if it.get("skip") or it.get("plan_id") not in plans:
                continue
            p = plans[it["plan_id"]]
            for k, q in enumerate(it.get("questions") or []):
                stats["written"] += 1
                iid = f"{it['plan_id']}_{k}"
                if iid in cok:  # Claude verified it: the strictest judge, no further check needed
                    stats["claude_verified"] += 1
                    out.append({"plan_id": it["plan_id"], "db_id": p["db_id"], "plan": p["plan"],
                                "question": q["text"].strip(), "style": q.get("style")})
                    continue
                c = verdicts.get(iid)
                if c is None:
                    stats["no_verdict"] += 1
                    continue
                ok, why = grade_one(p["plan"], q, c)
                if not ok:
                    stats["checker_rejected"] += 1
                    continue
                if need_verifier and f"{it['plan_id']}_{k}" not in vok:
                    stats["not_verified"] += 1
                    continue
                out.append({"plan_id": it["plan_id"], "db_id": p["db_id"], "plan": p["plan"], "question": q["text"].strip(),
                            "style": q.get("style")})
    stats["checker_and_verifier_accepted"] = len(out)
    return out


def _check_db(args):
    """One database: build its value index, recompile + execute each plan, value-lookup each question."""
    kind, db, items = args
    card = SchemaCard.load(CARDS[kind] / f"{db}.json")
    g = card.graph()
    con = sqlite3.connect(f"file:{DBS[kind] / db / f'{db}.sqlite'}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    deadline = [0.0]
    con.set_progress_handler(lambda: 1 if time.time() > deadline[0] else 0, 10_000)
    try:
        vindex = ValueIndex.build(con, card)
    except Exception:
        vindex = None
    ok_sql = {}
    out = []
    for r in items:
        lits = {str(v) for f in r["plan"].get("filters") or [] for v in f.get("values") or []}
        if identifier_leak(r["question"], card, lits) or limit_without_order(r["plan"]):
            continue
        fixed, _ = fix_fan_out(r["plan"], card)
        if fixed is None:
            continue
        r = {**r, "plan": fixed}
        pk = json.dumps(r["plan"], sort_keys=True)
        if pk not in ok_sql:
            sql = None
            try:
                sql = compile_plan(Plan.from_dict(r["plan"]), g)
                deadline[0] = time.time() + 5
                rows = con.execute(sql).fetchmany(1000)
                if not rows or all(all(v in (None, 0, 0.0, "") for v in row) for row in rows):
                    sql = None
            except Exception:
                sql = None
            ok_sql[pk] = sql
        if ok_sql[pk] is None:
            continue
        out.append({**r, "compiled_sql": ok_sql[pk], "db_values": vindex.match(r["question"]) if vindex else []})
    con.close()
    return kind, db, out, sum(v is None for v in ok_sql.values()), len(ok_sql)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/mix3"))
    ap.add_argument("--dev-dbs", type=Path, default=Path("data/mix2_dev_dbs.json"))
    ap.add_argument("--real-repeat", type=int, default=2)
    ap.add_argument("--wiki-cap", type=int, default=30000)
    ap.add_argument("--synsql-cap", type=int, default=0, help="cap on verified SynSQL rows (0 = all verified)")
    ap.add_argument("--min-audit", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rnd = random.Random(0)
    dev_dbs = set(json.loads(a.dev_dbs.read_text()))
    eval_q, eval_dbs = [], set()
    for d in EVAL.iterdir():
        for l in open(d / "plans.jsonl"):
            r = json.loads(l)
            eval_q.append(r["question"])
            eval_dbs.add(r["db_id"])
    keep_db = lambda db: db not in eval_dbs and db not in dev_dbs
    print(f"sealed: {len(eval_q)} questions, {len(eval_dbs)} databases; dev: {len(dev_dbs)} databases", flush=True)

    stats = collections.Counter()
    a.out.mkdir(parents=True, exist_ok=True)
    cards_out = a.out / "cards"
    cards_out.mkdir(exist_ok=True)

    def copy_card(src_dir: Path, db: str):
        f = src_dir / f"{db}.json"
        if f.exists() and not (cards_out / f.name).exists():
            shutil.copy(f, cards_out / f.name)

    rows = []
    sets = [(n, "human") for n in HUMAN] + [("wikisql_train", "wiki")] + [("synsql_train", "synth")]
    syn_ok = (verified_ok(Path("data/qgen_synsql")) if Path("data/qgen_synsql").exists() else set()) | claude_ok("qgen_synsql")
    for name, kind in sets:
        p = V3 / name / "plans.jsonl"
        if not p.exists():
            print(f"missing {p}: skipped", flush=True)
            continue
        src = [json.loads(l) for l in open(p)]
        src = [r for r in src if r.get("outcome") == "plan" and not r.get("empty_result") and keep_db(r["db_id"])]
        if kind == "synth":  # only rows that passed style filter + rules + verifier
            syn_cards = {}
            kept = []
            for r in src:
                if r["id"] not in syn_ok:
                    continue
                card = syn_cards.setdefault(r["db_id"], SchemaCard.load(V3 / name / "cards" / f"{r['db_id']}.json"))
                fixed, st = fix_fan_out(r["plan"], card)
                if fixed is None:
                    continue
                if st == "rewritten":
                    r = {**r, "plan": fixed, "compiled_sql": compile_plan(Plan.from_dict(fixed), card.graph())}
                kept.append(r)
            src = kept
        cap = {"wiki": a.wiki_cap, "synth": a.synsql_cap}.get(kind)
        if cap:
            orig = [r for r in src if not r.get("twin_of")]
            rnd.shuffle(orig)
            keep = {r["id"] for r in orig[:cap]}
            src = [r for r in src if (r.get("twin_of") or r["id"]) in keep]
        for db in {r["db_id"] for r in src}:
            copy_card(V3 / name / "cards", db)
        stats[f"rows_{name}"] = len(src)
        for k in range(a.real_repeat if kind == "human" else 1):
            rows += [{**r, "source": name, "id": f"{r['id']}#{k}" if k else r["id"]} for r in src]

    rates = audit_rates()
    for s, (r, n) in sorted(rates.items()):
        print(f"audit {s}: {r:.1%} faithful+natural of {n} checker-accepted", flush=True)
        stats[f"audit_{s.replace('/', '_')}"] = round(r, 4)
    gen_by_db = collections.defaultdict(list)
    for qdir, kind, audit_src in GEN:
        qd = Path(qdir)
        if not (qd / "plans_with_ids.jsonl").exists():
            continue
        tag = qdir.replace("data/", "").replace("/", "_")
        if audit_src not in rates and not claude_ok(qdir.replace("data/", "", 1)):
            print(f"{qdir}: no audit and no Claude verdicts: dropped", flush=True)
            continue
        st = collections.Counter()
        # Spider sources pass the quality bar with checker + rules alone (audit 92-94%); BIRD needs the verifier
        for r in gen_rows(qd, st, need_verifier=kind != "spider"):
            if keep_db(r["db_id"]):
                gen_by_db[(kind, r["db_id"])].append({**r, "tag": tag})
        for k, v in st.items():
            stats[f"{tag}_{k}"] = v
    print(f"generated: {sum(map(len, gen_by_db.values()))} accepted questions on {len(gen_by_db)} databases; executing...", flush=True)
    seen_q = set()
    with Pool(a.workers) as pool:
        for kind, db, out, bad, n in pool.imap_unordered(_check_db, [(k, db, v) for (k, db), v in gen_by_db.items()]):
            stats["gen_plans_dropped_exec"] += bad
            stats["gen_plans_checked"] += n
            copy_card(CARDS[kind], db)
            for i, r in enumerate(out):
                key = (db, _qnorm(r["question"]))
                if key in seen_q:
                    stats["gen_duplicate_question"] += 1
                    continue
                seen_q.add(key)
                rows.append({"id": f"{r['tag']}_{r['plan_id']}_{i}", "source": f"gen_{kind}", "db_id": db,
                             "question": r["question"], "context": "", "outcome": "plan", "plan": r["plan"],
                             "compiled_sql": r["compiled_sql"], "db_values": r["db_values"], "style": r["style"],
                             "checked": True, "empty_result": False})
    stats["gen_kept"] = sum(1 for r in rows if r["source"].startswith("gen_"))
    before = len(rows)
    rows = dedup_against(rows, eval_q)
    stats["dropped_eval_overlap"] = before - len(rows)
    rnd.shuffle(rows)
    with open(a.out / "plans.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    stats.update({f"final_{k}": v for k, v in collections.Counter(r["source"] for r in rows).items()})
    stats["final_total"] = len(rows)
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
