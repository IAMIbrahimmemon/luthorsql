"""Execution-label free-form (src "ff") candidates exactly as at grammar gen time and merge them into a pool.
CPU only (sqlite3 workers); runs on the Mac or the box.

  python scripts/freeform/merge.py --ff runs/ff/omni7b/decision_ff.jsonl \
      --pool runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl --ids data/sys1/heldout_v2/decision.json \
      --db-roots data/raw/bird_train/train_databases --out runs/ff/omni7b/decision_union.jsonl

Labelling: scripts/val2/execq.py (the copy of rerank_sq2._exec_db): read-only SQLite, first 100,000 rows, a time limit
per query (--cand-seconds 3 as at gen time, --gold-seconds for the gold), ok = result SET equals the gold's,
sig = gen-time signature (md5 of the sorted reprs of the row set, 12 hex). NOT luthor.sqlw.execute.signature (different
hash: it would never match grammar sigs).

SQL surface (--normalize, default on): each ff SQL is re-printed by sqlglot (sqlite dialect, every identifier
double-quoted, keywords upper-case: the compiler's surface style) and the normalised text is used only when it runs
and has the same sig as the raw text on this database; the raw text is kept in "sql_raw".

Merge per question (pool order is kept, so cands[0] and every grammar field are unchanged):
  - ff samples whose SQL text equals a pool candidate's: that candidate gets ff_n += 1 (no new entry);
  - every pool candidate gets ff_sig_n = number of live ff samples with its sig (a vote feature, never shown to the
    validator);
  - --same-sig drop (default): an ff class whose sig is already live in the pool adds no entry; keep: add it;
  - each new live sig adds ONE ff representative (its most frequent text, ties -> lowest rank) with ff_n (samples with
    that text) and ff_sig_n, ordered by ff_sig_n, at most --max-ff of them. Dead ff samples add nothing;
  - record fields: "ff" = {n, no_sql, live, sigs, vote_sig, greedy_sig, gold_status}; ids absent from the pool (the
    grammar-uncovered questions) become new records with only ff candidates (in_slots / covered false).
ff representatives carry no "cost" (rerank_sq2.pick_rerank reads cost with a 0.0 default) and rank = 1000 + rank.

--no-gold (blind sets): only runs / nonnull / sig / res are written, ok = null; nothing is scored.
RESERVED ids are refused without --confirm.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/val2"))
import execq  # noqa: E402

RESERVED = ROOT / "data/sys1/heldout_v2/reserved.json"
ADDED_SRCS = ("ff", "mut")  # candidate sources added by this script (everything else is the grammar pool)


def live(c: dict) -> bool:
    return bool(c.get("runs") and c.get("nonnull"))


def normalize(sql: str) -> str | None:
    try:
        import sqlglot
        out = sqlglot.transpile(sql, read="sqlite", write="sqlite", identify=True, normalize_functions="upper")
        return out[0] if len(out) == 1 else None
    except Exception:
        return None


# ---- execution ----------------------------------------------------------------------------------------------------

def exec_blind(task):
    """No gold: (db path, [(key, None, [sql | None])], _, cand_seconds) -> [(key, 'blind', None, [(runs, nonnull,
    None, sig, res)])]."""
    path, qs, _, csec = task
    try:
        rows = execq.connect(path, csec)
    except Exception:
        return [(k, "error", None, []) for k, _, _ in qs]
    out = []
    for key, _, sqls in qs:
        res, seen = [], {}
        for s in sqls:
            if s is None:
                res.append((False, False, None, None, None))
            elif s in seen:
                res.append(seen[s])
            else:
                try:
                    got = rows(s)
                    r = (True, any(v is not None for x in got for v in x), None, execq.sig_of(got),
                         execq.result_preview(got))
                except Exception:
                    r = (False, False, None, None, None)
                seen[s] = r
                res.append(r)
        out.append((key, "blind", None, res))
    return out


def label(recs: list[dict], paths: dict, a) -> dict:
    """-> {id: (gold_status, [labelled ff cand dicts])}; ff cands get sql / sql_raw / runs / nonnull / ok / sig / res."""
    by_db, todo = collections.defaultdict(list), {}
    for r in recs:
        sqls = [c.get("sql") for c in r["cands"]]
        norm = [normalize(s) if (s and a.normalize) else None for s in sqls]
        todo[r["id"]] = (sqls, norm)
        flat = sqls + [n for n in norm if n is not None]
        by_db[r["db_id"]].append((r["id"], None if a.no_gold else r.get("gold_sql"), flat))
    if a.no_gold:
        tasks = []
        for db, lst in by_db.items():
            if db in paths:
                tasks += [(paths[db], lst[i:i + 20], 0, a.cand_seconds) for i in range(0, len(lst), 20)]
        res = {}
        for t in tasks:
            res.update({k: (st, sg, rr) for k, st, sg, rr in exec_blind(t)})
    else:
        res = execq.run_tasks(by_db, paths, a.gold_seconds, a.cand_seconds, workers=a.workers)
    out = {}
    for r in recs:
        st, _, rr = res.get(r["id"], ("no-db", None, []))
        sqls, norm = todo[r["id"]]
        if not rr:
            out[r["id"]] = (st, [])
            continue
        raw_res, norm_res = rr[:len(sqls)], iter(rr[len(sqls):])
        cl = []
        for c, s, n, x in zip(r["cands"], sqls, norm, raw_res):
            c = dict(c)
            runs, nonnull, ok, sig, prev = x
            if n is not None:
                y = next(norm_res)
                if n != s and y[0] and runs and y[3] == sig:
                    c["sql_raw"], c["sql"] = s, n
            c.update(runs=runs, nonnull=nonnull, ok=ok, sig=sig, res=prev)
            cl.append(c)
        out[r["id"]] = (st, cl)
    return out


# ---- merge ---------------------------------------------------------------------------------------------------------

def vote_sig(ff: list[dict]) -> str | None:
    """Most common sig among live ff samples; ties -> the class seen first (rank order)."""
    lv = [c for c in sorted(ff, key=lambda c: c["rank"]) if live(c)]
    if not lv:
        return None
    n = collections.Counter(c["sig"] for c in lv)
    best = max(n.values())
    return next(c["sig"] for c in lv if n[c["sig"]] == best)


def merge_record(pool_rec: dict | None, q: dict, ff: list[dict], gold_status: str, max_ff: int = 8,
                 same_sig: str = "drop") -> dict:
    base = dict(pool_rec) if pool_rec else {k: q.get(k) for k in ("id", "db_id", "question", "context", "gold_sql")}
    if not pool_rec:
        base.update(in_slots=False, covered=False)
    pool = [dict(c) for c in (pool_rec or {}).get("cands", [])]
    by_text = {c["sql"]: c for c in pool if c.get("sql")}
    sig_n = collections.Counter(c["sig"] for c in ff if live(c))
    for c in pool:
        c["ff_n"] = 0
        c["ff_sig_n"] = sig_n.get(c.get("sig"), 0) if live(c) else 0
    pool_live_sigs = {c["sig"] for c in pool if live(c)}
    classes = collections.defaultdict(list)
    for c in ff:
        if c.get("sql") and c["sql"] in by_text:
            by_text[c["sql"]]["ff_n"] += 1
        elif live(c) and (same_sig == "keep" or c["sig"] not in pool_live_sigs):
            classes[c["sig"]].append(c)
    new = []
    for sig, members in classes.items():
        texts = collections.Counter(m["sql"] for m in members)
        top = max(texts.values())
        rep = min((m for m in members if texts[m["sql"]] == top), key=lambda m: m["rank"])
        e = {k: v for k, v in rep.items() if k != "cost"}
        e.update(rank=1000 + rep["rank"], ff_n=texts[rep["sql"]], ff_sig_n=sig_n[sig])
        new.append(e)
    new.sort(key=lambda e: (-e["ff_sig_n"], e["rank"]))
    g = next((c for c in ff if c["rank"] == 0 and c.get("temp") == 0.0), None)
    base["cands"] = pool + new[:max_ff]
    vs = vote_sig(ff)
    ok_of = {c["sig"]: c.get("ok") for c in ff if live(c)}  # ok is a function of the sig (same gold, same rows)
    base["ff"] = {"n": len(ff), "no_sql": sum(not c.get("sql") for c in ff), "live": sum(live(c) for c in ff),
                  "sigs": len(sig_n), "vote_sig": vs, "greedy_sig": g["sig"] if g and live(g) else None,
                  "vote_n": sig_n.get(vs, 0) if vs else 0, "vote_ok": ok_of.get(vs) if vs else None,
                  "greedy_ok": ok_of.get(g["sig"]) if g and live(g) else None,
                  "gold_status": gold_status, "added": min(len(new), max_ff), "dropped_over_cap": max(0, len(new) - max_ff)}
    return base


def add_extra(rec: dict, extra: list[dict], ff: list[dict], src: str = "mut", max_extra: int = 8,
              same_sig: str = "drop") -> dict:
    """Add labelled extra candidates (e.g. arithmetic mutants of grammar SQL) to a merged record as src=<src>: one
    representative per new live sig (same text or, with same_sig drop, a sig already live in the record adds nothing).
    They never enter the ff vote; ff_sig_n = live ff samples with their sig. rank = 2000 + rank, no cost."""
    rec = dict(rec)
    cands = list(rec["cands"])
    texts = {c["sql"] for c in cands if c.get("sql")}
    live_sigs = {c["sig"] for c in cands if live(c)}
    sig_n = collections.Counter(c["sig"] for c in ff if live(c))
    classes = collections.defaultdict(list)
    for c in extra:
        if live(c) and c.get("sql") and c["sql"] not in texts and (same_sig == "keep" or c["sig"] not in live_sigs):
            classes[c["sig"]].append(c)
    new = []
    for sig, members in classes.items():
        rep = min(members, key=lambda m: m.get("rank", 0))
        e = {k: v for k, v in rep.items() if k != "cost"}
        e.update(src=src, rank=2000 + int(rep.get("rank", 0)), ff_n=0, ff_sig_n=sig_n.get(sig, 0), n_same_sig=len(members))
        new.append(e)
    new.sort(key=lambda e: e["rank"])
    cap = max_extra or len(new)  # 0: no cap
    rec["cands"] = cands + new[:cap]
    rec[src] = {"n": len(extra), "live": sum(live(c) for c in extra), "added": min(len(new), cap),
                "dropped_over_cap": max(0, len(new) - cap)}
    return rec


# ---- summary (labelled sets only) -----------------------------------------------------------------------------------

def summary(recs: list[dict], ff_lab: dict) -> dict:
    n = len(recs)
    if not n:
        return {}

    def pct(x):
        return round(100.0 * x / n, 2)

    def ffok(r):
        return any(c.get("ok") for c in ff_lab.get(r["id"], ("", []))[1])
    pool_or = sum(any(c.get("ok") for c in r["cands"] if c.get("src") not in ADDED_SRCS) for r in recs)
    union_or = sum(any(c.get("ok") for c in r["cands"] if c.get("src") != "mut") or ffok(r) for r in recs)
    all_or = sum(any(c.get("ok") for c in r["cands"]) or ffok(r) for r in recs)
    return {"n": n, "pool_oracle": pct(pool_or), "ff_oracle": pct(sum(ffok(r) for r in recs)),
            "union_oracle": pct(union_or), "with_extra_oracle": pct(all_or),
            "ff_greedy": pct(sum(bool(r["ff"].get("greedy_ok")) for r in recs)),
            "ff_vote": pct(sum(bool(r["ff"].get("vote_ok")) for r in recs)),
            "newly_reachable": union_or - pool_or,
            "mean_new_ff_entries": round(sum(r["ff"]["added"] for r in recs) / n, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", type=Path, nargs="*", default=[], help="gen.py outputs (several models: concatenated); "
                    "none: only --extra is merged into --pool")
    ap.add_argument("--pool", type=Path, help="existing pool jsonl; omitted: an ff-only labelled file")
    ap.add_argument("--ids", type=Path, help="manifest restricting the questions (e.g. heldout_v2/decision.json)")
    ap.add_argument("--confirm", action="store_true", help="allow RESERVED ids")
    ap.add_argument("--db-roots", nargs="+", required=True)
    ap.add_argument("--gold-seconds", type=float, default=30.0, help="heldout_v2 gold limit")
    ap.add_argument("--cand-seconds", type=float, default=3.0, help="gen-time candidate limit")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-ff", type=int, default=8, help="new ff result classes per question")
    ap.add_argument("--same-sig", choices=["drop", "keep"], default="drop")
    ap.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-gold", action="store_true", help="blind set: no ok labels, no summary")
    ap.add_argument("--extra", type=Path, help="optional extra candidates (pool schema, e.g. arithmetic mutants): "
                    "relabelled here, added as --extra-src representatives after the ff ones; a missing file is skipped")
    ap.add_argument("--extra-src", default="mut")
    ap.add_argument("--max-extra", type=int, default=0, help="new --extra result classes per question (0: no cap)")
    ap.add_argument("--prelabelled", action="store_true", help="--ff files are merge.py --samples outputs: keep "
                    "their labels (sql / sql_raw / runs / ok / sig) instead of executing again")
    ap.add_argument("--samples", action="store_true", help="without --pool: keep EVERY labelled ff sample as a "
                    "candidate (per-sample probes: P1 text, accuracy, pass@k), not one representative per sig")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ids = None
    if a.ids:
        man = json.loads(a.ids.read_text())
        ids = list(man["ids"] if isinstance(man, dict) else man)
    reserved = set(json.loads(RESERVED.read_text())["ids"]) if RESERVED.exists() else set()

    ff_recs = {}
    for f in a.ff:
        for r in map(json.loads, open(f)):
            if r["id"] in ff_recs:  # several generators: concatenate, ranks made unique per question
                off = 1 + max((c["rank"] for c in ff_recs[r["id"]]["cands"]), default=-1)
                ff_recs[r["id"]]["cands"] += [{**c, "rank": c["rank"] + off, "temp": c.get("temp")} for c in r["cands"]]
            else:
                ff_recs[r["id"]] = r
    pool = {r["id"]: r for r in map(json.loads, open(a.pool))} if a.pool else {}
    order = ids if ids is not None else list(dict.fromkeys(list(pool) + list(ff_recs)))
    order = [i for i in order if i in pool or i in ff_recs]
    if reserved & set(order) and not a.confirm:
        sys.exit(f"{len(reserved & set(order))} RESERVED ids in scope: pass --ids decision.json, or --confirm")

    paths = {}
    for root in a.db_roots:
        for p in sorted(Path(root).rglob("*.sqlite")):
            paths.setdefault(p.stem, str(p))
    if a.samples and a.pool:
        sys.exit("--samples is for ff-only labelled files (no --pool)")
    q_ff = [ff_recs[i] for i in order if i in ff_recs]
    if a.prelabelled:  # --samples outputs of this script: reuse their labels (no second execution pass)
        if any("runs" not in c for r in q_ff for c in r["cands"]):
            sys.exit("--prelabelled: some ff candidates carry no labels (not a merge.py --samples output)")
        lab = {r["id"]: ((r.get("ff") or {}).get("gold_status", "ok"), r["cands"]) for r in q_ff}
    else:
        lab = label(q_ff, paths, a)
    ex_lab = {}
    if a.extra and a.extra.exists():
        ex = []
        for r in map(json.loads, open(a.extra)):
            if r["id"] in pool or r["id"] in ff_recs:
                g = (pool.get(r["id"]) or ff_recs[r["id"]]).get("gold_sql")
                ex.append({**r, "gold_sql": g, "cands": [{k: v for k, v in c.items() if k not in
                           ("runs", "nonnull", "ok", "sig", "res", "rr", "sql_raw")} for c in r["cands"]]})
        ex_lab = label(ex, paths, argparse.Namespace(**{**vars(a), "normalize": False}))  # grammar-style text: kept
        print(f"EXTRA {len(ex)} records from {a.extra} (src {a.extra_src})", flush=True)
    elif a.extra:
        print(f"EXTRA {a.extra} not found: skipped", flush=True)
    out = []
    for i in order:
        q = ff_recs.get(i) or pool[i]
        st, ff = lab.get(i, ("no-ff", []))
        m = merge_record(pool.get(i), q, ff, st, a.max_ff, a.same_sig)
        if a.samples:
            m["cands"] = ff
        if i in ex_lab:
            m = add_extra(m, ex_lab[i][1], ff, a.extra_src, a.max_extra, a.same_sig)
        out.append(m)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"MERGED {len(out)} questions ({sum(i not in pool for i in order)} not in the pool) -> {a.out}", flush=True)
    if not a.no_gold:
        print(f"SUMMARY {json.dumps(summary(out, lab))}", flush=True)


if __name__ == "__main__":
    main()
