"""Arithmetic mutants as extra pool candidates (fix #1 of docs/paper_notes/research/generator_misses_2026-10-10.md).

The generator almost never writes a bare division by a constant (`/ 100`, `/ N`); for covered questions whose gold
does, the pool has the right query up to a scale factor. For each question we take the top live candidates (the first
--cap distinct result signatures by rank), rewrite one numeric projection of the outer SELECT at a time and execute
the rewrite on the question's SQLite DB with the gen-time labelling (scripts/val2/execq.py: 3 s per candidate, gold
10 s, sig = md5 of the result as a set of rows, ok = set equality with the gold result).

Mutation kinds (each applied to one numeric projection -- integer/real in the parent's first result row):
  x100      e * 100
  d100      CAST(e AS REAL) / 100
  d100i     e / 100                       (only when e is an integer column: integer division, as a gold may write it)
  castreal  every division inside e gets a CAST(numerator AS REAL)  (fixes integer-division percentages/ratios)
  pct       castreal(e) * 100             (only when the evidence says "percent"/"%")
  dN / xN   CAST(e AS REAL) / N and e * N, for every arithmetic constant N in the evidence (see ev_constants)
  dqN       CAST(e AS REAL) / N for small integers N in the question (opt-in: --question-numbers)

The rules were set from SQL semantics and the BIRD-train evidence (see ev_constants), not from which DECISION
questions they recover. A mutant is kept only when it runs, is non-null, and its result is new to the question's pool:
its sig differs from every pool sig and its row set is not equal (Python ==, so 1 == 1.0; compared by set_key) to a
parent's or an earlier mutant's. Kept mutants get src="mut", mut=<kind>, parent=<parent rank>, rank 200+, rr=None, cost=<parent cost>.

  uv run python scripts/pool/arith_mutants.py --pool runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl --decision \
      --out data/pool/decision_arith.jsonl
  uv run python scripts/pool/arith_mutants.py --pool runs/phase1/p1w/p1w/r1/cands_train.jsonl \
      --out data/pool/train_r1_arith.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/val2"))
import execq  # noqa: E402

DBROOT = ROOT / "data/raw/bird_train/train_databases"
GEN_SECONDS = 3.0
GOLD_SECONDS = 10.0
RANK0 = 200
FORBIDDEN = ("bird_dev", "gate.json", "sealed", "score_")

_NUM = r"(\d+(?:\.\d+)?)"
_PCT = re.compile(r"(?i)percent|%")
_TYPES = re.compile(r"\d+ rows? x \d+ cols? \(([^)]*)\)")


def _guard(p) -> Path:
    p = Path(p)
    assert not any(f in str(p) for f in FORBIDDEN), p
    return p


# ------------------------------------------------------------------------------------------------------ constants
def _strip_literals(s: str) -> str:
    """Quoted strings and date-like tokens (2018/8/1, 4/4/20, 2011-09-01) removed, so their digits are not constants."""
    s = re.sub(r"'[^']*'|\"[^\"]*\"|`[^`]*`", " '' ", s)
    return re.sub(r"\b\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?\b", " DATE ", s)


def ev_constants(evidence: str | None) -> set[float]:
    """Numeric constants used arithmetically in the evidence: an operand of * or /, the last argument of a
    DIVIDE/MULTIPLY(...) call, or after "divided by / multiplied by / times". 0, 1 and 100 are excluded (100 has its
    own kinds). On BIRD-train this finds e.g. MULTIPLY(SUM(num_sales), 100000), Divide(Count(ward_no), 12),
    DIVIDE(COUNT(Code), 31); comparison literals (num_sales < 0.2) and dates are not picked up."""
    s = _strip_literals(evidence or "")
    out = set(re.findall(r"[*/]\s*\(?\s*" + _NUM + r"\b", s))
    out |= set(re.findall(r"\b" + _NUM + r"\s*\)?\s*[*/]", s))
    if re.search(r"(?i)\b(?:divide|multiply|div|mul|product)\s*\(", s):
        out |= set(re.findall(r",\s*" + _NUM + r"\s*\)", s))
    out |= set(re.findall(r"(?i)\b(?:divided by|divide by|multiplied by|multiply by|times)\s+" + _NUM + r"\b", s))
    return {float(x) for x in out} - {0.0, 1.0, 100.0}


def question_numbers(question: str | None) -> set[float]:
    """Small integers 2..100 written in the question (not years, not dates, not quoted)."""
    s = _strip_literals(question or "")
    return {float(x) for x in re.findall(r"(?<![\d.])(\d{1,3})(?![\d.])", s) if 2 <= int(x) <= 100} - {100.0}


def _lit(n: float):
    return exp.Literal.number(int(n) if float(n).is_integer() else n)


# ------------------------------------------------------------------------------------------------------- mutation
def result_kinds(res: str | None) -> list[str] | None:
    """Column kinds of the first result row from execq.result_preview text, e.g. ['integer', 'text']."""
    m = _TYPES.search(res or "")
    return [k.strip() for k in m.group(1).split(",")] if m else None


def _unalias(e):
    return e.this if isinstance(e, exp.Alias) else e


def _paren(e):
    return e if isinstance(e, (exp.Column, exp.Literal, exp.Func, exp.Paren, exp.Cast)) else exp.Paren(this=e)


def _real(e):
    return exp.Cast(this=e, to=exp.DataType.build("REAL"))


def _is_real(e) -> bool:
    """e already carries a REAL operand: a CAST to REAL/FLOAT/DOUBLE/NUMERIC or a decimal literal anywhere in it."""
    for n in e.walk():
        if isinstance(n, exp.Cast) and n.to.this in (exp.DataType.Type.FLOAT, exp.DataType.Type.DOUBLE,
                                                     exp.DataType.Type.DECIMAL):
            return True
        if isinstance(n, exp.Literal) and not n.is_string and "." in n.this:
            return True
    return False


def _castreal(e):
    """Every Div in e gets a REAL numerator, unless the numerator already is a CAST(... AS REAL/FLOAT) or a decimal
    literal; None when nothing changes."""
    e = e.copy()
    changed = False
    for d in list(e.find_all(exp.Div)):
        num = d.this
        if _is_real(num) or _is_real(d.expression):
            continue
        d.set("this", _real(_paren(num.copy())))
        changed = True
    return e if changed else None


def projection_mutants(e, kind_int: bool, consts: set[float], pct: bool, qnums: set[float]) -> list[tuple[str, object]]:
    """[(kind, new expression)] for one projection expression e (unaliased)."""
    out = [("x100", exp.Mul(this=_paren(e.copy()), expression=_lit(100))),
           ("d100", exp.Div(this=_real(e.copy()), expression=_lit(100), typed=True))]
    if kind_int:
        out.append(("d100i", exp.Div(this=_paren(e.copy()), expression=_lit(100), typed=True)))
    cr = _castreal(e)
    if cr is not None:
        out.append(("castreal", cr))
    if pct:
        base = cr if cr is not None else _real(e.copy())
        out.append(("pct", exp.Mul(this=_paren(base.copy()), expression=_lit(100))))
    for n in sorted(consts):
        tag = f"{n:g}"
        out.append((f"d{tag}", exp.Div(this=_real(e.copy()), expression=_lit(n), typed=True)))
        out.append((f"x{tag}", exp.Mul(this=_paren(e.copy()), expression=_lit(n))))
    for n in sorted(qnums - consts):
        out.append((f"dq{n:g}", exp.Div(this=_real(e.copy()), expression=_lit(n), typed=True)))
    return out


def mutants(sql: str, res: str | None, evidence: str | None, question: str | None = None,
            question_nums: bool = False, max_proj: int = 2) -> list[tuple[str, str]]:
    """[(kind, mutated sql)] for one candidate. Only outer-SELECT projections whose parent result kind is integer or
    real (or null/unknown) are mutated, at most max_proj of them, one at a time. [] when the SQL does not parse as a
    SELECT or projects a star."""
    try:
        t = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    if not isinstance(t, exp.Select) or any(isinstance(_unalias(p), exp.Star) or
                                            (isinstance(_unalias(p), exp.Column) and isinstance(_unalias(p).this, exp.Star))
                                            for p in t.expressions):
        return []
    kinds = result_kinds(res)
    if kinds is not None and len(kinds) != len(t.expressions):
        return []
    consts = ev_constants(evidence)
    pct = bool(_PCT.search(evidence or ""))
    qnums = question_numbers(question) if question_nums else set()
    out, seen = [], {sql}
    done = 0
    for i, p in enumerate(t.expressions):
        k = kinds[i] if kinds else "null"
        if k not in ("integer", "real", "null"):
            continue
        if done >= max_proj:
            break
        done += 1
        inner = _unalias(p)
        for kind, new in projection_mutants(inner, k == "integer", consts, pct, qnums):
            t2 = t.copy()
            p2 = t2.expressions[i]
            if isinstance(p2, exp.Alias):
                p2.set("this", new)
            else:
                p2.replace(new)
            try:
                s = t2.sql(dialect="sqlite")
            except Exception:
                continue
            if s not in seen:
                seen.add(s)
                out.append((kind if len(t.expressions) == 1 else f"{kind}@{i}", s))
    return out


def parents(cands: list[dict], cap: int) -> list[dict]:
    """The first `cap` live candidates with distinct result sigs, by rank."""
    out, sigs = [], set()
    for c in sorted((c for c in cands if c.get("sql")), key=lambda c: c.get("rank", 0)):
        if c.get("runs") and c.get("nonnull") and c.get("sig") not in sigs:
            sigs.add(c["sig"])
            out.append(c)
            if len(out) >= cap:
                break
    return out


# ------------------------------------------------------------------------------------------------------ execution
def _norm_val(v):
    return int(v) if isinstance(v, float) and v.is_integer() and abs(v) < 2 ** 53 else v


def set_key(rows) -> str:
    """Fingerprint of the result as a set of rows under Python equality (1 == 1.0), so two results with the same
    key are the same EX answer; keeps memory flat (whole row sets of large results are not held)."""
    gs = {tuple(_norm_val(v) for v in r) for r in rows}
    return execq.hashlib.md5(repr(sorted(map(repr, gs))).encode()).hexdigest()


def exec_task(task):
    """(db path, [(id, gold_sql, [parent sql], [(parent_idx, kind, sql)], pool sigs)]) ->
    [(id, gold_ok, [(parent_idx, kind, sql, runs, nonnull, ok, sig, res)])] -- kept mutants only."""
    path, qs = task
    try:
        g_rows = execq.connect(path, GOLD_SECONDS)
        c_rows = execq.connect(path, GEN_SECONDS)
    except Exception:
        return [(q[0], False, []) for q in qs]
    out = []
    for qid, gold_sql, psqls, muts, pool_sigs in qs:
        try:
            ref = set(map(tuple, g_rows(gold_sql)))
        except Exception:
            out.append((qid, False, []))
            continue
        seen_keys = set()
        for s in psqls:
            try:
                seen_keys.add(set_key(c_rows(s)))
            except Exception:
                pass
        sigs = set(pool_sigs)
        kept = []
        for pi, kind, s in muts:
            try:
                got = c_rows(s)
            except Exception:
                continue
            gs = set(map(tuple, got))
            nonnull = any(v is not None for x in got for v in x)
            if not got or not nonnull:
                continue
            sig, key = execq.sig_of(got), set_key(got)
            if sig in sigs or key in seen_keys:
                continue
            sigs.add(sig)
            seen_keys.add(key)
            kept.append((pi, kind, s, True, nonnull, gs == ref, sig, execq.result_preview(got)))
        out.append((qid, True, kept))
    return out


def build(records: list[dict], cap: int = 8, question_nums: bool = False, workers: int = 6, chunk: int = 10):
    """records (pool schema) -> {id: [new candidate dicts]}, plus {id: gold executed?}."""
    paths = {p.stem: str(p) for p in DBROOT.glob("*/*.sqlite")}
    by_db = defaultdict(list)
    meta = {}
    for r in records:
        ps = parents(r["cands"], cap)
        muts = []
        for pi, c in enumerate(ps):
            for kind, s in mutants(c["sql"], c.get("res"), r.get("context"), r.get("question"), question_nums):
                muts.append((pi, kind, s))
        meta[r["id"]] = ps
        if muts and r["db_id"] in paths:
            sigs = sorted({c.get("sig") for c in r["cands"] if c.get("sig")})
            by_db[r["db_id"]].append((r["id"], r["gold_sql"], [c["sql"] for c in ps], muts, sigs))
    tasks = [(paths[db], lst[i:i + chunk]) for db, lst in by_db.items() for i in range(0, len(lst), chunk)]
    res = []
    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor  # a dead worker raises instead of hanging (Pool does)
        from multiprocessing import get_context
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as p:
            for part in p.map(exec_task, tasks):
                res += part
    else:
        for t in tasks:
            res += exec_task(t)
    out = {}
    for qid, _gold_ok, kept in res:
        ps = meta[qid]
        out[qid] = [{"rank": RANK0 + j, "cost": ps[pi].get("cost"), "sql": s, "runs": runs, "nonnull": nn, "ok": ok,
                     "sig": sig, "res": prev, "src": "mut", "rr": None, "mut": kind, "parent": ps[pi].get("rank")}
                    for j, (pi, kind, s, runs, nn, ok, sig, prev) in enumerate(kept)]
    return out


def report(records: list[dict], extra: dict, label: str) -> dict:
    n = len(records)
    had = sum(any(c.get("ok") for c in r["cands"]) for r in records)
    gain = [r["id"] for r in records if not any(c.get("ok") for c in r["cands"])
            and any(c["ok"] for c in extra.get(r["id"], []))]
    cov = [r for r in records if r.get("covered")]
    gain_cov = [r for r in cov if r["id"] in set(gain)]
    added = [len(extra.get(r["id"], [])) for r in records]
    kinds = Counter(c["mut"].split("@")[0] for v in extra.values() for c in v)
    gain_kinds = Counter(next(c["mut"].split("@")[0] for c in extra[q] if c["ok"]) for q in gain)
    pool_n = sum(len(r["cands"]) for r in records)
    s = {"label": label, "questions": n, "oracle_before": had, "oracle_after": had + len(gain), "gain": len(gain),
         "covered": len(cov), "gain_covered": len(gain_cov),
         "added_total": sum(added), "added_per_q_mean": round(sum(added) / max(n, 1), 2),
         "q_with_added": sum(a > 0 for a in added), "added_max": max(added or [0]),
         "pool_cands_before": pool_n, "added_right_mutants": sum(c["ok"] for v in extra.values() for c in v),
         "kinds": dict(kinds.most_common()), "gain_by_first_right_kind": dict(gain_kinds.most_common())}
    print(json.dumps(s, indent=1))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decision", action="store_true", help="keep only DECISION ids (RESERVED is never read)")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--question-numbers", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report", default=None, help="write the summary json here")
    a = ap.parse_args()
    keep = None
    if a.decision:
        keep = set(json.load(open(_guard(ROOT / "data/sys1/heldout_v2/decision.json")))["ids"])
        keep -= set(json.load(open(_guard(ROOT / "data/sys1/heldout_v2/reserved.json")))["ids"])
    records = []
    for line in open(_guard(a.pool)):
        r = json.loads(line)
        if keep is None or r["id"] in keep:
            records.append(r)
    t0 = time.time()
    extra = build(records, a.cap, a.question_numbers, a.workers)
    out = _guard(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            if extra.get(r["id"]):
                f.write(json.dumps({k: r[k] for k in r if k != "cands"} | {"cands": extra[r["id"]]}) + "\n")
    s = report(records, extra, a.pool)
    s["seconds"] = round(time.time() - t0, 1)
    print(f"wrote {out} ({sum(1 for v in extra.values() if v)} questions) in {s['seconds']} s")
    if a.report:
        json.dump(s, open(_guard(a.report), "w"), indent=1)


if __name__ == "__main__":
    main()
