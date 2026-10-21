"""Mac-side verification of teacher SQL.

  execute     read-only SQLite, a time limit per query, one process per chunk of a database's queries
  signature   order-free fingerprint of a result set with BIRD's set semantics (1 == 1.0; floats to 6 places)
  gold match  for questions with gold SQL: keep a sample only when its signature equals the gold's
  vote        for gold-less questions: the majority signature over the samples, with its agreement share
  plan rows   the kept SQL -> Plan (luthor.sq.convert, grammar v3 by default, v4 behind --grammar v4 /
              LUTHOR_GRAMMAR=v4), literals grounded, compiled, re-executed: a row exactly as
              scripts/prepare_plans.py writes one (what train_sq2.py --plans loads), never trusted on the parser's word
"""
from __future__ import annotations

import collections
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

from ..sqlw.execute import connect

REPO = Path(__file__).resolve().parents[3]
GRAMMARS = ("v3", "v4")


def _norm(v):
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 6)
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def result_sig(rows) -> str:
    s = {tuple(_norm(v) for v in r) for r in rows}
    return hashlib.md5(repr(sorted(s, key=repr)).encode()).hexdigest()[:16]


def _exec_chunk(task):
    path, items, seconds = task
    try:
        run = connect(path, seconds)
    except Exception as e:
        return [(k, "error:open " + str(e)[:60], "", 0) for k, _ in items]
    out, seen = [], {}
    for k, sql in items:
        if not sql:
            out.append((k, "empty", "", 0))
            continue
        if sql in seen:
            out.append((k, *seen[sql]))
            continue
        try:
            rows = run(sql)
            r = ("ok", result_sig(rows), len(rows))
        except Exception as e:
            msg = str(e)
            r = ("timeout" if "interrupted" in msg else "error:" + msg[:80], "", 0)
        seen[sql] = r
        out.append((k, *r))
    return out


def execute(items, paths: dict, seconds: float = 10.0, workers: int = 1) -> dict:
    """[(key, db_id, sql)] -> {key: (status, sig, n_rows)}; status 'ok' | 'no-db' | 'empty' | 'timeout' | 'error:..'"""
    by_db = collections.defaultdict(list)
    for k, db, sql in items:
        by_db[db].append((k, sql))
    res, tasks = {}, []
    for db, lst in by_db.items():
        if db not in paths:
            res.update({k: ("no-db", "", 0) for k, _ in lst})
            continue
        tasks += [(paths[db], lst[i:i + 40], seconds) for i in range(0, len(lst), 40)]
    if workers > 1 and len(tasks) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as p:
            for part in p.imap_unordered(_exec_chunk, tasks):
                res.update({k: tuple(v) for k, *v in part})
    else:
        for t in tasks:
            res.update({k: tuple(v) for k, *v in _exec_chunk(t)})
    return res


def vote(sigs: list[str]) -> tuple[str, float, int]:
    """(majority signature, agreement = its share of ALL samples, votes). Failed samples ('') count against."""
    good = [s for s in sigs if s]
    if not good:
        return "", 0.0, 0
    sig, n = collections.Counter(good).most_common(1)[0]
    return sig, n / len(sigs), n


def db_paths(roots) -> dict[str, str]:
    """{db_id: path}: <root>/<db>/<db>.sqlite first (cheap), then any *.sqlite under the root."""
    out: dict[str, str] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.glob("*/*.sqlite"):
            out.setdefault(p.stem, str(p))
        for p in root.glob("*.sqlite"):
            out.setdefault(p.stem, str(p))
    return out


# ---- grammar ----------------------------------------------------------------------------------------------
def set_grammar(name: str | None = None, templates: Path | None = None) -> str:
    """Grammar for plan conversion. v3 (default): convert.V4 emptied, which the v4 converter documents as
    grammar v3 behaviour. v4: every v4 construct on (convert.V4_ALL). Without the v4 converter merged, v4 is
    refused rather than silently falling back. templates: grammar-v3 expression templates the trainer knows
    (data/sys1/templates.json + _nums.json), as prepare_plans.py --allowed-templates."""
    import json

    import luthor.sq.convert as cv
    name = name or os.environ.get("LUTHOR_GRAMMAR", "v3")
    if name not in GRAMMARS:
        raise ValueError(f"grammar {name}: expected one of {GRAMMARS}")
    if hasattr(cv, "V4_ALL"):
        cv.V4 = set(cv.V4_ALL) if name == "v4" else set()
    elif name == "v4":
        raise RuntimeError("grammar v4 requested but luthor.sq.convert has no V4 switch yet")
    if templates is not None and Path(templates).exists():
        cv.ALLOWED_TEMPLATES = set(json.loads(Path(templates).read_text()))
        nf = Path(templates).with_name(Path(templates).stem + "_nums.json")
        cv.ALLOWED_NUMS = set(json.loads(nf.read_text())) if nf.exists() else None
    return name


# ---- plan rows ----------------------------------------------------------------------------------------------
def _prepare_plans():
    sp = str(REPO / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    import prepare_plans
    return prepare_plans


def plan_row(sql: str, question: str, context: str, card, graph, con, vindex) -> tuple[dict | None, str]:
    """prepare_plans._one for one (question, SQL), minus twins: ({outcome, plan, compiled_sql, value_labels,
    limit_label, empty_result, db_values}, "ok") or (None, reason)."""
    from ..sq.candidates import find_candidates, ground
    from ..sq.convert import convert
    from ..sq.plan import CompileError, compile_plan
    pp = _prepare_plans()
    conv = convert(sql, graph)
    if conv.plan is None:
        return None, "unsupported:" + conv.reason.split(":")[0]
    plan = conv.plan
    text = question + (f"\nContext: {context}" if context else "")
    db_values = vindex.match(text) if vindex is not None else []
    cands = find_candidates(text, card, extra=db_values)
    value_labels: list = []
    if not pp._ground_plan(plan, cands, value_labels, top=True):
        return None, "ungrounded_value"
    limit_label = None
    if plan.limit is not None and plan.limit != 1:
        limit_label = ground(plan.limit, "", cands)
        if limit_label is None:
            return None, "ungrounded_limit"
    try:
        compiled = compile_plan(plan, graph)
    except CompileError as e:
        return None, "compile:" + str(e).split(" ")[0]
    try:
        ref, got = pp.run(con, sql, 10.0), pp.run(con, compiled, 10.0)
    except sqlite3.Error:
        return None, "exec_error"
    if pp.norm_rows(got) != pp.norm_rows(ref):
        return None, "exec_mismatch"
    empty = len(ref) == 0 or all(all(v in (None, 0) for v in r) for r in ref)
    return {"outcome": "plan", "plan": plan.to_dict(), "compiled_sql": compiled, "value_labels": value_labels,
            "limit_label": limit_label, "empty_result": empty, "db_values": db_values}, "ok"


def plan_rows_db(task):
    """(db_path, card_path, [(key, question, context, [sql, ...])], grammar, templates) -> [(key, row|None, reasons)]:
    the first SQL in each list that becomes a verified plan wins."""
    import random  # noqa: F401  (prepare_plans helpers expect a seeded module state only for twins)

    from ..sq.candidates import ValueIndex
    from ..sq.schema_card import SchemaCard
    db_path, card_path, items, grammar, templates = task
    set_grammar(grammar, Path(templates) if templates else None)
    card = SchemaCard.load(Path(card_path))
    graph = card.graph()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    try:
        vindex = ValueIndex.build(con, card)
    except Exception:
        vindex = None
    out = []
    for key, q, ctx, sqls in items:
        reasons, row = [], None
        for s in sqls:
            try:
                row, why = plan_row(s, q, ctx, card, graph, con, vindex)
            except Exception as e:  # one bad row must not stop a database
                row, why = None, "error:" + type(e).__name__
            reasons.append(why)
            if row is not None:
                row["sql_index"] = len(reasons) - 1
                break
        out.append((key, row, reasons))
    con.close()
    return out
