"""Step B, part 2: question writing with Claude Code subagents (no paid API).

  python scripts/qgen.py batches --plans data/qgen/plans.jsonl --cards DIR... --out data/qgen/batches --size 20
  python scripts/qgen.py checks  --batches data/qgen/batches            # after writers ran: build checker inputs
  python scripts/qgen.py grade   --batches data/qgen/batches --out data/qgen/accepted.jsonl

A WRITER subagent gets writer_NNNN.json (per plan: the schema of the tables it touches + 3
look-alike distractor tables from other databases, the plan in plain English, and its first
result rows) and writes questions_NNNN.json. A separate CHECKER subagent gets check_NNNN.json
(each question + the same schema, but NOT the plan) and writes checked_NNNN.json: the columns
and literals it thinks each question asks for. `grade` keeps a question only if the checker's
reading matches the plan (same shown columns, same filter columns, same literal values).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

WRITER_INSTRUCTIONS = """You write the questions a business analyst would type into a data tool.
For EACH plan in this file, write 6 questions that ask for exactly that plan's result:
  - 2 "plain": clear, complete questions.
  - 2 "casual": short, terse or chatty, the way people type into a search box or Slack.
  - 2 "synonym": avoid every table and column name in the schema. Use everyday words or business
    synonyms instead (e.g. "staff number" for employee_id, "headcount" for count of employees).
Rules:
  - NEVER write table or column names, SQL words or dotted names (no "time_slot.day", no "grouped by",
    no "ordered by row count"). Write what a person would type: "Which day has the most time slots?"
  - Use the literal values exactly as given (same spelling and case), never invent new ones.
  - Ask for exactly what the plan shows: no extra columns, no extra or missing conditions.
  - Counting: "the number of distinct X" means "how many different X"; "the number of rows" means
    "how many ..." (each record counts). Keep this distinction in every style.
  - Show exactly the columns the plan shows, in a natural order: if the plan shows a name and a count,
    the question must ask for both.
  - Use the result rows only to understand the data; never put result values in the question.
  - If a plan does not correspond to any sensible question a person would ask, skip it.
Example. Plan: "show time_slot.day; for each time_slot.day; sorted by the number of rows of time slot highest first; only the first one"
  BAD  plain: "Show the day from time_slot, grouped by day, ordered by row count descending, first result."
  GOOD plain: "Which day of the week has the most time slots?"
  GOOD casual: "busiest day for time slots?"
  GOOD synonym: "On what weekday are the most class periods scheduled?"
Output: JSON {"items": [{"plan_id": ..., "skip": false, "questions": [{"style": "plain", "text": "..."}, ...]}, ...]}
"""

CHECKER_INSTRUCTIONS = """You check generated analytics questions. Each item has a schema and several questions about it.
For EACH question, read it together with its item's schema and say what a correct answer must contain,
using only the schema's table.column names:
  - "shown": the columns the answer must show (use "<table>.*" for "number of rows of <table>"),
  - "aggregations": for each shown column, one of none/count/count_distinct/sum/avg/min/max,
  - "filter_columns": columns the question restricts on, and "literals": the literal values it uses,
  - "group_by", "order_by" (column or null), "limit" (number or null).
You do NOT see the intended answer. Judge only from the question and schema.
Judge each question independently (the questions of one item may differ from each other).
Output: JSON {"items": [{"qid": ..., "shown": [...], "aggregations": [...], "filter_columns": [...],
"literals": [...], "group_by": [...], "order_by": ..., "limit": ...}, ...]}   (one item per qid)
"""


def table_block(card: dict, tname: str, samples: int = 3) -> dict:
    t = next(x for x in card["tables"] if x["name"] == tname)
    return {"table": tname, "description": t.get("description", ""),
            "columns": [{"name": c["name"], "type": c["type"], "description": c.get("description", ""),
                         "examples": c.get("samples", [])[:samples]} for c in t["columns"]]}


def plan_tables(p: dict) -> list[str]:
    ts = [x["col"]["table"] for x in p["projections"]] + [f["col"]["table"] for f in p.get("filters") or []]
    ts += [g["table"] for g in p.get("group_by") or []] + [h["col"]["table"] for h in p.get("having") or []]
    if p.get("order"):
        ts.append(p["order"]["col"]["table"])
    for f in p.get("filters") or []:
        if f.get("sub"):
            ts += plan_tables(f["sub"])
    return list(dict.fromkeys(ts + list(p.get("require") or [])))


def _join_tables(card: dict, p: dict) -> list[str]:
    """The plan's tables plus the bridge tables its joins pass through (what the compiler uses)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from luthor.sq.plan import Plan, join_tree
    from luthor.sq.schema_card import SchemaCard

    try:
        sc = SchemaCard.load_dict(card) if hasattr(SchemaCard, "load_dict") else None
    except Exception:
        sc = None
    if sc is None:
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(card, f)
        sc = SchemaCard.load(f.name)
    plan = Plan.from_dict(p)
    base = plan_tables(p)
    try:
        tables, _ = join_tree(sc.graph(), [t for t in base if t in sc.graph().columns])
    except Exception:
        tables = base
    return list(dict.fromkeys(base + list(tables)))


def cmd_batches(a):
    cards = {}
    for d in a.cards:
        for f in Path(d).glob("*.json"):
            cards[f.stem] = json.loads(f.read_text())
    rows = [json.loads(l) for l in open(a.plans)]
    rnd = random.Random(0)
    other = [(db, t["name"]) for db, c in cards.items() for t in c["tables"] if not t["name"].lower().startswith("sqlite_")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for i, r in enumerate(rows):
        card = cards[r["db_id"]]
        tabs = [table_block(card, t) for t in _join_tables(card, r["plan"])]
        distract = []
        for db, t in rnd.sample(other, min(len(other), 12)):
            if db != r["db_id"] and len(distract) < 3:
                distract.append(table_block(cards[db], t))
        schema = tabs + distract
        rnd.shuffle(schema)
        items.append({"plan_id": f"p{i:06d}", "db_id": r["db_id"], "schema": schema, "plan": r["english"],
                      "result_first_rows": r["result_head"]})
        r["plan_id"] = f"p{i:06d}"
    (out / "plans_with_ids.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    for b in range(0, len(items), a.size):
        (out / f"writer_{b // a.size:04d}.json").write_text(json.dumps(
            {"instructions": WRITER_INSTRUCTIONS, "plans": items[b:b + a.size]}, indent=1, default=str))
    print(f"{len(items)} plans -> {(len(items) + a.size - 1) // a.size} writer batches in {out}")


def cmd_checks(a):
    d = Path(a.batches)
    for wf in sorted(d.glob("questions_*.json")):
        n = wf.stem.split("_")[1]
        w = json.loads((d / f"writer_{n}.json").read_text())
        schema = {p["plan_id"]: p["schema"] for p in w["plans"]}
        qs = json.loads(wf.read_text())["items"]
        items, nq = [], 0
        for it in qs:
            if it.get("skip") or not it.get("questions"):
                continue
            items.append({"schema": schema[it["plan_id"]],
                          "questions": [{"qid": f"{it['plan_id']}_{k}", "question": q["text"]}
                                        for k, q in enumerate(it["questions"])]})
            nq += len(it["questions"])
        (d / f"check_{n}.json").write_text(json.dumps({"instructions": CHECKER_INSTRUCTIONS, "items": items},
                                                      indent=1, default=str))
        print(f"check_{n}.json: {nq} questions")


_LEAKY = re.compile(r"\b\w+_\w+\b|\b[A-Za-z]+\.[A-Za-z_]+\b|grouped by|group by|ordered by|order by|descending|ascending"
                    r"|\bnumber of rows\b|\brows? of\b|\bselect\b|\bwhere\b.*\b(=|<|>)", re.I)


def leaky(text: str, literals: set[str]) -> bool:
    """A question that copies schema identifiers or SQL wording is not how people ask. Literal
    values (which may contain underscores) are allowed."""
    t = text
    for v in literals:
        t = t.replace(v, " ").replace(v.lower(), " ")
    return bool(_LEAKY.search(t))


def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower())


def grade_one(p: dict, q: dict, c: dict) -> tuple[bool, str]:
    """Does the checker's blind reading of question q match plan p? (ok, failed checks)"""
    shown = {f"{x['col']['table']}.{x['col']['column']}".lower() for x in p["projections"]}
    fcols = {f"{f['col']['table']}.{f['col']['column']}".lower() for f in p.get("filters") or []}
    lits = {_norm(v) for f in p.get("filters") or [] for v in f.get("values") or []}
    got_shown = {str(s).lower() for s in c.get("shown") or []}
    got_f = {str(s).lower() for s in c.get("filter_columns") or []}
    got_l = {_norm(v) for v in c.get("literals") or []}
    aggs = sorted("count" if x["col"]["column"] == "*" else x["agg"] for x in p["projections"])
    got_aggs = sorted(str(x).lower() for x in c.get("aggregations") or [])
    grp = {f"{g['table']}.{g['column']}".lower() for g in p.get("group_by") or []}
    got_grp = {str(x).lower() for x in c.get("group_by") or []}
    lim_ok = (p.get("limit") or None) == (c.get("limit") or None)
    ord_ok = (p.get("order") is None) == (not c.get("order_by"))
    checks = [("shown", got_shown == shown), ("filters", fcols <= got_f), ("literals", lits <= got_l),
              ("aggregations", aggs == got_aggs), ("group_by", grp == got_grp), ("limit", lim_ok), ("order", ord_ok)]
    raw_lits = {str(v) for f in p.get("filters") or [] for v in f.get("values") or []}
    checks.append(("wording", not leaky(q["text"], raw_lits)))
    bad = [k for k, v in checks if not v]
    return not bad, ",".join(bad)


def cmd_verifies(a):
    """verify_NNNN.json for the second-stage verifier: only questions the blind checker accepted and the
    quality rules pass (fan-out, identifiers, LIMIT without ORDER BY). vid = <plan_id>_<k>."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from quality import fix_fan_out, identifier_leak, limit_without_order
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from luthor.sq.schema_card import SchemaCard
    d = Path(a.batches)
    plans = {json.loads(l)["plan_id"]: json.loads(l) for l in open(d / "plans_with_ids.jsonl")}
    cards, n_in, n_out = {}, 0, 0
    for qf in sorted(d.glob("questions_*.json")):
        n = qf.stem.split("_")[1]
        cf, wf = d / f"checked_{n}.json", d / f"writer_{n}.json"
        if not cf.exists():
            continue
        try:
            verdicts = {c["qid"]: c for c in json.loads(cf.read_text())["items"] if c.get("qid")}
            w = {p["plan_id"]: p for p in json.loads(wf.read_text())["plans"]}
            items = json.loads(qf.read_text())["items"]
        except Exception:
            continue
        out = []
        for it in items:
            pid = it.get("plan_id")
            if it.get("skip") or pid not in plans:
                continue
            p = plans[pid]
            card = cards.setdefault(p["db_id"], SchemaCard.load(Path(a.cards) / f"{p['db_id']}.json"))
            lits = {str(v) for f in p["plan"].get("filters") or [] for v in f.get("values") or []}
            for k, q in enumerate(it.get("questions") or []):
                c = verdicts.get(f"{pid}_{k}")
                if c is None:
                    continue
                n_in += 1
                ok, _ = grade_one(p["plan"], q, c)
                fixed, st = fix_fan_out(p["plan"], card)
                if not ok or fixed is None or identifier_leak(q["text"], card, lits) or limit_without_order(p["plan"]):
                    continue
                out.append({"vid": f"{pid}_{k}", "question": q["text"], "sql": p["sql"], "plan_english": w[pid]["plan"],
                            "schema": w[pid]["schema"]})
                n_out += 1
        if out:
            (d / f"verify_{n}.json").write_text(json.dumps({"items": out}, default=str))
    print(f"{d}: {n_out} of {n_in} checked questions go to the verifier")


def cmd_grade(a):
    d = Path(a.batches)
    plans = {json.loads(l)["plan_id"]: json.loads(l) for l in open(d / "plans_with_ids.jsonl")}
    acc, rej, out = 0, 0, []
    for cf in sorted(d.glob("checked_*.json")):
        n = cf.stem.split("_")[1]
        qs = {f"{it['plan_id']}_{k}": (it["plan_id"], q) for it in json.loads((d / f"questions_{n}.json").read_text())["items"]
              if not it.get("skip") for k, q in enumerate(it["questions"])}
        for c in json.loads(cf.read_text())["items"]:
            if c.get("qid") not in qs:
                continue
            pid, q = qs[c["qid"]]
            ok, reason = grade_one(plans[pid]["plan"], q, c)
            p = plans[pid]["plan"]
            out.append({"plan_id": pid, "db_id": plans[pid]["db_id"], "question": q["text"], "style": q["style"],
                        "accepted": ok, "reject_reason": reason, "plan": p, "sql": plans[pid]["sql"]})
            acc += ok
            rej += not ok
    Path(a.out).write_text("".join(json.dumps(r, default=str) + "\n" for r in out))
    print(f"accepted {acc}, rejected {rej} ({acc / max(1, acc + rej):.0%} kept) -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("batches"); b.add_argument("--plans", required=True); b.add_argument("--cards", nargs="+", required=True)
    b.add_argument("--out", required=True); b.add_argument("--size", type=int, default=20)
    c = sub.add_parser("checks"); c.add_argument("--batches", required=True)
    v = sub.add_parser("verifies"); v.add_argument("--batches", required=True); v.add_argument("--cards", required=True)
    g = sub.add_parser("grade"); g.add_argument("--batches", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    {"batches": cmd_batches, "checks": cmd_checks, "grade": cmd_grade, "verifies": cmd_verifies}[a.cmd](a)
