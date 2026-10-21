"""Direct-SQL curriculum for the writer (round 5, step 2): SynSQL questions the slot grammar cannot
express (subqueries, HAVING, windows, set operations, multi-column output, ...), tagged by construct.

The round-4 writer's generated and SynSQL training rows came almost entirely from slot-expressible plans,
so they never taught these constructs. Rows here pass the round-3 SynSQL style rules (no vague /
metaphorical / dialogue questions, no schema identifiers in the question, no LIMIT without ORDER BY),
must run and return a non-empty result, and must not overlap a sealed question. A stratified sample goes
into verify packs (data/claude_verify/pack_NNNN.json, source "curr_synsql") for the usual checker
(Gemini / Codex on public data, Claude only where they disagree); only verified rows enter training.

  python scripts/curriculum_prep.py --total 12000
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

import ijson

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_mix3 import EVAL, V3  # noqa: E402
from prepare_plans import dedup_against  # noqa: E402
from quality import identifier_leak  # noqa: E402

from luthor.sq.schema_card import SchemaCard  # noqa: E402

BAD = {"Vague", "Metaphorical", "Multi-turn Dialogue"}
DIALOG = re.compile(r"\b(User|Assistant)\s*:|SQL|query", re.I)
DBS = Path("data/raw/synsql/databases")
OUT = Path("data/claude_verify")

TAGS = [  # (tag, pattern on lower-cased SQL); a row gets every tag that matches
    ("window", r"\bover\s*\("),
    ("cte", r"^\s*with\b"),
    ("set_op", r"\b(union|intersect|except)\b"),
    ("anti_join", r"\bnot\s+(exists|in)\s*\(\s*select\b|\bleft\s+join\b.*\bis\s+null\b"),
    ("correlated", r"\(\s*select\b[^()]*\bwhere\b[^()]*\b(t\d|[a-z_]+)\.\w+\s*=\s*(t\d|[a-z_]+)\.\w+"),
    ("subquery", r"\(\s*select\b"),
    ("having", r"\bhaving\b"),
    ("case", r"\bcase\s+when\b"),
    ("date_fn", r"\b(strftime|julianday|date|datetime)\s*\("),
    ("count_distinct", r"count\s*\(\s*distinct\b"),
    ("multi_order", r"\border\s+by\b[^)]*,"),
]
WEIGHT = {"window": 3, "cte": 2, "set_op": 2, "anti_join": 3, "correlated": 3, "date_fn": 2, "case": 2,
          "count_distinct": 2, "subquery": 1, "having": 1, "multi_order": 1}


def tags_of(sql: str) -> list[str]:
    s = " ".join(sql.lower().split())
    return [t for t, p in TAGS if re.search(p, s)]


def literals(sql: str) -> set[str]:
    return {m.group(1).replace("''", "'") for m in re.finditer(r"'((?:[^']|'')*)'", sql)}


def card_schema(card: SchemaCard, sql: str) -> list[dict]:
    s = sql.lower()
    return [{"table": t.name, "description": (t.description or "")[:200],
             "columns": [{"name": c.name, "type": c.type, "description": c.description, "examples": c.samples[:2]}
                         for c in t.columns]}
            for t in card.tables if t.name.lower() in s]


def runs_nonempty(db: str, sql: str, seconds: float = 5.0) -> bool:
    p = DBS / db / f"{db}.sqlite"
    if not p.exists():
        return False
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    t = time.time() + seconds
    con.set_progress_handler(lambda: 1 if time.time() > t else 0, 10_000)
    try:
        rows = con.execute(sql).fetchmany(1000)
        return any(v is not None for r in rows for v in r)
    except Exception:
        return False
    finally:
        con.close()


def collect():
    """Rows judged faithful and natural. Claude's verdict (pack_NNNN.result.jsonl) wins over Gemini's."""
    verdict = {}
    for tag in ("gemini", "codex", "result"):  # later tags override earlier ones
        for f in OUT.glob(f"pack_*.{tag}.jsonl"):
            for l in open(f):
                v = json.loads(l)
                if str(v.get("vid", "")).startswith("curr_synsql|"):
                    verdict[v["vid"].split("|", 1)[1]] = bool(v.get("faithful")) and bool(v.get("natural"))
    picked = [json.loads(l) for l in open("data/curr/picked.jsonl")]
    ok = [r for r in picked if verdict.get(r["id"])]
    Path("data/curr/verified.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ok))
    tags = collections.Counter(t for r in ok for t in r["tags"])
    print(f"CURR verified {len(ok)} of {len(picked)} picked ({sum(r['id'] in verdict for r in picked)} judged); tags {dict(tags)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=12000)
    ap.add_argument("--pack-size", type=int, default=500)
    ap.add_argument("--first-pack", type=int, default=0, help="0 = after the highest existing pack number")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect", action="store_true",
                    help="after verification: write data/curr/verified.jsonl (faithful and natural) and stop")
    a = ap.parse_args()
    if a.collect:
        return collect()
    rnd = random.Random(0)
    src = V3 / "synsql_train"
    rows = [json.loads(l) for l in open(src / "plans.jsonl")]
    rows = [r for r in rows if r.get("outcome") != "plan" and not r.get("twin_of") and r.get("gold_sql")]
    print(f"outside the slot grammar: {len(rows)}", flush=True)
    want = {r["question"].strip() for r in rows}
    style = {}
    with open("data/raw/synsql/data.json", "rb") as f:
        for row in ijson.items(f, "item"):
            q = (row.get("question") or "").strip()
            if q in want:
                style[q] = row.get("question_style")
    eval_q = [json.loads(l)["question"] for d in EVAL.iterdir() for l in open(d / "plans.jsonl")]
    n0 = len(rows)
    rows = dedup_against(rows, eval_q)
    st = collections.Counter({"sealed_overlap": n0 - len(rows)})
    cards, ok = {}, []
    for r in rows:
        text = r["question"] + " " + (r.get("context") or "")
        cp = src / "cards" / f"{r['db_id']}.json"
        if not cp.exists():
            st["no_card"] += 1
            continue
        card = cards.setdefault(r["db_id"], SchemaCard.load(cp))
        s = " ".join(r["gold_sql"].lower().split())
        why = ("style" if style.get(r["question"].strip()) in BAD or style.get(r["question"].strip()) is None
               else "dialog" if DIALOG.search(text)
               else "identifier" if identifier_leak(text, card, literals(r["gold_sql"]))
               else "limit" if re.search(r"\blimit\b", s) and not re.search(r"\border\s+by\b", s) else None)
        st[why or "pass_rules"] += 1
        if why is None:
            r["tags"] = tags_of(r["gold_sql"]) or ["other"]
            ok.append(r)
    print("rules:", dict(st), flush=True)
    # stratified draw: weight each row by its rarest, most wanted construct; spread over databases
    rnd.shuffle(ok)
    per_db = collections.Counter()
    score = lambda r: max(WEIGHT.get(t, 1) for t in r["tags"]) + rnd.random()
    picked, ran = [], collections.Counter()
    for r in sorted(ok, key=score, reverse=True):
        if len(picked) >= a.total:
            break
        if per_db[r["db_id"]] >= 3:
            continue
        good = runs_nonempty(r["db_id"], r["gold_sql"])
        ran["runs_nonempty" if good else "fails_or_empty"] += 1
        if good:
            per_db[r["db_id"]] += 1
            picked.append(r)
    tag_n = collections.Counter(t for r in picked for t in r["tags"])
    print(f"picked {len(picked)} on {len(per_db)} databases; execution {dict(ran)}; tags {dict(tag_n)}", flush=True)
    Path("data/curr").mkdir(parents=True, exist_ok=True)
    Path("data/curr/picked.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("id", "db_id", "question", "context", "gold_sql", "tags")}) + "\n" for r in picked))
    if a.dry_run:
        return
    first = a.first_pack or 1 + max(int(p.stem.split("_")[1]) for p in OUT.glob("pack_[0-9][0-9][0-9][0-9].json"))
    items = [{"vid": f"curr_synsql|{r['id']}", "question": r["question"] + (f"\nContext: {r['context']}" if r.get("context") else ""),
              "sql": r["gold_sql"], "plan_english": "", "schema": card_schema(cards[r["db_id"]], r["gold_sql"])}
             for r in picked]
    for k in range(0, len(items), a.pack_size):
        n = first + k // a.pack_size
        (OUT / f"pack_{n:04d}.json").write_text(json.dumps({"source": "curr_synsql", "items": items[k:k + a.pack_size]}))
    print(f"packs {first}..{first + (len(items) - 1) // a.pack_size}", flush=True)


if __name__ == "__main__":
    main()
