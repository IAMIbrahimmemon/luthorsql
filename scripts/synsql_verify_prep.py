"""SynSQL rows that may enter the round-3 mix: originals (no twins) with a non-empty result, a question
style the audit found reliable (not Vague / Metaphorical / Multi-turn Dialogue), no dialogue or schema
identifiers, no LIMIT without ORDER BY, no unrepairable join fan-out. A random --cap of them go to the
verifier (data/qgen_synsql/verify_NNNN.json, vid = row id); only verified rows enter the mix.
"""
import argparse, json, random, re, sys
from pathlib import Path
import ijson
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality import fix_fan_out, identifier_leak, limit_without_order
from luthor.sq.schema_card import SchemaCard

V3 = Path("/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v3/synsql_train")
BAD = {"Vague", "Metaphorical", "Multi-turn Dialogue"}
DIALOG = re.compile(r"\b(User|Assistant)\s*:|SQL|query", re.I)
ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=80000); ap.add_argument("--out", default="data/qgen_synsql")
a = ap.parse_args()
rows = [json.loads(l) for l in open(V3 / "plans.jsonl")]
rows = [r for r in rows if r.get("outcome") == "plan" and not r.get("twin_of") and not r.get("empty_result")]
want = {r["question"].strip() for r in rows}
style = {}
with open("data/raw/synsql/data.json", "rb") as f:
    for row in ijson.items(f, "item"):
        q = (row.get("question") or "").strip()
        if q in want:
            style[q] = row.get("question_style")
cards, keep, st = {}, [], {}
for r in rows:
    s = style.get(r["question"].strip())
    text = r["question"] + " " + (r.get("context") or "")
    card = cards.setdefault(r["db_id"], SchemaCard.load(V3 / "cards" / f"{r['db_id']}.json"))
    lits = {str(v) for fl in r["plan"].get("filters") or [] for v in fl.get("values") or []}
    why = ("style" if s in BAD or s is None else "dialog" if DIALOG.search(text) else
           "identifier" if identifier_leak(text, card, lits) else "limit" if limit_without_order(r["plan"]) else None)
    if why is None:
        p, fs = fix_fan_out(r["plan"], card)
        if p is None:
            why = "fan_out"
    st[why or "ok"] = st.get(why or "ok", 0) + 1
    if why is None:
        keep.append(r)
print(len(rows), "originals;", st)
random.Random(0).shuffle(keep)
keep = keep[:a.cap]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for i in range(0, len(keep), 500):
    items = []
    for r in keep[i:i + 500]:
        card = cards[r["db_id"]]
        used = {t.name for t in card.tables if t.name.lower() in r["compiled_sql"].lower()}
        sch = [{"table": t.name, "description": t.description,
                "columns": [{"name": c.name, "type": c.type, "description": c.description, "examples": c.samples[:3]} for c in t.columns]}
               for t in card.tables if t.name in used]
        items.append({"vid": r["id"], "question": r["question"] + (f"\nContext: {r['context']}" if r.get("context") else ""),
                      "sql": r["compiled_sql"], "plan_english": "", "schema": sch})
    (out / f"verify_{i // 500:04d}.json").write_text(json.dumps({"items": items}, default=str))
print(len(keep), "rows ->", out)
