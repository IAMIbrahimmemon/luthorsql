"""Work packs for Claude-subagent verification (when the DeepSeek quota is spent).

Each pack is 500 items {vid, question, sql, plan_english, schema (only tables the SQL uses)} in
data/claude_verify/pack_NNNN.json; a subagent writes pack_NNNN.result.jsonl with
{vid, faithful, natural, problem} - the same judgement as the round-3 Claude audit.
vid = "<source dir>|<item id>" so build_mix3.py can map verdicts back.

Priority order (BIRD first: it is where quality and data are short):
  1. data/qgen/bird/ds          DeepSeek-written, checker-accepted, not yet verified
  2. data/qgen3/bird/ds         new DeepSeek-written questions (no checker yet; Claude's verdict replaces it)
  3. data/qgen_synsql           SynSQL rows that passed the style filter + rules
Claude-written questions are not sent to Claude (writer and judge must differ).
Every item must already pass the audit rules (fan-out, identifiers, LIMIT without ORDER BY).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality import fix_fan_out, identifier_leak, limit_without_order  # noqa: E402

from luthor.sq.schema_card import SchemaCard  # noqa: E402

OUT = Path("data/claude_verify")
V2 = Path("/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2")


def slim_schema(schema: list[dict], sql: str) -> list[dict]:
    s = sql.lower()
    out = []
    for t in schema or []:
        if t.get("table", "").lower() in s:
            out.append({"table": t["table"], "description": (t.get("description") or "")[:200],
                        "columns": [{k: c.get(k) for k in ("name", "type", "description", "examples")} for c in t["columns"]]})
    return out


def done_vids(d: Path) -> set[str]:
    out = set()
    for f in d.glob("*.verified.jsonl"):
        for l in open(f):
            try:
                out.add(json.loads(l)["vid"])
            except Exception:
                pass
    return out


def from_verify_files(d: Path, tag: str) -> list[dict]:
    """Items prepared for the DeepSeek verifier that it never reached."""
    seen = done_vids(d)
    items = []
    for f in sorted(d.glob("verify_*.json")):
        for x in json.loads(f.read_text())["items"]:
            if x["vid"] not in seen:
                items.append({"vid": f"{tag}|{x['vid']}", "question": x["question"], "sql": x["sql"],
                              "plan_english": x.get("plan_english", ""), "schema": slim_schema(x.get("schema"), x["sql"])})
    return items


def from_questions(d: Path, tag: str, cards_dir: Path) -> list[dict]:
    """Written but unchecked questions: apply the audit rules, then send to Claude."""
    plans = {json.loads(l)["plan_id"]: json.loads(l) for l in open(d / "plans_with_ids.jsonl")}
    cards, items = {}, []
    for qf in sorted(d.glob("questions_*.json")):
        wf = qf.with_name(qf.name.replace("questions_", "writer_"))
        try:
            w = {p["plan_id"]: p for p in json.loads(wf.read_text())["plans"]}
            qs = json.loads(qf.read_text())["items"]
        except Exception:
            continue
        for it in qs:
            pid = it.get("plan_id")
            if it.get("skip") or pid not in plans:
                continue
            p = plans[pid]
            card = cards.setdefault(p["db_id"], SchemaCard.load(cards_dir / f"{p['db_id']}.json"))
            lits = {str(v) for f in p["plan"].get("filters") or [] for v in f.get("values") or []}
            fixed, _ = fix_fan_out(p["plan"], card)
            if fixed is None or limit_without_order(p["plan"]):
                continue
            for k, q in enumerate(it.get("questions") or []):
                if identifier_leak(q["text"], card, lits):
                    continue
                items.append({"vid": f"{tag}|{pid}_{k}", "question": q["text"], "sql": p["sql"],
                              "plan_english": w[pid]["plan"], "schema": slim_schema(w[pid]["schema"], p["sql"])})
    return items


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    have = set()
    for f in OUT.glob("pack_*.json"):
        have |= {x["vid"] for x in json.loads(f.read_text())["items"]}
    groups = [("qgen/bird/ds", from_verify_files(Path("data/qgen/bird/ds"), "qgen/bird/ds")),
              ("qgen3/bird/ds", from_questions(Path("data/qgen3/bird/ds"), "qgen3/bird/ds", V2 / "bird_train" / "cards")),
              ("qgen_synsql", from_verify_files(Path("data/qgen_synsql"), "qgen_synsql"))]
    n = len(list(OUT.glob("pack_*.json")))
    for name, items in groups:
        items = [x for x in items if x["vid"] not in have]
        for i in range(0, len(items), 500):
            (OUT / f"pack_{n:04d}.json").write_text(json.dumps({"source": name, "items": items[i:i + 500]}, default=str))
            n += 1
        print(f"{name}: {len(items)} items")
    print(f"{n} packs in {OUT}")


if __name__ == "__main__":
    main()
