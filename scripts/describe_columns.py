"""Write a short description for every column that has none, from its name, type and example values.

Runs once when a database is connected (never per question), so it costs nothing at answer time.
The describer sees only the schema and a few values from the database itself; never the questions.
Cryptic names are the point: `ST TEXT e.g. AAPL, CRWD, MSFT` -> "stock ticker symbol".

  python scripts/describe_columns.py --cards data/x/cards --out data/x_desc/cards            # Mac (MLX)
  python scripts/describe_columns.py --strip --cards data/sys1/bird_dev/cards --out data/sys1/bird_dev_nodesc/cards

--strip only removes every description (the test condition: a database that ships none).
--all re-describes columns that already have a description.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luthor.sq.schema_card import SchemaCard  # noqa: E402

PROMPT = """You are documenting a database so an analyst can understand each column.
Database: {db}
Other tables: {others}

Table {table}:
{cols}

For each column of table {table}, write a short plain-English description (at most 12 words) of what it holds.
Expand abbreviations and codes when the name or the example values make the meaning clear (for example
ST with values AAPL, MSFT -> "stock ticker symbol"). Say what a key refers to. Do not invent units or facts
the name and values do not support. If the name and values do not make the meaning clear (a code like A7 with
plain numbers), write exactly "unclear": a wrong description misleads more than none.
Answer with JSON only: {{"table": "one-line description of the table", "columns": {{"<column>": "<description>", ...}}}}"""


def examples(card: SchemaCard, t, c, n: int = 5) -> list[str]:
    vals = card.values.get(f"{t.name}.{c.name}") or []
    out = list(dict.fromkeys([str(v)[:30] for v in (c.samples or [])] + [str(v)[:30] for v in vals]))
    return out[:n]


def table_prompt(card: SchemaCard, t) -> str:
    fk = {(a, ca): f"{b}.{cb}" for a, ca, b, cb in card.fks}
    lines = []
    for c in t.columns:
        bits = [c.type.lower() or "any"] + (["primary key"] if c.pk else [])
        if (t.name, c.name) in fk:
            bits.append(f"references {fk[(t.name, c.name)]}")
        ex = examples(card, t, c)
        lines.append(f"  {c.name} ({', '.join(bits)})" + (f" examples: {', '.join(ex)}" if ex else ""))
    others = ", ".join(x.name for x in card.tables if x.name != t.name) or "none"
    return PROMPT.format(db=card.db_id, others=others, table=t.name, cols="\n".join(lines))


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return d if isinstance(d, dict) else {}


class MLX:
    def __init__(self, model: str):
        from mlx_lm import generate, load
        self.model, self.tok = load(model)
        self._gen = generate

    def __call__(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self._gen(self.model, self.tok, prompt=p, max_tokens=2500, verbose=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strip", action="store_true", help="only remove all descriptions")
    ap.add_argument("--all", action="store_true", help="describe columns that already have a description too")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    llm = None if a.strip else MLX(a.model)
    t0, n_cols, n_desc = time.time(), 0, 0
    for f in sorted(a.cards.glob("*.json")):
        card = SchemaCard.load(f)
        for t in card.tables:
            if a.strip:
                t.description = ""
                for c in t.columns:
                    c.description = ""
                continue
            todo = [c for c in t.columns if a.all or not c.description]
            if not todo and t.description:
                continue
            d = parse(llm(table_prompt(card, t)))
            cols = {k.lower(): v for k, v in (d.get("columns") or {}).items() if isinstance(v, str)}
            if not t.description and isinstance(d.get("table"), str):
                t.description = d["table"].strip()[:120]
            for c in todo:
                n_cols += 1
                v = cols.get(c.name.lower())
                if v and v.strip().strip(".").lower() != "unclear":
                    c.description = v.strip()[:120]
                    n_desc += 1
        card.save(a.out / f.name)
        print(f"{card.db_id}: done ({n_desc}/{n_cols} columns described, {time.time() - t0:.0f}s)", flush=True)
    print(f"FINAL {n_desc}/{n_cols} columns described in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
