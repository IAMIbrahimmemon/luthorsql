"""Round-6 prompts: the full schema (no table cut), inferred join keys shown, and three prompt styles whose
samples are pooled before the vote (diverse generators):

  think   - OmniSQL's own template (reason step by step, then the SQL in a code block)
  sql     - our round-5 format (SQL only), full schema
  fewshot - SQL only, with three solved questions from other (training) databases
  arctic  - Arctic-Text2SQL-R1's own inference template (system + OmniSQL-style user turn, <think>/<answer>,
            assistant turn pre-filled with "Let me solve this step by step. <think>"), from the official
            ArcticTraining bird_eval/infer.py

Step 0 found the round-5 9,000-character budget dropped a gold table for 15% of BIRD dev; every BIRD dev
schema fits in 24,000 characters (at most about 7.6k tokens).
"""
from __future__ import annotations

import re

from ..sq.schema_card import SchemaCard
from .prompt import SYSTEM, _q, _words, clean_sql, table_ddl

BUDGET6 = 24000  # characters; beyond it columns are pruned, never tables

OMNISQL = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""


ARCTIC_SYSTEM = ("You are a data science expert. Below, you are provided with a database schema and a natural language "
                 "question. Your task is to understand the schema and generate a valid SQL query to answer the question.")
ARCTIC_FORMAT = """Please provide a detailed chain-of-thought reasoning process and include your thought process within `<think>` tags. Your final answer should be enclosed within `<answer>` tags.

Ensure that your SQL query follows the correct syntax and is formatted as follows:

```sql
-- Your SQL query here
```

Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]

```sql
Correct SQL query here
```
</answer>"""
ARCTIC_USER = """Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
{fmt}"""
ARCTIC_PREFILL = "Let me solve this step by step. \n<think>"


def ddl6(card: SchemaCard, question: str = "", context: str = "", budget: int = BUDGET6) -> str:
    """Every table. Declared foreign keys plus the card's inferred join keys (marked). Over the budget, the
    columns least related to the question are dropped (keys and join columns always stay)."""
    declared = {(t1, c1, t2, c2) for t1, c1, t2, c2 in card.fks}
    inferred = [fk for fk in (card.inferred_fks or []) if tuple(fk) not in declared]
    parts = []
    for t in card.tables:
        d = table_ddl(card, t)
        extra = [f"  -- join key (inferred): {_q(c1)} = {_q(t2)}.{_q(c2)}" for t1, c1, t2, c2 in inferred if t1 == t.name]
        if extra:
            d = d[:-2] + "\n" + "\n".join(extra) + "\n);"
        parts.append(d)
    out = "\n\n".join(parts)
    if len(out) <= budget:
        return out
    # rare: prune the columns least related to the question, table by table
    qw = _words(question + " " + context)
    keys = {(a, b) for a, b, _, _ in card.fks} | {(c, d) for _, _, c, d in card.fks}
    keys |= {(a, b) for a, b, _, _ in inferred} | {(c, d) for _, _, c, d in inferred}
    lines = out.split("\n")
    scored = []
    cur = None
    for i, ln in enumerate(lines):
        m = re.match(r'CREATE TABLE "?([^"\s(]+)', ln)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r'  "?([^"\s]+)"? \w', ln)
        if m and cur and "PRIMARY KEY" not in ln and (cur, m.group(1)) not in keys:
            scored.append((len(qw & _words(ln)), i))
    drop = set()
    for _, i in sorted(scored):
        if sum(len(l) + 1 for j, l in enumerate(lines) if j not in drop) <= budget:
            break
        drop.add(i)
    return "\n".join(l for j, l in enumerate(lines) if j not in drop)


def _hints_block(hints):
    return ("Matching database values:\n" + "\n".join(hints) + "\n\n") if hints else ""


def think_messages(card, question, context="", hints=None):
    q = (context.strip() + "\n" + question.strip()) if context.strip() else question.strip()  # OmniSQL: evidence + question
    details = ddl6(card, question, context) + ("\n\n" + _hints_block(hints).strip() if hints else "")
    return [{"role": "user", "content": OMNISQL.format(db_details=details, question=q)}]


def arctic_messages(card, question, context="", hints=None):
    """Ends with a partial assistant turn: generate with continue_final_message=True."""
    q = (context.strip() + "\n" + question.strip()) if context.strip() else question.strip()
    details = ddl6(card, question, context) + ("\n\n" + _hints_block(hints).strip() if hints else "")
    return [{"role": "system", "content": ARCTIC_SYSTEM},
            {"role": "user", "content": ARCTIC_USER.format(db_details=details, question=q, fmt=ARCTIC_FORMAT)},
            {"role": "assistant", "content": ARCTIC_PREFILL}]


def sql_messages(card, question, context="", hints=None):
    user = f"Database schema:\n{ddl6(card, question, context)}\n\n{_hints_block(hints)}"
    if context.strip():
        user += f"Evidence: {context.strip()}\n"
    user += f"Question: {question.strip()}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def fewshot_messages(card, question, context="", hints=None, shots=()):
    ex = "\n\n".join(f"Question: {s['question']}\n" + (f"Evidence: {s['context']}\n" if s.get("context") else "")
                     + f"SQL: {' '.join(s['gold_sql'].split())}" for s in shots)
    msgs = sql_messages(card, question, context, hints)
    if ex:
        msgs[1]["content"] = ("Solved examples from other databases (for style only; their tables differ):\n\n"
                              + ex + "\n\n---\n\n" + msgs[1]["content"])
    return msgs


class ShotIndex:
    """Nearest training questions by word overlap, for few-shot examples (never from the same database)."""

    def __init__(self, rows):
        self.rows = rows
        self.words = [_words(r["question"] + " " + r.get("context", "")) for r in rows]
        self.inv = {}
        for i, w in enumerate(self.words):
            for t in w:
                self.inv.setdefault(t, []).append(i)

    def get(self, question, context, db_id, k=3):
        qw = _words(question + " " + context)
        score = {}
        for t in qw:
            ids = self.inv.get(t, [])
            if len(ids) > 2000:
                continue
            for i in ids:
                score[i] = score.get(i, 0) + 1
        best = sorted(score, key=lambda i: -score[i] / (len(self.words[i] | qw) or 1))
        out, dbs = [], set()
        for i in best:
            r = self.rows[i]
            if r["db_id"] == db_id or r["db_id"] in dbs:
                continue
            out.append(r)
            dbs.add(r["db_id"])
            if len(out) == k:
                break
        return out


def extract_sql(text: str) -> str:
    """The last fenced code block (think style ends with it), else the round-5 cleaner."""
    text = re.sub(r"<\|(?:im_end|endoftext|im_start)\|>.*", "", text, flags=re.S)
    blocks = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.S | re.I)
    if blocks:
        s = re.sub(r"^\s*--[^\n]*\n", "", blocks[-1].strip()).strip()
        return s.split(";")[0].strip() if ";" in s else s
    return clean_sql(text)


def unresolved_identifiers(card: SchemaCard, sql: str) -> list[str]:
    """Double-quoted names that are neither a table, a column nor an alias defined in the query. SQLite turns
    such a name into a string literal instead of failing (bird_dev_73 returned the text 'FRPM5to17')."""
    names = {t.name.lower() for t in card.tables} | {c.name.lower() for t in card.tables for c in t.columns}
    code = re.sub(r"'(?:[^']|'')*'", "''", sql)  # string literals may contain double quotes
    aliases = {a.lower() for a in re.findall(r'\bAS\s+"([^"]+)"', code, re.I)}
    return [x for x in re.findall(r'"([^"]+)"', code) if x.lower() not in names and x.lower() not in aliases]
