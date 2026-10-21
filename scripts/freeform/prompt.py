"""OmniSQL's official input prompt (model card of seeklhy/OmniSQL-7B; RUCKBReasoning/OmniSQL
train_and_evaluate/process_dataset.py), built from a SQLite file. Pure Python + sqlite3: no model, CPU only.

Schema: CREATE TABLE statements, one line per column, with a comment holding the column description (when the BIRD
database_description CSVs exist next to the .sqlite file; they are not on this Mac) and up to --examples distinct
non-null values ("example: [...]"), then PRIMARY KEY / FOREIGN KEY lines. Question: BIRD evidence + newline + question
(OmniSQL: `question = evidence + "\\n" + question` when the evidence is not empty).

The official pipeline also retrieves question-matched values (an LSH value index); this skeleton keeps a cheap
substitute: for TEXT columns, values that contain a quoted literal of the evidence/question (exact, case-insensitive,
LIMIT 2, under a time limit). Flagged as a deviation in docs/paper_notes/research/freeform_plan_2026-10-10.md.
"""
from __future__ import annotations

import csv
import re
import sqlite3
import time
from pathlib import Path

TEMPLATE = """Task Overview:
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
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""


def ident(name: str) -> str:
    """OmniSQL's format_identifier: plain when it is a simple identifier, else backquoted."""
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else f"`{name}`"


def _descriptions(db_path: str) -> dict[tuple[str, str], str]:
    d = Path(db_path).parent / "database_description"
    out = {}
    if not d.is_dir():
        return out
    for f in d.glob("*.csv"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                with open(f, encoding=enc, newline="") as fh:
                    for row in csv.DictReader(fh):
                        col = (row.get("original_column_name") or "").strip()
                        desc = " ".join(x.strip() for x in (row.get("column_description"), row.get("value_description"))
                                        if x and x.strip())
                        if col and desc:
                            out[(f.stem.lower(), col.lower())] = re.sub(r"\s+", " ", desc)[:200]
                break
            except UnicodeDecodeError:
                continue
    return out


def _literals(text: str) -> list[str]:
    pairs = re.findall(r"'([^']{2,60})'|\"([^\"]{2,60})\"", text or "")
    return [x for pair in pairs for x in pair if x][:6]


def db_details(db_path: str, question_text: str = "", examples: int = 2, descriptions: bool = True,
               seconds: float = 2.0, max_value_chars: int = 40) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)

    def q(sql, args=()):
        box["t"] = time.time() + seconds
        try:
            return con.execute(sql, args).fetchall()
        except sqlite3.Error:
            return []
    desc = _descriptions(db_path) if descriptions else {}
    lits = _literals(question_text)
    parts = []
    try:
        tables = [r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                                  "ORDER BY rowid")]
        for t in tables:
            cols = q(f'PRAGMA table_info("{t}")')  # cid, name, type, notnull, dflt, pk
            fks = q(f'PRAGMA foreign_key_list("{t}")')  # id, seq, table, from, to, ...
            body = []
            for _, name, typ, _, _, _ in cols:
                vals = []
                if lits and (not typ or any(k in typ.upper() for k in ("CHAR", "TEXT", "CLOB"))):
                    for lit in lits:
                        vals += [r[0] for r in q(f'SELECT DISTINCT "{name}" FROM "{t}" WHERE "{name}" = ? COLLATE NOCASE '
                                                 f'LIMIT 2', (lit,))]
                if examples:
                    vals += [r[0] for r in q(f'SELECT DISTINCT "{name}" FROM "{t}" WHERE "{name}" IS NOT NULL LIMIT ?',
                                             (examples,))]
                seen, shown = set(), []
                for v in vals:
                    s = v if not isinstance(v, str) else v[:max_value_chars]
                    if repr(s) not in seen:
                        seen.add(repr(s))
                        shown.append(s)
                comment = desc.get((t.lower(), name.lower()), "")
                line = f"    {ident(name)} {(typ or 'text').lower()},"
                if comment or shown:
                    line += " --" + (f" {comment}" if comment else "") + (
                        f"{',' if comment else ''} example: {shown[:examples + 2]}" if shown else "")
                body.append(line)
            pk = [c[1] for c in sorted(cols, key=lambda c: c[5]) if c[5]]
            if pk:
                body.append(f"    PRIMARY KEY ({', '.join(ident(p) for p in pk)}),")
            for fk in fks:
                if fk[4] is not None:
                    body.append(f"    CONSTRAINT fk_{t.lower()}_{fk[3].lower()} FOREIGN KEY ({ident(fk[3])}) REFERENCES "
                                f"{ident(fk[2])} ({ident(fk[4])}),")
            if body:
                last = body[-1]
                body[-1] = (last.split(" --", 1)[0].rstrip(",") + " --" + last.split(" --", 1)[1]) if " --" in last \
                    else last.rstrip(",")
            parts.append(f"CREATE TABLE {ident(t)} (\n" + "\n".join(body) + "\n);")
    finally:
        con.close()
    return "\n\n".join(parts)


def question_text(question: str, evidence: str) -> str:
    return question if not (evidence or "").strip() else f"{evidence.strip()}\n{question}"


def build(db_path: str, question: str, evidence: str = "", **kw) -> str:
    qt = question_text(question, evidence)
    return TEMPLATE.format(db_details=db_details(db_path, qt, **kw), question=qt)


_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str | None:
    """The last ```sql block (OmniSQL's own parser takes the last block); else None. Comment-only lines and a trailing
    semicolon are removed; a second statement makes the candidate None (sqlite3 runs one statement)."""
    blocks = _BLOCK.findall(text or "")
    if not blocks:
        return None
    lines = [ln for ln in blocks[-1].strip().splitlines() if not ln.strip().startswith("--")]
    sql = "\n".join(lines).strip().rstrip(";").strip()
    if not sql or ";" in re.sub(r"'(?:[^']|'')*'", "", sql):
        return None
    return sql
