"""BIRD-train column descriptions (database_description/<table>.csv) merged into schema cards. Opt-in.

BIRD ships, per database and table, a CSV with original_column_name, column_name (expanded name),
column_description, data_format and value_description. The cards built so far carry none of it (hockey: 0/300
columns described), so look-alike columns (Scoring vs ScoringSC, nameGiven vs firstName) can only be told apart by
their sample values. This module turns each row into one short description and writes it into
ColumnCard.description, which every renderer already shows (sq2 serialize.schema_text, sqlw table_ddl / ddl6).

Off by default so existing runs reproduce. Enable with the environment variable

  LUTHOR_BIRD_DESC=1            (descriptions from data/raw/bird_train_desc)
  LUTHOR_BIRD_DESC=<dir>        (another directory with <db>/<table>.csv)

(or --bird-desc on train_sq2.py / rerank_sq2.py / eval_sq2.py, which set it). It is read by SchemaCard.load, so every
loader -- training, reranking, eval, headdump, freeform prompts -- sees the same cards. Only columns with no
description yet are filled; only databases with a description directory are touched (BIRD-train: 69 databases).

The CSVs are downloaded data (scripts/fetch_bird_train_desc.py): they are parsed as CSV text, never executed.
"""
from __future__ import annotations

import csv
import difflib
import io
import os
import re
from pathlib import Path

ENV = "LUTHOR_BIRD_DESC"
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data/raw/bird_train_desc"
DESC_CAP = 120  # characters per column (= sq2 serialize.DESC_MAX)
_BOILER = re.compile(r"(?i)^\s*(commonsense evidence|common sense evidence|commonsense reasoning)\s*:?\s*")
_cache: dict[str, dict] = {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _one_line(s: str) -> str:
    s = _BOILER.sub("", (s or "").replace("﻿", ""))
    s = re.sub(r"\s*\n\s*", "; ", s.strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(;\s*)+", "; ", s).strip(" ;")
    return _BOILER.sub("", s)


def _decode(b: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def parse_csv(text: str) -> dict[str, tuple[str, str, str]]:
    """{normalised original column name: (column_name, column_description, value_description)}."""
    out = {}
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return out
    head = [_norm(h) for h in rows[0]]

    def col(*names):
        for n in names:
            if n in head:
                return head.index(n)
        return None
    i_orig = col("originalcolumnname")
    i_name, i_desc, i_val = col("columnname"), col("columndescription"), col("valuedescription")
    if i_orig is None:
        return out
    for r in rows[1:]:
        if len(r) <= i_orig or not r[i_orig].strip():
            continue

        def get(i):
            return r[i].strip() if i is not None and i < len(r) else ""
        out[_norm(r[i_orig])] = (get(i_name), get(i_desc), get(i_val))
    return out


def load_dir(root: str | Path) -> dict[str, dict[str, dict[str, tuple[str, str, str]]]]:
    """{db: {normalised table: {normalised column: (column_name, description, value description)}}}."""
    root = Path(root)
    out = {}
    if not root.is_dir():
        return out
    for db in sorted(p for p in root.iterdir() if p.is_dir()):
        tabs = {}
        for f in sorted(db.glob("*.csv")):
            tabs[_norm(f.stem)] = parse_csv(_decode(f.read_bytes()))
        if tabs:
            out[db.name] = tabs
    return out


def describe(col_name: str, entry: tuple[str, str, str], cap: int = DESC_CAP) -> str:
    """One line: the description (or the expanded column name when the description is empty / just the name),
    then the value description when it fits the cap (it is cut with "..." otherwise, once at least 30 characters
    of it fit). "" when nothing says more than the column name itself."""
    name, desc, val = (_one_line(x) for x in entry)
    base = desc if desc and _norm(desc) not in (_norm(col_name), _norm(name)) else ""
    if not base and name and _norm(name) != _norm(col_name):
        base = name
    elif base and name and _norm(name) != _norm(col_name) and _norm(name) not in _norm(base):
        base = f"{name}: {base}"
    if val and _norm(val) not in (_norm(base), _norm(col_name)):
        room = cap - len(base) - (2 if base else 0)
        if len(val) <= room:
            base = f"{base}; {val}" if base else val
        elif room >= 30:
            v = val[:room - 3].rsplit(" ", 1)[0]
            base = f"{base}; {v}..." if base else f"{v}..."
    if len(base) > cap:
        base = base[:cap - 3].rsplit(" ", 1)[0] + "..."
    return base


def _closest(t, tabs: dict) -> dict | None:
    """A CSV whose file name differs from the table name (BIRD: playstore -> googleplaystore.csv,
    filed_for_bankrupcy -> filed_for_bankruptcy.csv): the most similar name among those that hold at least half of
    the table's columns."""
    names = {_norm(c.name) for c in t.columns}
    best, score = None, 0.0
    for k, cols in tabs.items():
        if len(names & set(cols)) * 2 < len(names):
            continue
        r = difflib.SequenceMatcher(None, _norm(t.name), k).ratio()
        if r > score:
            best, score = cols, r
    return best if score >= 0.5 else None


_STOP = {"the", "a", "an", "of", "unique", "id", "ids", "identifier", "identification", "number", "no", "num", "code",
         "key", "primary", "foreign", "name", "names", "this", "is", "for", "in", "column", "value", "values", "which",
         "refers", "to", "and", "or", "it", "its", "means", "mean", "stands"}


def _words(s: str) -> set:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s or "")
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def informative(col_name: str, entry: tuple[str, str, str]) -> bool:
    """Does the CSV row say anything beyond the column name and its expanded name? ("unique order number" for
    orderNumber: no.) A word counts when it is not in either name and not a filler word (_STOP)."""
    name, desc, val = (_one_line(x) for x in entry)
    return bool((_words(desc) | _words(val)) - _words(col_name) - _words(name) - _STOP)


def cryptic(col_name: str) -> bool:
    """A column name a reader cannot decode on its own: at most 4 characters, or no word of 4+ letters with a vowel
    (GP, lis_id, ft_pct, PIM)."""
    ws = [w for w in _words(col_name) if not w.isdigit()]
    return len(col_name) <= 4 or not any(len(w) >= 4 and re.search(r"[aeiou]", w) for w in ws)


MODES = ("full", "info", "cryptic")  # every column / informative rows only / informative rows of cryptic columns


def apply(card, descs: dict | None = None, cap: int = DESC_CAP, mode: str = "full") -> int:
    """Fill empty column descriptions of `card` from `descs` (load_dir output). Returns the number filled.
    mode "full" (default, as v4desc): every row that says more than the column name; "info": only rows that add a
    word beyond the column and expanded names (informative); "cryptic": "info" and only for cryptic column names."""
    if mode not in MODES:
        raise ValueError(f"bird_desc mode {mode!r}: expected one of {MODES}")
    if descs is None:
        descs = active()
    tabs = (descs or {}).get(card.db_id)
    if not tabs:
        return 0
    n = 0
    for t in card.tables:
        cols = tabs.get(_norm(t.name)) or _closest(t, tabs)
        if not cols:
            continue
        for c in t.columns:
            e = cols.get(_norm(c.name))
            if e is None or c.description:
                continue
            if mode != "full" and (not informative(c.name, e) or (mode == "cryptic" and not cryptic(c.name))):
                continue
            d = describe(c.name, e, cap)
            if d:
                c.description = d
                n += 1
    return n


def active() -> dict | None:
    """The descriptions selected by LUTHOR_BIRD_DESC, or None when the flag is off (the default)."""
    v = os.environ.get(ENV, "").strip()
    if not v or v == "0":
        return None
    root = str(DEFAULT_DIR) if v == "1" else v
    if root not in _cache:
        _cache[root] = load_dir(root)
        if not _cache[root]:
            raise FileNotFoundError(f"{ENV}={v}: no <db>/<table>.csv under {root}")
    return _cache[root]


def enable(flag: bool | str) -> None:
    """For CLI flags: --bird-desc sets LUTHOR_BIRD_DESC (inherited by spawned workers)."""
    if flag:
        os.environ[ENV] = "1" if flag is True else str(flag)
