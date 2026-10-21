"""A database's schema as the bi-encoder sees it: one text per option.

Options are what the column-slot heads choose from:
  * one "rows" option per table (COUNT(*) over that table),
  * one option per column (name, type, business description, example values).
Value options are the literal values a filter can pick (see candidates.py). All option
texts are encoded once per database with the backbone and cached; a question costs one
forward pass plus dot products against that cache.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .plan import ROWS, Graph

_LOG = logging.getLogger("luthorsql")

OPT_TEMPLATE = "Premise: -\nHypothesis: {h}"
Q_TEMPLATE = "Question about a database: {q}\nThe request is for:"


def natural(ident: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", ident)
    return re.sub(r"[_\.\-]+", " ", s).strip()


@dataclass
class ColumnCard:
    name: str
    type: str
    description: str = ""
    pk: bool = False
    samples: list[str] = field(default_factory=list)
    n_distinct: int | None = None


@dataclass
class TableCard:
    name: str
    description: str = ""
    columns: list[ColumnCard] = field(default_factory=list)


@dataclass
class SchemaCard:
    db_id: str
    dialect: str
    tables: list[TableCard]
    fks: list[tuple[str, str, str, str]]  # (t1, c1, t2, c2)
    values: dict[str, list[str]] = field(default_factory=dict)  # "t.c" -> distinct values (indexed)
    inferred_fks: list[tuple[str, str, str, str]] = field(default_factory=list)  # guessed from names

    def graph(self) -> Graph:
        inferred = [tuple(e) for e in self.inferred_fks if not self._pk_to_pk(e)]
        return Graph({t.name: [c.name for c in t.columns] for t in self.tables},
                     [tuple(e) for e in self.fks] + inferred, self.dialect, set(inferred))

    def _pk_to_pk(self, e) -> bool:
        """A guessed edge from one table's own single primary key to another's ("customers.id =
        tracks.id" because both are named id) is never a real foreign key. Filtered at load, so
        cards written before this fix are corrected too."""
        own = {}
        for t in self.tables:
            pks = [c.name for c in t.columns if c.pk]
            if len(pks) == 1:
                own[t.name] = pks[0]
        a, ca, b, cb = e
        return own.get(a) == ca and own.get(b) == cb

    def options(self) -> list[dict]:
        """[{id, table, column, text}] with rows options first per table, stable order."""
        out = []
        for t in self.tables:
            tdesc = f": {t.description}" if t.description else ""
            out.append({"id": f"{t.name}.{ROWS}", "table": t.name, "column": ROWS,
                        "text": f"the number of rows (records) in table {natural(t.name)}{tdesc}"})
            for c in t.columns:
                desc = f": {c.description}" if c.description else ""
                ex = f" Examples: {', '.join(c.samples[:3])}." if c.samples else ""
                out.append({"id": f"{t.name}.{c.name}", "table": t.name, "column": c.name,
                            "text": f"column {natural(c.name)} of table {natural(t.name)} ({c.type.lower()}){desc}.{ex}"})
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> "SchemaCard":
        d = json.loads(Path(path).read_text())
        card = cls(d["db_id"], d["dialect"],
                   [TableCard(t["name"], t.get("description", ""), [ColumnCard(**c) for c in t["columns"]])
                    for t in d["tables"]],
                   [tuple(e) for e in d["fks"]], d.get("values", {}),
                   [tuple(e) for e in d.get("inferred_fks", [])])
        from . import bird_desc  # opt-in (LUTHOR_BIRD_DESC): BIRD-train column descriptions; off = card unchanged
        descs = bird_desc.active()
        if descs:
            bird_desc.apply(card, descs)
        return card


def value_text(col_id: str, value: str) -> str:
    t, c = col_id.split(".", 1)
    return f"the value {value} of column {natural(c)} in table {natural(t)}"


def card_from_spider_tables(entry: dict, sqlite_path: Path, descriptions: dict[str, dict[str, str]] | None = None,
                            value_cap: int = 2000, samples: int = 3) -> SchemaCard:
    """Build a card from a Spider/BIRD tables.json entry plus the SQLite file (samples, values)."""
    tnames = entry["table_names_original"]
    cols = entry["column_names_original"]
    types = entry["column_types"]
    pks = set()
    for pk in entry["primary_keys"]:
        pks.update(pk if isinstance(pk, list) else [pk])
    tables = [TableCard(t) for t in tnames]
    for i, (ti, cname) in enumerate(cols):
        if ti < 0:
            continue
        desc = (descriptions or {}).get(tnames[ti].lower(), {}).get(cname.lower(), "")
        tables[ti].columns.append(ColumnCard(cname, types[i], desc, i in pks))
    fks = []
    for a, b in entry["foreign_keys"]:
        ta, ca = cols[a]
        tb, cb = cols[b]
        fks.append((tnames[ta], ca, tnames[tb], cb))
    card = SchemaCard(entry["db_id"], "sqlite", tables, fks, inferred_fks=infer_fks(tables, fks))
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    for t in tables:
        for c in t.columns:
            q = f'"{c.name}"'
            try:
                rows = con.execute(f'SELECT {q}, COUNT(*) n FROM "{t.name}" WHERE {q} IS NOT NULL '
                                   f'GROUP BY {q} ORDER BY n DESC LIMIT {value_cap + 1}').fetchall()
            except sqlite3.Error:
                _LOG.warning("value sampling skipped for %s.%s", t.name, c.name, exc_info=True)
                continue
            c.samples = [str(r[0])[:40] for r in rows[:samples]]
            c.n_distinct = len(rows) if len(rows) <= value_cap else None
            texty = any(k in c.type.lower() for k in ("text", "char", "string")) or c.type == ""
            if texty and c.n_distinct is not None:
                vals = [str(r[0]) for r in rows if isinstance(r[0], str) and 0 < len(r[0]) <= 80]
                if vals:
                    card.values[f"{t.name}.{c.name}"] = vals
    con.close()
    return card


def infer_fks(tables: list[TableCard], known: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    """Name-based join keys for databases that declare few foreign keys: a column whose name
    equals another table's single-column primary key (or '<table>_id' / '<table>id')."""
    have = {(a, b, c, d) for a, b, c, d in known} | {(c, d, a, b) for a, b, c, d in known}
    connected = {frozenset((a, c)) for a, _, c, _ in known}
    pk = {}
    for t in tables:
        keys = [c.name for c in t.columns if c.pk]
        if len(keys) == 1:
            pk[t.name] = keys[0]
    out = []
    for t in tables:
        for c in t.columns:
            for u, k in pk.items():
                if u == t.name or frozenset((t.name, u)) in connected:
                    continue
                names = {k.lower(), f"{u.lower()}_{k.lower()}", f"{u.lower()}id", f"{u.lower()}_id"}
                if c.pk and len([x for x in t.columns if x.pk]) == 1:
                    continue  # a table's own primary key is not a foreign key into another table
                if c.name.lower() in names:
                    e = (t.name, c.name, u, k)
                    if e not in have:
                        out.append(e)
                        have.add(e)
                        connected.add(frozenset((t.name, u)))
    return out
