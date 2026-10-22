"""Build the renamed-schema test of the routed half (experiments/renamed_routed/PREREG.md).

Routed RESERVED questions (t = 3.0) -> renamed database copies, rewritten gold / candidates / evidence, paraphrased
questions, agent chunks in the cascade's routed-arm format. All new text comes from a non-Claude model
(deepseek-v4.1-flash:cloud via the local Ollama API). Every rewritten SQL is checked by execution.

  .venv/bin/python experiments/renamed_routed/build.py --box <sandbox> --out experiments/renamed_routed/build
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import sqlite3
import sys
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

ROOT = Path(__file__).resolve().parents[2]
SRC_DB = ROOT / "data/raw/bird_train/train_databases"
MODEL = "deepseek-v4.1-flash:cloud"
T = 3.0
KEYWORDS = {"select", "from", "where", "group", "order", "by", "limit", "join", "on", "as", "and", "or", "not", "null",
            "table", "index", "key", "values", "case", "when", "then", "else", "end", "in", "is", "like", "between",
            "union", "all", "distinct", "having", "count", "sum", "avg", "min", "max", "date", "time", "year", "default",
            "check", "primary", "foreign", "references", "unique", "set", "update", "delete", "insert", "into", "cast"}


def llm(prompt: str, tries: int = 3) -> dict:
    body = json.dumps({"model": MODEL, "stream": False, "format": "json", "options": {"temperature": 0.2},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(json.loads(r.read())["message"]["content"])
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"llm failed: {last}")


def words(name: str) -> set[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", s) if len(w) >= 3}


def execute(path: Path, sql: str):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    t = threading.Timer(30, con.interrupt)
    t.start()
    try:
        return frozenset(con.execute(sql).fetchall())
    except Exception:  # noqa: BLE001
        return None
    finally:
        t.cancel()
        con.close()


def schema_of(path: Path) -> dict[str, list[str]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        out[name] = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
    con.close()
    return out


def mapping_for(db: str, sch: dict[str, list[str]], stats: dict) -> dict:
    desc = "\n".join(f"{t}({', '.join(cols)})" for t, cols in sch.items())
    got = llm(
        "Rename these SQLite identifiers. Every new name must be a meaning-preserving snake_case synonym (lowercase "
        "letters, digits, underscores) that shares NO word of 3 or more letters with the original name, and must not "
        "be an SQL keyword. Table names must be unique; column names must be unique within their table. "
        'Return JSON {"tables": {"<old table>": "<new>"}, "columns": {"<old table>": {"<old column>": "<new>"}}} '
        f"covering every table and every column.\n\n{desc}")
    tmap, cmap, used_t = {}, {}, set()
    for k, (t, cols) in enumerate(sch.items()):
        new = str((got.get("tables") or {}).get(t, ""))
        if not re.fullmatch(r"[a-z][a-z0-9_]*", new) or new in KEYWORDS or new in used_t or words(new) & words(t):
            new = f"t_{k}"
            stats["table_fallbacks"] += 1
        used_t.add(new)
        tmap[t] = new
        cm, used_c = {}, set()
        gc = ((got.get("columns") or {}).get(t) or {})
        for j, c in enumerate(cols):
            nc = str(gc.get(c, ""))
            if not re.fullmatch(r"[a-z][a-z0-9_]*", nc) or nc in KEYWORDS or nc in used_c or words(nc) & words(c):
                nc = f"c_{j}"
                stats["column_fallbacks"] += 1
            used_c.add(nc)
            cm[c] = nc
        cmap[t] = cm
        stats["columns"] += len(cols)
    stats["tables"] += len(sch)
    return {"tables": tmap, "columns": cmap}


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_db(src: Path, dst: Path, m: dict):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    con = sqlite3.connect(dst)
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("PRAGMA legacy_alter_table=OFF")
    for t, cm in m["columns"].items():  # two steps so swapped names never collide
        for k, c in enumerate(cm):
            con.execute(f"ALTER TABLE {q(t)} RENAME COLUMN {q(c)} TO {q(f'__c{k}')}")
        for k, (c, nc) in enumerate(cm.items()):
            con.execute(f"ALTER TABLE {q(t)} RENAME COLUMN {q(f'__c{k}')} TO {q(nc)}")
    for k, t in enumerate(m["tables"]):
        con.execute(f"ALTER TABLE {q(t)} RENAME TO {q(f'__t{k}')}")
    for k, (t, nt) in enumerate(m["tables"].items()):
        con.execute(f"ALTER TABLE {q(f'__t{k}')} RENAME TO {q(nt)}")
    con.commit()
    con.close()


def rewrite_sql(sql: str, sch: dict[str, list[str]], m: dict) -> str:
    """Resolve every column to its base table (sqlglot qualify), then rename tables and columns."""
    low_t = {t.lower(): t for t in sch}
    low_c = {t.lower(): {c.lower(): c for c in cols} for t, cols in sch.items()}
    tree = sqlglot.parse_one(sql, read="sqlite")
    tree = qualify(tree, schema={t: {c: "TEXT" for c in cols} for t, cols in sch.items()}, dialect="sqlite",
                   validate_qualify_columns=True, identify=False)
    scopes = list(traverse_scope(tree))
    # 1. base columns -> new names (remember the original); auto table aliases (alias == table name) -> new table name
    for scope in scopes:
        for col in scope.columns:
            src = scope.sources.get(col.table)
            if isinstance(src, exp.Table) and src.name.lower() in low_t:
                t = low_t[src.name.lower()]
                c = low_c[t.lower()].get(col.name.lower())
                if c is None:
                    raise ValueError(f"unresolved column {col.sql()}")
                col.meta["orig"] = c
                col.set("this", exp.to_identifier(m["columns"][t][c]))
                if col.table.lower() == t.lower():
                    col.set("table", exp.to_identifier(m["tables"][t]))
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for tb in tree.find_all(exp.Table):
        n = tb.name.lower()
        if n in low_t and n not in ctes:
            new = m["tables"][low_t[n]]
            if tb.alias and tb.alias.lower() == n:
                tb.set("alias", exp.TableAlias(this=exp.to_identifier(new)))
            tb.set("this", exp.to_identifier(new))
    # 2. qualify adds output aliases carrying the original column name. Inside subqueries / CTEs rename them (and the
    #    outer references that use them); at the top level strip them. Repeat for nested levels.
    top = {id(tree)} | ({id(x) for x in tree.find_all(exp.Select) if x.find_ancestor(exp.Subquery, exp.CTE) is None}
                        if isinstance(tree, exp.Union) else set())
    for _ in range(6):
        derived = {}
        for sel in tree.find_all(exp.Select):
            if id(sel) in top:
                continue
            for proj in sel.expressions:
                if isinstance(proj, exp.Alias) and isinstance(proj.this, exp.Column) and "orig" in proj.this.meta \
                        and proj.alias.lower() == proj.this.meta["orig"].lower() and proj.alias != proj.this.name:
                    derived.setdefault(id(sel), {})[proj.alias.lower()] = (proj.this.name, proj.this.meta["orig"])
                    proj.set("alias", exp.to_identifier(proj.this.name))
        if not derived:
            break
        for scope in scopes:
            for col in scope.columns:
                src = scope.sources.get(col.table)
                if src is not None and not isinstance(src, exp.Table):
                    hit = derived.get(id(src.expression), {}).get(col.name.lower())
                    if hit:
                        col.set("this", exp.to_identifier(hit[0]))
                        col.meta["orig"] = hit[1]
    for sel in tree.find_all(exp.Select):
        if id(sel) in top:
            for proj in list(sel.expressions):
                if isinstance(proj, exp.Alias) and isinstance(proj.this, exp.Column) and "orig" in proj.this.meta \
                        and proj.alias.lower() == proj.this.meta["orig"].lower():
                    proj.replace(proj.this)
    out = tree.sql(dialect="sqlite")
    # 3. leak check: no identifier may still carry an original table or column name
    olds = {t.lower() for t in sch} | {c.lower() for cols in sch.values() for c in cols}
    news = {v.lower() for v in m["tables"].values()} | {v.lower() for cm in m["columns"].values() for v in cm.values()}
    leaked = {i.name.lower() for i in sqlglot.parse_one(out, read="sqlite").find_all(exp.Identifier)} & (olds - news)
    if leaked:
        raise ValueError(f"original names left in rewritten SQL: {sorted(leaked)}")
    return out


IDENT = re.compile(r"[A-Za-z]*[_0-9][A-Za-z0-9_]*|[a-z]+[A-Z][A-Za-z0-9]*")


def rewrite_text(question: str, evidence: str, m: dict, stats: dict) -> tuple[str, str]:
    flat = {f"{t}.{c}": nc for t, cm in m["columns"].items() for c, nc in cm.items()}
    lits = re.findall(r"'([^']*)'|\"([^\"]*)\"", question + " " + evidence)
    lits = [a or b for a, b in lits if (a or b).strip()] + re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", question + " " + evidence)
    olds = {n for t, cm in m["columns"].items() for n in [t, *cm] if IDENT.fullmatch(n) and len(n) >= 4}
    for _ in range(3):
        got = llm(
            "You rewrite a text-to-SQL question and its evidence note for a database whose tables and columns were "
            "renamed.\n1. question: reword it so the meaning, every constraint and every literal value (names, "
            "numbers, quoted strings) stay exactly the same, but the wording differs from the original (change sentence "
            "structure and word choice where possible).\n2. evidence: replace every mention of an old table or column "
            "name with its new name from the mapping; keep everything else exactly, including values and formulas.\n"
            f"Table mapping: {json.dumps(m['tables'])}\nColumn mapping (table.column -> new column): {json.dumps(flat)}\n"
            f"Question: {question}\nEvidence: {evidence}\n"
            'Return JSON {"question": "...", "evidence": "..."}.')
        nq, ne = str(got.get("question", "")).strip(), str(got.get("evidence", "")).strip()
        text = nq + " " + ne
        if nq and all(l in text for l in lits) and not any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(o)}(?![A-Za-z0-9_])", text) for o in olds):
            return nq, ne
    stats["text_fallbacks"] += 1
    ne = evidence
    for o in sorted({n for t, cm in m["columns"].items() for n in cm}, key=len, reverse=True):
        nc = next(cm[o] for cm in m["columns"].values() if o in cm)
        ne = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(o)}(?![A-Za-z0-9_])", nc, ne)
    return question, ne


def write_descriptions(src_dir: Path, dst_dir: Path, m: dict):
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    low_t = {t.lower(): t for t in m["tables"]}
    for f in sorted(src_dir.glob("*.csv")):
        t = low_t.get(f.stem.lower())
        if t is None:
            continue
        low_c = {c.lower().strip(): nc for c, nc in m["columns"][t].items()}
        rows = list(csv.reader(io.StringIO(f.read_bytes().decode("utf-8", errors="replace"))))
        out = io.StringIO()
        w = csv.writer(out)
        for i, row in enumerate(rows):
            if i and row:
                nc = low_c.get(row[0].lower().strip())
                if nc:
                    row = [nc, nc, *row[2:]] if len(row) > 1 else [nc]
            w.writerow(row)
        (dst_dir / f"{m['tables'][t]}.csv").write_text(out.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    rows = [json.loads(l) for l in open(ROOT / "runs/phase2/cost_confirmation/s1_reserved.jsonl")]
    routed = [r for r in rows if r["margin"] is None or r["margin"] < T]
    dbs = sorted({r["db_id"] for r in routed})
    newid = {d: f"db_{k + 1:02d}" for k, d in enumerate(dbs)}
    stats = defaultdict(int)
    schemas = {d: schema_of(SRC_DB / d / f"{d}.sqlite") for d in dbs}
    with ThreadPoolExecutor(8) as ex:
        maps = dict(zip(dbs, ex.map(lambda d: mapping_for(d, schemas[d], stats), dbs)))
    for d in dbs:
        dst = a.box / "dbs" / newid[d] / f"{newid[d]}.sqlite"
        build_db(SRC_DB / d / f"{d}.sqlite", dst, maps[d])
        write_descriptions(SRC_DB / d / "database_description", a.box / "dbs" / newid[d] / "database_description", maps[d])
    (a.out / "mapping.json").write_text(json.dumps({"db_ids": newid, "maps": maps}, indent=1))

    def one(r):
        d = r["db_id"]
        i = int(r["id"].rsplit("_", 1)[1])
        gold = tr[i]["SQL"]
        src, dst = SRC_DB / d / f"{d}.sqlite", a.box / "dbs" / newid[d] / f"{newid[d]}.sqlite"
        rec = {"id": r["id"], "db_id": newid[d], "orig_db_id": d}
        g0 = execute(src, gold)
        try:
            ng = rewrite_sql(gold, schemas[d], maps[d])
        except Exception as e:  # noqa: BLE001
            return {**rec, "excluded": f"gold rewrite failed: {e}"}
        if g0 is None or execute(dst, ng) != g0:
            return {**rec, "excluded": "rewritten gold result differs"}
        cands, dropped = [], 0
        for c in r.get("candidates", []):
            try:
                ns = rewrite_sql(c["sql"], schemas[d], maps[d])
                ok = execute(dst, ns) == execute(src, c["sql"])
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                cands.append({**c, "sql": ns})
            else:
                dropped += 1
        nq, ne = rewrite_text(r["question"], r.get("evidence") or "", maps[d], stats)
        return {**rec, "gold": ng, "question": nq, "evidence": ne, "orig_question": r["question"],
                "candidates": cands, "candidates_dropped": dropped}

    with ThreadPoolExecutor(8) as ex:
        items = list(ex.map(one, routed))
    keep = [x for x in items if "excluded" not in x]
    (a.out / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    by = defaultdict(list)
    for x in keep:
        by[x["db_id"]].append(x)
    (a.box / "chunks/C").mkdir(parents=True, exist_ok=True)
    (a.box / "answers/C").mkdir(parents=True, exist_ok=True)
    (a.box / "work").mkdir(parents=True, exist_ok=True)
    nch = 0
    for d, rs in sorted(by.items()):
        k = -(-len(rs) // 25)
        n = -(-len(rs) // k)
        for j in range(k):
            ch = rs[j * n:(j + 1) * n]
            (a.box / "chunks/C" / f"{d}_{j + 1}.jsonl").write_text("".join(json.dumps(
                {"question_id": f"q{x['id'].rsplit('_', 1)[1]}", "db_id": x["db_id"], "question": x["question"],
                 "evidence": x["evidence"], "candidates": [{k2: c[k2] for k2 in ("label", "sql", "result", "n_of_32")}
                                                           for c in x["candidates"]]}) + "\n" for x in ch))
            nch += 1
    (a.box / "TOOLS").mkdir(exist_ok=True)
    shutil.copyfile(a.box.parent / "s2res/TOOLS/sql.py", a.box / "TOOLS/sql.py")
    summary = {"routed": len(routed), "eligible": len(keep), "excluded": len(items) - len(keep),
               "exclusion_reasons": [x["excluded"] for x in items if "excluded" in x], "chunks": nch,
               "candidates_kept": sum(len(x["candidates"]) for x in keep),
               "candidates_dropped": sum(x["candidates_dropped"] for x in keep), **stats}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
