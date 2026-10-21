"""What may be prompted, and what may become training data.

Two levels:
  never_prompt  BIRD dev databases (data/sys1/bird_dev_dbs.json) and sealed sources. A row on one of these is
                refused before any prompt is built, in every job, bake-off included.
  never_train   held-out evaluation databases (the BIRD-train held-out dbs, Spider dev/test, sealed sets, the mix
                dev lists) plus everything never_prompt covers. They may be labelled for evaluation (the teacher
                bake-off), but export refuses them, and refuses any run marked eval_only.

Contamination: luthor.contamination.near_dup / load_dev_questions (owned by another module). When it cannot be
imported the export refuses unless explicitly allowed, and records "contamination": "unchecked" in the manifest.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"

BIRD_DEV_FILE = DATA / "sys1/bird_dev_dbs.json"
HELD_OUT_FILES = [DATA / "sys1/dev_dbs_bird.json", DATA / "sys1/dev_dbs.json", DATA / "sys1/dev_dbs_mix.json",
                  DATA / "mix2_dev_dbs.json", DATA / "sys1/sealed/kaggledbqa.dbs.json",
                  DATA / "sys1/sealed/spider_dk.dbs.json", DATA / "sys1/sealed/spider_realistic.dbs.json"]
SPIDER_EVAL = [DATA / "raw/spider_data/dev.json", DATA / "raw/spider_data/test.json"]
# the 11 BIRD dev databases, in case the json is missing (same list qwrite.py hard-codes)
BIRD_DEV_11 = {"california_schools", "card_games", "codebase_community", "debit_card_specializing",
               "european_football_2", "financial", "formula_1", "student_club", "superhero",
               "thrombosis_prediction", "toxicology"}
SEALED_SOURCES = ("bird_dev", "spider_dk", "spider_realistic", "kaggledbqa", "sealed", "spider_dev", "spider_test")
SEALED_PATH_PARTS = ("sealed", "bird_dev", "spider_dk", "spider_realistic", "kaggledbqa", "dev_frozen")


class Refused(Exception):
    pass


def bird_dev_dbs() -> set[str]:
    try:
        return set(json.loads(BIRD_DEV_FILE.read_text())) | BIRD_DEV_11
    except OSError:
        return set(BIRD_DEV_11)


def held_out_dbs() -> set[str]:
    out = bird_dev_dbs()
    for f in HELD_OUT_FILES:
        try:
            out |= set(json.loads(f.read_text()))
        except OSError:
            pass
    for f in SPIDER_EVAL:
        try:
            out |= {q["db_id"] for q in json.loads(f.read_text())}
        except OSError:
            pass
    return out


def check_input_path(path: str | Path) -> None:
    """Inputs from a sealed or BIRD-dev file are refused by name before they are read."""
    low = str(Path(path).resolve()).lower().replace("-", "_")
    for part in SEALED_PATH_PARTS:
        if part in low:
            raise Refused(f"refusing input {path}: looks like a sealed / BIRD-dev file ({part})")


def _norm_db(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def check_prompt_row(row: dict, bird_dev: set[str]) -> None:
    if row["db_id"] in bird_dev or _norm_db(row["db_id"]) in {_norm_db(d) for d in bird_dev}:
        raise Refused(f"BIRD dev database {row['db_id']} (row {row.get('id')})")
    src = str(row.get("source", "")).lower()
    if any(s in src for s in SEALED_SOURCES):
        raise Refused(f"sealed source {src} (row {row.get('id')})")


def contamination_checker():
    """(near_dup, load_dev_questions) or None, with a clear warning when the module is not there yet."""
    try:
        from luthor.contamination import load_dev_questions, near_dup  # noqa: F401
        return near_dup, load_dev_questions
    except Exception as e:  # TODO: drop this fallback once luthor.contamination lands
        warnings.warn(f"luthor.contamination not importable ({type(e).__name__}: {e}); export contamination "
                      "check is UNAVAILABLE", stacklevel=2)
        return None


def near_dup_flags(texts: list[str]) -> list[bool] | None:
    """True = near duplicate of a BIRD dev question (drop). texts = question + " " + evidence, the same form
    load_dev_questions() returns. None when the checker is unavailable."""
    cc = contamination_checker()
    if cc is None:
        return None
    near_dup, load_dev_questions = cc
    try:
        ref = load_dev_questions()
    except TypeError:  # signature not settled yet; the module owner decides the default sources
        ref = load_dev_questions(None)
    return list(near_dup(texts, list(ref)))


def near_dev_schema(db_id: str, tables: list[str]) -> str | None:
    """The BIRD dev db this database is (or copies: e.g. Spider's formula_1 has BIRD dev formula_1's tables), else
    None. Uses luthor.contamination.dev_db_flags / schema_near_dev; name check only when it is unavailable."""
    try:
        from luthor.contamination import dev_db_flags, schema_near_dev
    except Exception:
        return db_id if _norm_db(db_id) in {_norm_db(d) for d in bird_dev_dbs()} else None
    if dev_db_flags([db_id])[0]:
        return db_id
    return schema_near_dev(tables)
