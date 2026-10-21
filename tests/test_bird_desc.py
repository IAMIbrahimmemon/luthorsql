"""BIRD-train column descriptions in schema cards (luthor.sq.bird_desc): parsing, one-line text, opt-in flag."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from luthor.sq import bird_desc  # noqa: E402
from luthor.sq.schema_card import ColumnCard, SchemaCard, TableCard  # noqa: E402
from luthor.sq2.serialize import schema_text  # noqa: E402
from luthor.sqlw.prompt6 import ddl6  # noqa: E402

CSV = ("﻿original_column_name,column_name,column_description,data_format,value_description\n"
       "playerID,,id number identifying the player,text,\n"
       "nameGiven,name given,Given name,text,\n"
       "firstName,first name,first name,text,\n"
       "hrefID,,,text,\n"
       "PIM,Penalty minutes,Penalty minutes,integer,\"commonsense evidence:\nhigher means more fouls\"\n")


def _card():
    return SchemaCard("hockey", "sqlite", [TableCard("Master", "", [ColumnCard(n, "TEXT") for n in
                                                                    ("playerID", "nameGiven", "firstName", "hrefID", "PIM")]),
                                           TableCard("Other", "", [ColumnCard("x", "INTEGER", "kept")])], [])


def _dir(tmp_path):
    d = tmp_path / "desc" / "hockey"
    d.mkdir(parents=True)
    (d / "master.csv").write_bytes(CSV.encode("utf-8"))
    return tmp_path / "desc"


def test_parse_and_describe(tmp_path):
    descs = bird_desc.load_dir(_dir(tmp_path))
    cols = descs["hockey"]["master"]
    assert set(cols) == {"playerid", "namegiven", "firstname", "hrefid", "pim"}
    assert bird_desc.describe("playerID", cols["playerid"]) == "id number identifying the player"
    assert bird_desc.describe("nameGiven", cols["namegiven"]) == "Given name"  # expanded name not repeated
    assert bird_desc.describe("firstName", cols["firstname"]) == ""  # says nothing beyond the name
    assert bird_desc.describe("hrefID", cols["hrefid"]) == ""
    assert bird_desc.describe("PIM", cols["pim"]) == "Penalty minutes; higher means more fouls"
    long = ("x", "a " * 50, "b " * 60)
    assert len(bird_desc.describe("c", long, cap=120)) <= 120


def test_flag_off_by_default_and_on(tmp_path, monkeypatch):
    root = _dir(tmp_path)
    p = tmp_path / "hockey.json"
    _card().save(p)
    monkeypatch.delenv(bird_desc.ENV, raising=False)
    off = SchemaCard.load(p)
    assert all(not c.description for c in off.tables[0].columns)
    base = schema_text(off).text
    monkeypatch.setenv(bird_desc.ENV, str(root))
    on = SchemaCard.load(p)
    d = {c.name: c.description for c in on.tables[0].columns}
    assert d["playerID"] == "id number identifying the player" and d["firstName"] == ""
    assert on.tables[1].columns[0].description == "kept"  # existing descriptions are never overwritten
    txt = schema_text(on).text
    assert "nameGiven (text): Given name" in txt and len(txt) > len(base)
    assert "-- id number identifying the player" in ddl6(on)
    # a database with no description directory is unchanged
    other = SchemaCard("other_db", "sqlite", [TableCard("Master", "", [ColumnCard("PIM", "INTEGER")])], [])
    assert bird_desc.apply(other) == 0
