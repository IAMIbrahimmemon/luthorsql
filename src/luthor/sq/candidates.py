"""Literal candidates a filter (or LIMIT) can point at, found deterministically.

Two kinds:
  * value  - a stored value of an indexed text column that the question (or context)
             mentions, plus every value of very low-cardinality columns (codes like 'F'/'M'
             whose meaning only the description gives). Encoded like options.
  * span   - numbers, dates, years, quoted strings and capitalised phrases in the question.
             Represented by pooling the question's own token states over the span, so they
             cost nothing extra at inference.
The model's value pointer chooses among these; it never invents a literal.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict

from .schema_card import SchemaCard

_LOG = logging.getLogger("luthorsql")

LOWCARD_ENUM = 12
_NUM = re.compile(r"(?<!\w)(?<!\d\.)-?\d+(?:\.\d+)?(?!\w|\.\d)")  # "No.933" counts, "0.5" stays whole
_DATE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{4}[-/]\d{1,2}\b")
_QUOTED = re.compile(r"(?<!\w)['\"‘’“”]([^'\"‘’“”]{1,80})['\"‘’“”](?!\w)")
_CAPS = re.compile(r"\b(?:[A-Z][\w\-'&.]*)(?:\s+(?:of|de|the|and|&|[A-Z][\w\-'&.]*))*")
_WORD = re.compile(r"[a-z0-9]+")
_MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                            "september", "october", "november", "december"])}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_MDY = re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DATE_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON}\.?,?\s+(\d{{4}})\b", re.I)
_DATE_US = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b")
_DATE_MY = re.compile(rf"\b{_MON}\.?\s+(\d{{4}})\b", re.I)
_NUM_SEP = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\w|\.\d)")
_NUM_SCALE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(k|thousand|m|mn|million|b|bn|billion)\b", re.I)
_NUM_UNIT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(cm|mm|km|kg|g|lbs?|ft|mph|kmh|hrs?|mins?|secs?|yrs?|gb|mb|tb)\b", re.I)
_PCT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
_SMALL = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_SMALL.update({"once": 1, "twice": 2, "thrice": 3, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
               "hundred": 100, "dozen": 12})
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


@dataclass
class Candidate:
    kind: str           # value | span
    value: str | int | float
    col: str | None     # value candidates: "table.column"
    span: tuple[int, int] | None  # char offsets into the model input text (spans, and value hits)

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(s: str) -> str:
    return " ".join(_WORD.findall(str(s).lower()))


def _num(s: str):
    return int(s) if re.fullmatch(r"-?\d+", s) else float(s)


_STOP = set("the a an of in on at to for and or with by from is are was were be has have had what which who whom "
             "how many much all any each every list show give find name names number count".split())


class ValueIndex:
    """Normalised value -> [(column, value)] for the text columns too large for the card's value list
    (more than the card's cap, up to `max_distinct`). Built from the database once at connect time
    (serving) or per database at data-prep time; questions are matched against it by n-grams."""

    def __init__(self, by_norm: dict[str, list[tuple[str, str]]]):
        self.by_norm = by_norm

    @classmethod
    def build(cls, con, card: SchemaCard, max_distinct: int = 200_000, max_len: int = 80,
              skip_card_values: bool = True) -> "ValueIndex":
        by_norm: dict[str, list[tuple[str, str]]] = {}
        for t in card.tables:
            for c in t.columns:
                cid = f"{t.name}.{c.name}"
                texty = any(k in c.type.lower() for k in ("text", "char", "string")) or c.type == ""
                if not texty or (skip_card_values and cid in card.values):
                    continue
                try:
                    rows = con.execute(f'SELECT DISTINCT "{c.name}" FROM "{t.name}" WHERE "{c.name}" IS NOT NULL '
                                       f'LIMIT {max_distinct + 1}').fetchall()
                except Exception:
                    _LOG.warning("value scan skipped for %s.%s", t.name, c.name, exc_info=True)
                    continue
                if len(rows) > max_distinct:
                    continue
                for (v,) in rows:
                    if isinstance(v, str) and 0 < len(v) <= max_len:
                        k = _norm(v)
                        if k and k not in _STOP:
                            lst = by_norm.setdefault(k, [])
                            if len(lst) < 4:
                                lst.append((cid, v))
        return cls(by_norm)

    def match(self, text: str, max_hits: int = 24) -> list[tuple[str, str, int, int]]:
        """(column, value, start, end) for question n-grams (longest first) that equal a stored value."""
        low = text.lower()
        toks = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(low)]
        out, covered = [], set()
        for n in range(min(6, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(j in covered for j in range(i, i + n)):
                    continue
                k = " ".join(t for t, _, _ in toks[i:i + n])
                if n == 1 and (len(k) < 3 or k.isdigit() or k in _STOP):
                    continue
                hits = self.by_norm.get(k)
                if hits:
                    for col, v in hits:
                        out.append((col, v, toks[i][1], toks[i + n - 1][2]))
                    covered.update(range(i, i + n))
                    if len(out) >= max_hits:
                        return out
        return out


def find_candidates(text: str, card: SchemaCard, max_value_hits: int = 40,
                    extra: list | None = None) -> list[Candidate]:
    """`extra`: [(column, value, start, end)] database-lookup hits (ValueIndex.match on this text)."""
    out: list[Candidate] = []
    seen: set[tuple] = set()

    def add(c: Candidate):
        key = (c.kind, str(c.value), c.col)
        if key not in seen:
            seen.add(key)
            out.append(c)

    # spans
    for m in _DATE.finditer(text):
        add(Candidate("span", m.group(0), None, (m.start(), m.end())))
        if "/" in m.group(0):
            add(Candidate("span", m.group(0).replace("/", "-"), None, (m.start(), m.end())))
        parts = re.split(r"[-/]", m.group(0))  # 1995-9-4 -> 1995-09-04
        add(Candidate("span", "-".join([parts[0]] + [f"{int(x):02d}" for x in parts[1:]]), None, (m.start(), m.end())))
    for m in _DATE_US.finditer(text):  # 9/27/1993 -> 1993-09-27
        add(Candidate("span", f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}", None, (m.start(), m.end())))
    for m in _TIME.finditer(text):  # 0:01:40 -> 1:40 (lap and qualifying times are stored as m:ss)
        add(Candidate("span", m.group(0), None, (m.start(), m.end())))
        short = re.sub(r"^0:0?", "", m.group(0))
        if short != m.group(0):
            add(Candidate("span", short, None, (m.start(), m.end())))
    for m in re.finditer(rf"\b{_MON}\b(?!\.?\s+\d)", text, re.I):  # a month alone: '09' (and 9 below)
        mo = _MONTHS[m.group(1).lower()[:3]]
        if m.group(1).lower() != "may" or m.group(1)[0] == "M":
            add(Candidate("span", f"{mo:02d}", None, (m.start(), m.end())))
            add(Candidate("span", mo, None, (m.start(), m.end())))
    for m in _DATE_MDY.finditer(text):
        mo = _MONTHS[m.group(1).lower()[:3]]
        add(Candidate("span", f"{m.group(3)}-{mo:02d}-{int(m.group(2)):02d}", None, (m.start(), m.end())))
    for m in _DATE_DMY.finditer(text):
        mo = _MONTHS[m.group(2).lower()[:3]]
        add(Candidate("span", f"{m.group(3)}-{mo:02d}-{int(m.group(1)):02d}", None, (m.start(), m.end())))
    for m in _DATE_MY.finditer(text):
        mo = _MONTHS[m.group(1).lower()[:3]]
        add(Candidate("span", f"{m.group(2)}-{mo:02d}", None, (m.start(), m.end())))
    for m in _NUM_SEP.finditer(text):
        add(Candidate("span", _num(m.group(0).replace(",", "")), None, (m.start(), m.end())))
    for m in _NUM_SCALE.finditer(text):
        v = float(m.group(1)) * _SCALE[m.group(2).lower()]
        add(Candidate("span", int(v) if v == int(v) else v, None, (m.start(), m.end())))
    for m in _NUM_UNIT.finditer(text):
        add(Candidate("span", _num(m.group(1)), None, (m.start(1), m.end(1))))
    for m in _PCT.finditer(text):
        v = _num(m.group(1))
        add(Candidate("span", v / 100, None, (m.start(), m.end())))
    for m in re.finditer(r"\b[a-z]+\b", text.lower()):
        if m.group(0) in _SMALL:
            add(Candidate("span", _SMALL[m.group(0)], None, (m.start(), m.end())))
    taken = [(m.start(), m.end()) for rx in (_NUM_SEP, _NUM_SCALE, _DATE) for m in rx.finditer(text)]
    for m in _NUM.finditer(text):
        if any(a <= m.start() and m.end() <= b and (a, b) != (m.start(), m.end()) for a, b in taken):
            continue  # "50" inside "50,000", "2004" inside "2004-08-09"
        add(Candidate("span", _num(m.group(0)), None, (m.start(), m.end())))
    for m in _QUOTED.finditer(text):
        add(Candidate("span", m.group(1), None, (m.start(1), m.end(1))))
    for m in _CAPS.finditer(text):
        s = m.group(0).strip(" .")
        if len(s) > 1 and m.start() > 0:
            add(Candidate("span", s, None, (m.start(), m.start() + len(s))))

    # value-index hits: n-grams of the text against every indexed value
    low = text.lower()
    hits = 0
    by_norm: dict[str, list[tuple[str, str]]] = {}
    for col, vals in card.values.items():
        for v in vals:
            k = _norm(v)
            if k:
                by_norm.setdefault(k, []).append((col, v))
    toks = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(low)]
    for n in range(min(6, len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            k = " ".join(t for t, _, _ in toks[i:i + n])
            if n == 1 and (len(k) < 2 or k.isdigit()):
                continue
            for col, v in by_norm.get(k, [])[:4]:
                add(Candidate("value", v, col, (toks[i][1], toks[i + n - 1][2])))
                hits += 1
            if hits >= max_value_hits:
                break
    # looser matches: plural/singular of single-word values, and multi-word values whose words
    # all occur in the question (any order), for values not already found
    words = {t for t, _, _ in toks}
    stems = {w.rstrip("s"): (a, b) for w, a, b in toks if len(w) > 3}
    for k, pairs in by_norm.items():
        if hits >= max_value_hits:
            break
        parts = k.split()
        if len(parts) == 1 and len(k) > 3 and k.rstrip("s") in stems and k not in words:
            a, b = stems[k.rstrip("s")]
            for col, v in pairs[:2]:
                add(Candidate("value", v, col, (a, b)))
                hits += 1
        elif 1 < len(parts) <= 4 and all(p in words for p in parts) and k not in low:
            spans = [(a, b) for w, a, b in toks if w in parts]
            for col, v in pairs[:2]:
                add(Candidate("value", v, col, (min(x for x, _ in spans), max(y for _, y in spans))))
                hits += 1
    for col, v, a, b in extra or []:
        add(Candidate("value", v, col, (a, b)))
    # a stored value with stray whitespace ('Norway  ') is dropped when its trimmed twin exists
    vals_by_col = {}
    for c in out:
        if c.kind == "value":
            vals_by_col.setdefault(c.col, set()).add(str(c.value))
    out[:] = [c for c in out if not (c.kind == "value" and str(c.value) != str(c.value).strip()
                                     and str(c.value).strip() in vals_by_col.get(c.col, ()))]
    # flag columns (0/1, Y/N stored as numbers): enumerate their values ("primarily affiliated" -> 1)
    for t in card.tables:
        for c in t.columns:
            if c.n_distinct is not None and c.n_distinct <= 2 and c.samples and f"{t.name}.{c.name}" not in card.values:
                for v in c.samples[:2]:
                    add(Candidate("value", _num(v) if re.fullmatch(r"-?\d+(\.\d+)?", v) else v, f"{t.name}.{c.name}", None))
    # enumerate very low-cardinality columns (codes)
    for col, vals in card.values.items():
        if len(vals) <= LOWCARD_ENUM:
            for v in vals:
                add(Candidate("value", v, col, None))
    return out


def ground(value, col_id: str, cands: list[Candidate]) -> int | None:
    """Index of the candidate that produces this gold literal for this column, or None.
    Prefers a stored value of the same column, then an exact span, then a case-insensitive one."""
    sv = str(value)
    for i, c in enumerate(cands):
        if c.kind == "value" and c.col == col_id and str(c.value) == sv:
            return i
    for i, c in enumerate(cands):
        if c.kind == "span":
            if isinstance(value, (int, float)) and isinstance(c.value, (int, float)) and float(c.value) == float(value):
                return i
            if str(c.value) == sv:
                return i
    for i, c in enumerate(cands):
        if c.kind == "value" and c.col == col_id and str(c.value).lower() == sv.lower():
            return i
    for i, c in enumerate(cands):
        if c.kind == "span" and str(c.value).lower() == sv.lower():
            return i
    return None
