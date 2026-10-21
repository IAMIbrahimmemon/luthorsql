"""Contamination filter against BIRD dev: near-duplicate questions and dev databases.

Training rows (teacher-generated, synthetic or mined) must not copy BIRD dev questions or come from BIRD dev
databases, or every later dev number is optimistic. Two checks:

* `near_dup(questions, ref)`: True where a question's character-shingle Jaccard similarity to any reference text is at
  least `threshold`. Exact (no MinHash approximation): reference shingles go into an inverted index and each query
  probes only its rarest `|A| - ceil(t*|A|) + 1` shingles (prefix filtering), which every pair with Jaccard >= t must
  share, then verifies candidates exactly. About 10 s for 10k queries vs 1.5k references on one core.
* `dev_db_flags(db_ids)` / `is_dev_db(db_id)`: True for the 11 BIRD dev databases (name match after normalisation);
  `schema_near_dev(tables)` also catches a renamed copy by table-name overlap.

`load_dev_questions()` returns the ORIGINAL BIRD dev (2024-06-27) texts, question + " " + evidence by default; build
the query side the same way (question + evidence) when the rows carry evidence; a question-only query against
question+evidence references loses recall (60% of exact dev questions at 0.6), so for rows without evidence pass
`load_dev_questions(with_evidence=False)` as `ref`.

Calibration (scripts/analysis/contamination_calib.out): at the default 0.6 on character 5-grams, 1 of 9,428 BIRD-train
questions is flagged against dev (q+evidence; 4 question-only), none at 0.7; recall on synthetic dev near-copies is
100% exact, 98.8% one word dropped, 87.7% two dropped, 92.5% one content word replaced, 99.5% with a prefix.
Speed: ~0.3-0.5 ms per query against 1,534 references (500k queries ~3-4 min on one core).
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEV_JSON = REPO / "data/bench/dev_20240627/dev.json"
DEV_TABLES = REPO / "data/bench/dev_20240627/dev_tables.json"
DEFAULT_THRESHOLD = 0.6
DEFAULT_N = 5

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """NFKC, lower case, punctuation -> space, collapsed whitespace."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def shingles(text: str, n: int = DEFAULT_N, unit: str = "char") -> frozenset:
    """Set of n-grams of the normalised text: character n-grams (unit='char') or word n-grams (unit='word').
    Texts shorter than n give one shingle (the whole text); empty text gives the empty set."""
    t = normalize(text)
    if not t:
        return frozenset()
    if unit == "word":
        w = t.split()
        return frozenset([" ".join(w)]) if len(w) < n else frozenset(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
    if unit != "char":
        raise ValueError(f"unit must be 'char' or 'word', got {unit!r}")
    return frozenset([t]) if len(t) < n else frozenset(t[i:i + n] for i in range(len(t) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)


class RefIndex:
    """Inverted shingle index over reference texts; `best(text, threshold)` -> (max Jaccard, ref index) over refs with
    Jaccard >= threshold, else (0.0, -1). Exact for every pair at or above `threshold`."""

    def __init__(self, ref: list[str], n: int = DEFAULT_N, unit: str = "char"):
        self.n, self.unit = n, unit
        self.sets = [shingles(r, n, unit) for r in ref]
        self.post: dict[str, list[int]] = defaultdict(list)
        for j, s in enumerate(self.sets):
            for g in s:
                self.post[g].append(j)
        self.df = {g: len(p) for g, p in self.post.items()}

    def best(self, text: str, threshold: float) -> tuple[float, int]:
        a = shingles(text, self.n, self.unit)
        if not a:
            return 0.0, -1
        need = max(1, math.ceil(threshold * len(a) - 1e-9))  # J(A,B) >= t  =>  |A & B| >= t * |A|
        prefix = sorted(a, key=lambda g: (self.df.get(g, 0), g))[:len(a) - need + 1]
        cands = set()
        for g in prefix:
            cands.update(self.post.get(g, ()))
        best, arg = 0.0, -1
        for j in cands:
            jac = jaccard(a, self.sets[j])
            if jac > best:
                best, arg = jac, j
        return (best, arg) if best >= threshold else (0.0, -1)


def max_similarity(questions: list[str], ref: list[str], floor: float = 0.3, n: int = DEFAULT_N,
                   unit: str = "char") -> list[tuple[float, int]]:
    """Per question: (best Jaccard to any ref, ref index) when it is >= floor, else (0.0, -1)."""
    idx = RefIndex(ref, n, unit)
    return [idx.best(q, floor) for q in questions]


def near_dup(questions: list[str], ref: list[str], threshold: float = DEFAULT_THRESHOLD, *, n: int = DEFAULT_N,
             unit: str = "char") -> list[bool]:
    """True where a question's shingle Jaccard to some reference text is >= threshold (char 5-grams by default)."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    idx = RefIndex(ref, n, unit)
    return [idx.best(q, threshold)[1] >= 0 for q in questions]


def _dev_rows() -> list[dict]:
    return json.load(open(DEV_JSON))


def load_dev_questions(with_evidence: bool = True) -> list[str]:
    """ORIGINAL BIRD dev (2024-06-27), all 1,534 questions in file order: question + " " + evidence (or question only)."""
    return [(r["question"] + " " + (r.get("evidence") or "")).strip() if with_evidence else r["question"]
            for r in _dev_rows()]


def _norm_db(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


@lru_cache(maxsize=1)
def dev_db_ids() -> frozenset:
    """The 11 BIRD dev database ids."""
    return frozenset(r["db_id"] for r in _dev_rows())


def is_dev_db(db_id: str) -> bool:
    nd = {_norm_db(d) for d in dev_db_ids()}
    return _norm_db(db_id) in nd


def dev_db_flags(db_ids: list[str]) -> list[bool]:
    nd = {_norm_db(d) for d in dev_db_ids()}
    return [_norm_db(d) in nd for d in db_ids]


@lru_cache(maxsize=1)
def dev_table_sets() -> dict:
    """dev db_id -> frozenset of normalised table names (from dev_tables.json)."""
    return {t["db_id"]: frozenset(_norm_db(x) for x in t["table_names_original"]) for t in json.load(open(DEV_TABLES))}


def schema_near_dev(tables: list[str], threshold: float = 0.6) -> str | None:
    """The BIRD dev db_id whose table-name set has Jaccard >= threshold with `tables`, else None (renamed copies)."""
    s = frozenset(_norm_db(x) for x in tables)
    best, arg = 0.0, None
    for db, ts in dev_table_sets().items():
        j = jaccard(s, ts)
        if j > best:
            best, arg = j, db
    return arg if best >= threshold else None


def flag_rows(rows: list[dict], threshold: float = DEFAULT_THRESHOLD, ref: list[str] | None = None) -> list[dict]:
    """Rows with db_id / question / optional evidence -> per row {"dev_db": bool, "near_dup": bool}.
    The query text is question + evidence, matching load_dev_questions()."""
    ref = load_dev_questions() if ref is None else ref
    texts = [(r.get("question", "") + " " + (r.get("evidence") or "")).strip() for r in rows]
    nd = near_dup(texts, ref, threshold)
    db = dev_db_flags([r.get("db_id", "") for r in rows])
    return [{"dev_db": a, "near_dup": b} for a, b in zip(db, nd)]


def counts(flags: list[bool]) -> Counter:
    return Counter(flags)
