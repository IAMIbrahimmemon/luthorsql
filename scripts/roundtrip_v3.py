"""Grammar v3 slot round trip: plan -> slot targets -> decode with perfect scores -> SQL -> execute.

The share of questions that survive is the ceiling of the slot model on that set (what a model that
predicts every slot right would score). It also catches any mismatch between targets and decode.

  python scripts/roundtrip_v3.py --plans data/plans_v3/dev/bird_dev --vocab data/plans_v3/templates.json \
      --db-root data/bench/dev_20240627/dev_databases --total 1534
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.candidates import find_candidates  # noqa: E402
from luthor.sq.plan import ROWS, Plan, compile_plan  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq2 import targets as T  # noqa: E402


class Oracle:
    """targets.Scores that put all mass on the gold targets."""

    def __init__(self, t: T.Targets, option_ids: list[str], cands):
        self.t, self.option_ids, self.cands = t, option_ids, cands
        self.ix = {o: i for i, o in enumerate(option_ids)}
        self.pairs = {(s, o, h): c for s, o, h, c in t.pair}
        self.vals_ = {(s, o): idx for s, o, idx in t.vals}

    def _hot(self, n, ks):
        out = [0.0] * n
        for k in ks:
            out[k] = 20.0  # a confident logit (IN keeps values with softmax share >= 0.2)
        return out

    def choice(self, key):
        b, k = key.split(".", 1)
        names = (T.MAIN_CHOICES if k in T.MAIN_CHOICES and b == "m" else T.CHOICES)[k]
        return self._hot(len(names), [self.t.choice.get(key, 0)])

    def ptr(self, slot):
        n = len(self.option_ids)
        if slot in self.t.ptr:
            return self._hot(n, [self.ix[self.t.ptr[slot]]])
        if slot in self.t.sets:
            out = [0.0] * n
            for r, o in enumerate(self.t.sets[slot]):
                out[self.ix[o]] = 1.0 + 0.001 * (len(self.t.sets[slot]) - r)
            return out
        return [0.0] * n

    def pair(self, slot, opt, head):
        return self._hot(len(T.PAIRS[head]), [self.pairs.get((slot, self.option_ids[opt], head), 0)])

    def vals(self, slot, opt):
        return self._hot(len(self.cands), self.vals_.get((slot, self.option_ids[opt]), []))

    def limit(self, slot):
        return self._hot(len(self.cands), [self.t.limit[slot]] if slot in self.t.limit else [])


def option_ids(card: SchemaCard) -> list[str]:
    out = []
    for t in card.tables:
        out.append(f"{t.name}.{ROWS}")
        out += [f"{t.name}.{c.name}" for c in t.columns]
    return out


def rows(con, sql):
    return {tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in con.execute(sql).fetchmany(200_001)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path, required=True, help="prepare_plans output folder (plans.jsonl, cards/)")
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--db-root", type=Path, required=True)
    ap.add_argument("--total", type=int, required=True, help="questions in the set (the denominator)")
    a = ap.parse_args()
    T.set_vocab(json.loads(a.vocab.read_text()), json.loads(a.vocab.with_name(a.vocab.stem + "_nums.json").read_text()))
    stats, ok = collections.Counter(), 0
    cards, cons = {}, {}
    for line in open(a.plans / "plans.jsonl"):
        r = json.loads(line)
        if r.get("outcome") != "plan" or r.get("twin_of"):
            continue
        db = r["db_id"]
        if db not in cards:
            cards[db] = SchemaCard.load(a.plans / "cards" / f"{db}.json")
            cons[db] = sqlite3.connect(f"file:{a.db_root / db / (db + '.sqlite')}?mode=ro", uri=True)
            cons[db].text_factory = lambda b: b.decode("utf-8", "replace")
        card = cards[db]
        text = r["question"] + (f"\nContext: {r['context']}" if r.get("context") else "")
        cands = find_candidates(text, card, extra=[tuple(x) for x in r.get("db_values") or []])
        try:
            t = T.build_targets(Plan.from_dict(r["plan"]), cands)
        except T.Unsupported as e:
            stats["targets: " + str(e)] += 1
            continue
        try:
            plan = T.decode(Oracle(t, option_ids(card), cands))
            sql = compile_plan(plan, card.graph())
            same = rows(cons[db], sql) == rows(cons[db], r["compiled_sql"])
        except Exception as e:
            stats[f"decode/compile: {type(e).__name__}: {str(e)[:60]}"] += 1
            continue
        if same:
            ok += 1
        else:
            stats["round trip differs"] += 1
            if stats["round trip differs"] <= 5:
                print("DIFF", r["id"], "\n  gold-plan:", r["compiled_sql"][:200], "\n  decoded:  ", sql[:200])
    print(f"ROUNDTRIP slot-encodable and exact: {ok} = {100 * ok / a.total:.1f}% of {a.total}")
    for k, v in stats.most_common():
        print(f"{v:5d}  {k}")


if __name__ == "__main__":
    main()
