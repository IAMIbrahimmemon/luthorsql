"""Paired comparison of two systems on BIRD dev with the exact tests in luthor.stats (replaces paired_bootstrap.py
for decisions; that file's percentile interval is anti-conservative with 11 databases, see
scripts/analysis/paired_test_size.out).

  python scripts/analysis/paired_test.py A_scored.jsonl B_scored.jsonl A.stack_cons.jsonl B.stack_cons.jsonl

Prints, for first pick / validator / oracle@8 / VAL+DROP: A -> B in points, the exact DB-level sign-flip p and its
inverted 95% interval (the claim "would generalise to new databases"), the exact McNemar p (the claim "better on
these questions" only), and the old percentile interval for comparison.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from luthor.stats import cluster_signflip, percentile_bootstrap, question_signflip  # noqa: E402


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def main(a_sc, b_sc, a_st, b_st):
    B0, B1, S0, S1 = load(a_sc), load(b_sc), load(a_st), load(b_st)
    ids = sorted(B0)
    assert set(ids) == set(B1) == set(S0) == set(S1), "files must cover the same questions"
    db = {i: B0[i]["db_id"] for i in ids}
    first = lambda r: bool(r["cands"]) and bool(r["cands"][0].get("ok"))
    orc = lambda r: any(c.get("ok") for c in r["cands"])
    M = {"first pick": ({i: first(B0[i]) for i in ids}, {i: first(B1[i]) for i in ids}),
         "validator": ({i: S0[i]["res"]["V:-"][0] for i in ids}, {i: S1[i]["res"]["V:-"][0] for i in ids}),
         "oracle@8": ({i: orc(B0[i]) for i in ids}, {i: orc(B1[i]) for i in ids}),
         "VAL+DROP": ({i: S0[i]["res"]["V:VAL+DROP"][0] for i in ids}, {i: S1[i]["res"]["V:VAL+DROP"][0] for i in ids})}
    n = len(ids)
    print(f"n={n}, {len(set(db.values()))} databases")
    for name, (a, b) in M.items():
        c = cluster_signflip(a, b, db)
        q = question_signflip(a, b)
        lo, hi = percentile_bootstrap(a, b, db)
        print(f"{name:10s} {100*sum(a.values())/n:6.2f} -> {100*sum(b.values())/n:6.2f}  delta {c['delta']:+5.2f}  "
              f"DB sign-flip p={c['p']:.3f} 95% CI [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]  "
              f"McNemar p={q['p']:.3f} (+{q['wins']}/-{q['losses']})  [old percentile CI {lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main(*sys.argv[1:5])
