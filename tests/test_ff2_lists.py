"""Phase-2 mixed lists (scripts/val2/build_lists.py --ff-extend) and the ff2 eval helpers (CPU, no DB)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/val2"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import build_lists as bl  # noqa: E402
import ff2_eval as fe  # noqa: E402


def g(rank, sql, sig, ok=False, **kw):
    return {"rank": rank, "cost": float(rank), "sql": sql, "runs": True, "nonnull": True, "ok": ok, "sig": sig,
            "res": "1 row", "src": "r1", **kw}


def ff(rank, sql, sig, ok=False, live=True, raw=None):
    c = {"rank": rank, "sql": sql, "src": "ff", "gen": "m", "temp": 1.0, "runs": live, "nonnull": live, "ok": ok,
         "sig": sig if live else None, "res": "1 row", "ff_n": 3, "ff_sig_n": 5}
    if raw:
        c["sql_raw"] = raw
    return c


Q = {"id": "bird_train_train_10", "db_id": "d", "question": "q", "context": "", "gold_sql": "GOLD",
     "gold_sql_orig": "GOLD_ORIG", "in_slots": True, "covered": True, "gold_corrected": False}
canon = str.lower


def test_ff_classes_hygiene_and_representative():
    smp = {"qwen": [ff(0, "Q_A", "sa", True), ff(1, "Q_A2", "sa", True), ff(2, "Q_A2", "sa", True),
                    ff(3, "gold", "sg", True), ff(4, None, None, live=False), ff(5, "dead", "x", live=False)],
           "omni": [ff(0, "O_A", "sa", True), ff(1, "O_B", "sb"), ff(2, "x", "sc", raw="GOLD_ORIG")]}
    cl, st = bl.ff_classes(smp, {"gold", "gold_orig"}, canon)
    assert set(cl) == {"sa", "sb"}                       # gold copies (normalised or raw text) dropped
    assert st["gold_copy_qwen"] == 1 and st["gold_copy_omni"] == 1 and st["no_sql"] == 1 and st["dead"] == 1
    assert cl["sa"]["src"] == "qwen" and cl["sa"]["rep"]["sql"] == "Q_A2"   # clean source's most frequent text
    assert cl["sa"]["srcs"] == ["omni", "qwen"] and cl["sa"]["support"] == 4
    assert cl["sb"]["src"] == "omni" and not cl["sb"]["ok"]


def test_cap_source_rate_matches_clean_rate():
    k = lambda ok, srcs: {"ok": ok, "srcs": srcs}  # noqa: E731
    by = {f"q{i}": {"r": k(True, ["omni"]), "w": k(False, ["omni"])} for i in range(10)}  # omni: 10/20 right
    by.update({f"p{i}": {"r": k(True, ["qwen"]), "w1": k(False, ["qwen"]), "w2": k(False, ["qwen"]),
                         "w3": k(False, ["qwen"])} for i in range(5)})                     # qwen: 5/20 right
    st = bl.cap_source_rate(by, "qwen", "omni", seed=0)
    assert st["clean_rate"] == 0.25 and st["capped_rate_before"] == 0.5
    assert st["dropped"] == 7 and st["capped_rate_after"] <= 0.25   # (10 - d) / (20 - d) <= 0.25 -> d = 7
    assert sum("r" in v for q, v in by.items() if q.startswith("q")) == 3
    st2 = bl.cap_source_rate(by, "qwen", "omni", seed=0)            # already at the target: nothing more dropped
    assert st2["dropped"] == 0


def test_cap_never_drops_shared_or_clean_right():
    by = {"a": {"r": {"ok": True, "srcs": ["omni", "qwen"]}, "w": {"ok": False, "srcs": ["qwen"]}}}
    st = bl.cap_source_rate(by, "qwen", "omni", seed=0)
    assert st["dropped"] == 0 and "r" in by["a"]


def test_mix_list_adds_new_ff_swaps_shared_and_strips_fields():
    glist = {**Q, "weight": 3.0, "cands": [g(0, "G_R", "sr", True), g(1, "G_W1", "s1"), g(2, "G_W2", "s2"),
                                           {**g(100, "MUT", "sm"), "mutant": True, "mutation": "x100"}]}
    smp = {"qwen": [ff(0, "F_R", "sr", True), ff(1, "F_W1", "s1"), ff(2, "F_N1", "n1"), ff(3, "F_N1", "n1"),
                    ff(4, "F_N2", "n2"), ff(5, "F_M", "sm")]}
    ffc, _ = bl.ff_classes(smp, set(), canon)
    rec, st = bl.mix_list(Q, glist, ffc, [], seed=0)
    by_sig = {c["sig"]: c for c in rec["cands"]}
    assert set(by_sig) == {"sr", "s1", "s2", "sm", "n1", "n2"}     # the mutant's sig is not re-added as ff
    assert by_sig["n1"]["prov"] == "ff" and by_sig["n1"]["rank"] >= 1000
    assert by_sig["sm"]["prov"] == "mut" and by_sig["sm"]["mutant"]
    assert rec["weight"] == 3.0 and bl.has_both(rec)
    for c in rec["cands"]:
        assert not {"src", "ff_n", "ff_sig_n", "gen", "sql_raw", "temp"} & set(c)
    # a shared sig shows the ff text for about SWAP_P of the questions (seeded per question id)
    swapped = 0
    for i in range(200):
        r, _ = bl.mix_list({**Q, "id": f"bird_train_train_{i}"}, glist, ffc, [], seed=0)
        swapped += next(c for c in r["cands"] if c["sig"] == "sr")["sql"] == "F_R"
    assert 70 < swapped < 130


def test_mix_list_budget_and_trim_order():
    gl = {**Q, "weight": 1.0, "cands": [g(0, "R", "sr", True)] + [g(k, f"W{k}", f"s{k}") for k in range(1, 12)]
          + [{**g(100 + k, f"M{k}", f"m{k}"), "mutant": True} for k in range(2)]}
    smp = {"qwen": [ff(k, f"F{k}", f"n{k}") for k in range(6)]}
    ffc, _ = bl.ff_classes(smp, set(), canon)
    arith = [{"rank": 200, "sql": "AW", "runs": True, "nonnull": True, "ok": False, "sig": "aw"},
             {"rank": 205, "sql": "AR", "runs": True, "nonnull": True, "ok": True, "sig": "ar"}]
    rec, st = bl.mix_list(Q, gl, ffc, arith, seed=0)
    real = [c for c in rec["cands"] if not c.get("mutant")]
    assert len(real) == bl.CAP and len(rec["cands"]) == bl.CAP + bl.MAX_MUT == 14
    assert sum(c["prov"] == "ff" for c in rec["cands"]) == bl.FF_MAX
    assert [c["sql"] for c in rec["cands"] if c["prov"] == "arith"] == ["AR"]   # right arithmetic mutant first
    kept_g = {c["sql"] for c in real if c["prov"] == "g"}
    assert "R" in kept_g and kept_g == {"R"} | {f"W{k}" for k in range(1, 7)}  # worst-ranked grammar wrongs trimmed
    assert st["trimmed_g"] == 5


def test_rescue_and_ff_only_lists():
    # grammar list with no live right class: an ff right class rescues it
    gl = {**Q, "weight": 3.0, "cands": [g(0, "W0", "s0"), g(1, "W1", "s1")]}
    ffc, _ = bl.ff_classes({"qwen": [ff(0, "F_R", "sr", True), ff(1, "F_W", "s9")]}, set(), canon)
    rec, _ = bl.mix_list(Q, gl, ffc, [], seed=0)
    assert bl.has_both(rec) and next(c for c in rec["cands"] if c["ok"])["prov"] == "ff"
    # no grammar list: ff-only list, right and wrong guaranteed when available, at most FF_ONLY_MAX classes
    smp = {"qwen": [ff(k, f"W{k}", f"w{k}") for k in range(10)] * 2 + [ff(50, "R", "sr", True)]}
    ffc, _ = bl.ff_classes(smp, set(), canon)
    rec, _ = bl.mix_list({**Q, "covered": False}, None, ffc, [], seed=0)
    assert len(rec["cands"]) == bl.FF_ONLY_MAX and bl.has_both(rec)
    assert rec["covered"] is False and rec["in_slots"] is False and rec["weight"] == 1.0


def test_ensure_mix():
    R, W = {"ok": True}, {"ok": False}
    ranked = [("a", W), ("b", W), ("c", W), ("d", R)]
    assert [s for s, _ in bl.ensure_mix(ranked, ranked[:3])] == ["a", "b", "d"]
    assert bl.ensure_mix(ranked[:3], ranked[:3]) == ranked[:3]   # no right available: unchanged


def test_ff_extend_end_to_end(tmp_path):
    def wl(p, rows):
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p
    qs = [{**Q, "id": "bird_train_train_1", "db_id": "train_db", "kind": "list"},
          {**Q, "id": "bird_train_train_2", "db_id": "disney", "kind": "list"},          # a pinned val DB
          {**Q, "id": "bird_train_train_3", "db_id": "train_db", "kind": "uncovered", "covered": False},
          {**Q, "id": "bird_train_train_4", "db_id": "train_db", "kind": "list"},          # gold failed at label time
          {**Q, "id": "bird_train_train_5", "db_id": "train_db", "kind": "list"}]          # no ff samples
    gls = [{**q, "weight": 1.0, "cands": [g(0, "R", "sr", True), g(1, "W", "sw")]} for q in qs if q["kind"] == "list"]

    def labf(i, cands, status="ok"):
        return {"id": f"bird_train_train_{i}", "cands": cands, "ff": {"gold_status": status}}
    lq = [labf(1, [ff(0, "X", "n1")]), labf(2, [ff(0, "Y", "n2")]), labf(3, [ff(0, "U", "u1", True), ff(1, "V", "u2")]),
          labf(4, [], "timeout")]
    lo = [labf(1, [ff(0, "Z", "n3", True)]), labf(3, [ff(0, "GOLD", "u1", True)])]
    a = argparse.Namespace(questions=wl(tmp_path / "q.jsonl", qs), grammar_lists=wl(tmp_path / "g.jsonl", gls),
                           ff_qwen=wl(tmp_path / "lq.jsonl", lq), ff_omni=wl(tmp_path / "lo.jsonl", lo), arith=None,
                           out=tmp_path / "out", seed=0)
    bl.ff_extend(a)
    tr = [json.loads(l) for l in open(tmp_path / "out/train_lists.jsonl")]
    va = [json.loads(l) for l in open(tmp_path / "out/val_lists.jsonl")]
    st = json.loads((tmp_path / "out/build_stats.json").read_text())
    assert [r["id"] for r in tr] == ["bird_train_train_1", "bird_train_train_3"] and [r["id"] for r in va] == ["bird_train_train_2"]
    assert st["build"]["no_ff_samples"] == 1 and st["build"]["gold_not_ok_at_label_time"] == 1
    assert st["build"]["gold_copy_omni"] == 1                     # OmniSQL's verbatim gold dropped
    u = next(r for r in tr if r["id"] == "bird_train_train_3")
    assert {c["sql"] for c in u["cands"]} == {"U", "V"} and u["covered"] is False
    # omni-only right class n3 vs qwen rate 1/3 (u1 right; n1, n2, u2 wrong): omni 1/1 -> its right class dropped
    assert st["omni_cap"]["dropped"] == 1
    assert "n3" not in {c["sig"] for c in tr[0]["cands"]}


def test_default_mode_untouched():
    # the val2 call (no --stage-ff / --ff-extend) keeps its constants and pure functions
    assert bl.CAP == 12 and bl.MAX_MUT == 2 and bl.OUT == ROOT / "data/val2"
    assert bl.VAL_DBS == ("citeseer", "disney", "professional_basketball", "trains")


# ---- ff2_eval helpers --------------------------------------------------------------------------------------------------

def cand(ok, src, s, ffs=0, live=True):
    return {"ok": ok, "live": live, "src": src, "ffs": ffs, "v_new": s, "v_sumz": s}


def test_eval_rules_and_calibration():
    rec = {"db": "d1", "vote_ok": True, "vote_share": 0.6,
           "cands": [cand(False, "g", 2.0), cand(True, "ff", 1.9, ffs=4), cand(False, "ff", 0.0, ffs=1)]}
    assert not fe.ok_of(rec, "A", "new")                       # validator alone: the wrong grammar candidate
    assert fe.ok_of(rec, "B", "new", beta=1.0)                 # vote feature lifts the ff right class
    assert fe.ok_of(rec, "C", "new")
    assert fe.ok_of(rec, "E", "new", beta=0.0, tau=0.5)        # margin 0.1 < tau with a 60% vote -> the vote
    assert not fe.ok_of(rec, "D", "new")
    cal = fe.pairs_calibration({"q": rec}, ["q"], "new")
    assert cal["ff-g"] == {"acc": 0.0, "pairs": 1} and cal["ff-ff"] == {"acc": 100.0, "pairs": 1}
    assert cal["right_ff/wrong_g"]["pairs"] == 1


def test_eval_lodo_uses_other_dbs_only():
    # db1 needs beta > 0 to be right, db2 is right only with beta = 0: LODO fits on the other DB
    p = {"a": {"db": "db1", "vote_ok": None, "vote_share": 0.0,
               "cands": [cand(False, "g", 1.0), cand(True, "ff", 0.5, ffs=9)]},
         "b": {"db": "db2", "vote_ok": None, "vote_share": 0.0,
               "cands": [cand(True, "g", 1.0), cand(False, "ff", 0.5, ffs=9)]}}
    vec, fits = fe.per_question(p, ["a", "b"], "B", "new")
    assert vec == {"a": False, "b": False} and len(fits) == 2   # each DB gets the other's beta: both wrong
