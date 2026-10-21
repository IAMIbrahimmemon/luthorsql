"""Head-score dumps (src/luthor/sq2/headdump.py, scripts/dump_heads.py) on a tiny randomly initialised Qwen2 (OmniSQL's
architecture) with the R1/R2 heads (column mentions, value-side input) and a small grammar-v3 template vocabulary.

The critical check: replaying a dump through the unchanged decoders gives the model's own greedy plan, flip pool
and lazy k-best lists back exactly (same plans, order, costs, SQL). CPU, a few seconds."""
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import luthor.sq2.targets as T  # noqa: E402
import luthor.sq2.train as tr  # noqa: E402
from luthor.sq.model import Backbone  # noqa: E402
from luthor.sq.plan import compile_plan  # noqa: E402
from luthor.sq2 import headdump as H  # noqa: E402
from luthor.sq2.model import LuthorSQ2, TorchScores  # noqa: E402
from luthor.sq2.serialize import SLOTS  # noqa: E402
from test_sq2_r1_levers import _card, _rows, _tok_dir  # noqa: E402

TEMPLATES = ["{A} * {N0}", "CAST({A} AS REAL) / {B}", "SUBSTR({A}, 1, {N0})", "CASE WHEN {A} = {V0} THEN 1 ELSE 0 END",
             "(CAST({P0} AS REAL) / {P1}) * {N0}"]
NUMS = [1, 4, 100]
KS = (8, 16)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Grammar v3 vocabulary (restored afterwards: it is module state), tiny model, examples, a dump on disk."""
    saved = {n: copy.deepcopy(getattr(T, n)) for n in ("VOCAB", "PAIRS", "MAIN_CHOICES", "CHOICES")}
    T.set_vocab(TEMPLATES, NUMS)
    try:
        from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
        tok = AutoTokenizer.from_pretrained(_tok_dir())
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2)
        torch.manual_seed(0)
        m = LuthorSQ2(Backbone(Qwen2Model(cfg), tok), d_proj=32, col_mentions=True, value_side=True).eval()
        with torch.no_grad():  # a random model writes absurd plans; lean each head to its first (plain) class so
            # most plans compile and k-best has SQL to walk, while the random rest keeps the lists varied
            for k, lin in m.choice.items():
                lin.bias[1 if k.endswith("n_filter") else 0] += 4.0
            for h in ("sub", "or", "part", "fn"):
                m.pair[h][-1].bias[0] += 3.0
        card = _card()
        rows = _rows() + [{"id": f"u{i}", "db_id": "shop", "question": q, "context": c, "outcome": "uncovered"}
                          for i, (q, c) in enumerate([
                              ("list the names of customers from Kenya", "Kenya refers to country = 'Kenya'"),
                              ("how many orders were cancelled", ""),
                              ("average order amount per country", ""),
                              ("which codes have flag3 c3v2 or c3v5", "")])]
        exs = [tr.make_example(r, card, random.Random(0), strict=False, order="v2") for r in rows]
        gold = {r["id"]: r for r in rows}
        dumps = {}
        for dt in ("float16", "float32"):
            out = tmp_path_factory.mktemp(f"dump_{dt}")
            w = H.DumpWriter(out, {"checkpoint": H.checkpoint_info(None, m), "git": H.git_rev(ROOT),
                                   "heads": H.head_spec(m), "lists": {"flips_k": 8, "kbest_ks": list(KS)}},
                             H.head_weights(m), dt, shard_size=3)
            import dump_heads
            n = dump_heads.dump_group(m, card, exs, w, gold, 8, KS, bs=4)
            assert n == len(exs)
            w.close()
            dumps[dt] = out
        yield {"m": m, "card": card, "exs": exs, "dumps": dumps}
    finally:
        for n, v in saved.items():
            getattr(T, n).clear()
            getattr(T, n).update(v)


def _own_lists(s):
    """The model's own outputs, straight from TorchScores (no recording wrapper), as rerank_sq2 gen computes them."""
    m, card, exs = s["m"], s["card"], s["exs"]
    g = card.graph()
    fn = lambda p: compile_plan(p, g)  # noqa: E731
    out = {}
    with torch.no_grad():
        for tabs, chunk in tr.serve_chunks(card, exs, 4):
            st, oix, enc = tr.encode_db(m, card, chunk, tabs)
            for e, x in zip(chunk, enc):
                out[e.id] = (H.run_lists(TorchScores(m, x, st.option_ids, e.cands), fn, 8, KS), x, st.option_ids, e)
    return out


def test_replay_reproduces_greedy_flips_and_kbest(setup):
    own = _own_lists(setup)
    d = H.HeadDump(setup["dumps"]["float16"])
    assert d.ids == [e.id for e in setup["exs"]]
    longest = 0
    for qd in d:
        ref = own[qd.id][0]
        assert H.same_lists(qd.lists, ref) == {"greedy": True, "flips": True, "kbest8": True, "kbest16": True}
        lists, ds = H.redecode(qd, strict=True)  # strict: every vector served from the box's record
        assert H.same_lists(lists, ref) == {"greedy": True, "flips": True, "kbest8": True, "kbest16": True}, qd.id
        assert ds.hits > 0 and ds.dense == 0 and ds.recomputed == 0
        assert [p["repr"] for p in lists["kbest8"]] == [p["repr"] for p in ref["kbest8"]]  # same order
        assert [p["cost"] for p in lists["kbest8"]] == [p["cost"] for p in ref["kbest8"]]
        assert repr(qd.plans("greedy")[0][0]) == ref["greedy"]["repr"]  # plans round-trip through to_dict
        longest = max(longest, len(ref["kbest16"]))
    assert longest > 8  # the lists are non-trivial


def test_dense_and_recomputed_scores_match_the_model(setup):
    own = _own_lists(setup)
    m = setup["m"]
    for dt, atol in (("float32", 1e-4), ("float16", 5e-2)):
        d = H.HeadDump(setup["dumps"][dt])
        for qd in d:
            _, x, oids, e = own[qd.id]
            sc = TorchScores(m, x, oids, e.cands)
            ds = qd.scores(recorded=False)
            for s in SLOTS:  # dense pointer and choice scores are exact whatever the state dtype
                assert ds.ptr(s) == sc.ptr(s)
            for k in H.choice_keys(m):
                assert ds.choice(k) == sc.choice(k)
            rnd = random.Random(qd.id)
            heads = list(m.pair.keys())
            for _ in range(25):
                s, o, h = rnd.choice(SLOTS), rnd.randrange(len(oids)), rnd.choice(heads)
                np.testing.assert_allclose(ds.pair(s, o, h), sc.pair(s, o, h), atol=atol, rtol=0)
                for slot in (s, s + "|t0"):
                    np.testing.assert_allclose(ds.vals(slot, o), sc.vals(slot, o), atol=atol, rtol=0)
            np.testing.assert_allclose(ds.limit("m.limit"), sc.limit("m.limit"), atol=atol, rtol=0)
            assert ds.recomputed > 0
            if dt == "float32":  # fp32 states: even without the record the decoders give the same lists
                lists, _ = H.redecode(qd, recorded=False)
                assert H.same_lists(lists, qd.lists, cost_tol=1e-5) == {"greedy": True, "flips": True, "kbest8": True,
                                                         "kbest16": True}


def test_format_manifest_and_compactness(setup):
    p = setup["dumps"]["float16"]
    man = json.loads((p / "manifest.json").read_text())
    assert man["format"] == H.FORMAT and man["n_questions"] == len(setup["exs"]) and man["state_dtype"] == "float16"
    assert man["heads"]["slots"] == list(SLOTS) and man["heads"]["grammar"] in ("v3", "v4")
    assert len(man["checkpoint"]["heads_tensor_sha256"]) == 64 and "rev" in man["git"]
    assert man["n_shards"] >= 2 and (p / "cards" / "shop.json").exists()  # shard_size=3 over 7 questions
    d = H.HeadDump(p)
    qd = d.question(d.ids[0])
    assert qd.a["q"].dtype == np.float16 and qd.a["ptr"].dtype == np.float32 and qd.a["rec"].dtype == np.float32
    assert qd.ptr.shape == (len(SLOTS), len(qd.option_ids))
    # option keys are stored once per schema group; later questions keep only the rows a mention changed
    for q2 in list(d)[1:]:
        assert len(q2.a["k_idx"]) < len(q2.option_ids)
    # the decisions trace (RL's format) starts at the projection count
    tr_ = H.decisions(qd)
    assert tr_ and tr_[0][0] == ("c", "m.n_proj")
    # strict replay refuses a vector the box never computed, even inside topk's try/except
    key = next(k for k in qd.recorded if k[0] == "q")
    del qd.recorded[key]
    with pytest.raises(H.MissingScore):
        H.run_lists(qd.scores(strict=True), qd.compile_fn(), 8, KS)


def test_verify_cli(setup, capsys):
    import dump_heads
    out = dump_heads.main(["verify", "--dump", str(setup["dumps"]["float16"])])
    assert out["mode"] == "replay" and all(v == 1.0 for v in out["identical"].values())
    out = dump_heads.main(["verify", "--dump", str(setup["dumps"]["float16"]), "--no-recorded"])
    assert out["sources"]["recomputed"] > 0 and out["sources"]["hits"] == 0
    dump_heads.main(["info", "--dump", str(setup["dumps"]["float16"])])
    assert '"n_questions": 7' in capsys.readouterr().out
