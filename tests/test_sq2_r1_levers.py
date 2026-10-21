"""Round-1 levers for the System-1 slot model, on a tiny randomly initialised Qwen2 (OmniSQL's architecture):
train/serve candidate parity (--cand-order v2), value-side input (--value-side), the flags' args.json round
trip, and old checkpoints (no flags) unchanged. CPU, a few seconds."""
import argparse
import glob
import json
import random
import sqlite3
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import luthor.sq2.train as tr  # noqa: E402
from luthor.sq.candidates import ValueIndex, find_candidates  # noqa: E402
from luthor.sq.model import Backbone  # noqa: E402
from luthor.sq.plan import Col, Filter, Plan, Proj  # noqa: E402
from luthor.sq.schema_card import ColumnCard, SchemaCard, TableCard  # noqa: E402
from luthor.sq2.model import LuthorSQ2  # noqa: E402
from luthor.sq2.serialize import MAX_CANDS, question_text  # noqa: E402


def _tok_dir():
    for p in glob.glob(str(Path.home() / ".cache/huggingface/hub/models--seeklhy--OmniSQL-7B/snapshots/*")) + \
            [str(ROOT / "runs/sq4b_v1/merged")]:
        if Path(p, "tokenizer.json").exists():
            return p
    pytest.skip("no local tokenizer")


def _tiny(value_side=False, col_mentions=True, seed=0):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
    tok = AutoTokenizer.from_pretrained(_tok_dir())
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(seed)
    return LuthorSQ2(Backbone(Qwen2Model(cfg), tok), d_proj=32, col_mentions=col_mentions, value_side=value_side)


def _card(n_code_cols=8):
    """A shop, plus a `codes` table of low-cardinality columns: 8 x 12 enumerated values push the list past 64."""
    codes = [ColumnCard("id", "int", pk=True)] + [ColumnCard(f"flag{k}", "text") for k in range(n_code_cols)]
    vals = {"customers.country": ["Egypt", "Kenya"], "orders.status": ["shipped", "cancelled"]}
    for k in range(n_code_cols):
        vals[f"codes.flag{k}"] = [f"c{k}v{j}" for j in range(12)]
    return SchemaCard("shop", "sqlite", [
        TableCard("customers", "people who buy", [ColumnCard("id", "int", pk=True), ColumnCard("name", "text"),
                                                  ColumnCard("country", "text", "country of residence")]),
        TableCard("orders", "purchases", [ColumnCard("id", "int", pk=True), ColumnCard("customer_id", "int"),
                                          ColumnCard("status", "text", "shipped or cancelled"),
                                          ColumnCard("amount", "real")]),
        TableCard("codes", "", codes)], [("orders", "customer_id", "customers", "id")], values=vals)


def _row(i, q, plan, **kw):
    return {"id": i, "db_id": "shop", "question": q, "context": "", "outcome": "plan", "plan": plan.to_dict(), **kw}


def _rows():
    egypt = Plan([Proj("sum", Col("orders", "amount"))], filters=[Filter(Col("customers", "country"), "=", ["Egypt"])])
    late = Plan([Proj("count", Col("codes", "id"))], filters=[Filter(Col("codes", "flag7"), "=", ["c7v11"])])
    many = Plan([Proj("sum", Col("orders", "amount"))], filters=[Filter(Col("orders", "status"), "=", ["shipped"])])
    return [_row("a", "total amount of orders from customers in Egypt", egypt),
            _row("b", "how many codes have the last code", late),  # gold c7v11: enumerated, unlinked, past the cap
            _row("c", "total amount of shipped orders with order status in 2023 and 2024", many)]


def _old_select(cands, need=frozenset(), cap=MAX_CANDS, rnd=None):
    """select_cands as it was before --cand-order (round 6), for the old-behaviour check."""
    ment = [i for i, c in enumerate(cands) if c.span is not None]
    enum = [i for i, c in enumerate(cands) if c.span is None]
    if rnd is not None:
        rnd.shuffle(enum)
    keep = list(dict.fromkeys(sorted(need) + ment + enum))[:max(cap, len(need))]
    keep.sort()
    return [cands[i] for i in keep], {old: new for new, old in enumerate(keep)}


# ---- candidate parity --------------------------------------------------------------------------------------

def test_cand_parity_train_equals_serve():
    card = _card()
    for r in _rows():
        body = tr.question_body(r)
        serve, _ = tr.select_cands(find_candidates(body, card), order="v2", body=body)
        for seed in (0, 1, 2):  # training randomness must not change the list
            e = tr.make_example(r, card, random.Random(seed), order="v2")
            assert [(c.kind, c.value, c.col, c.span) for c in e.cands] == [(c.kind, c.value, c.col, c.span) for c in serve]
        assert len(serve) <= MAX_CANDS


def test_cand_v2_drops_cut_target_and_v1_force_keeps():
    card = _card()
    r = _rows()[1]
    assert len(find_candidates(tr.question_body(r), card)) > MAX_CANDS
    e1 = tr.make_example(r, card, random.Random(0), order="v1")
    e2 = tr.make_example(r, card, random.Random(0), order="v2")
    assert e1.t.vals and "c7v11" in {c.value for c in e1.cands}  # v1: the gold literal is force-kept
    serve, _ = tr.select_cands(find_candidates(tr.question_body(r), card), order="v1", body=tr.question_body(r))
    assert "c7v11" not in {c.value for c in serve}  # ... though serving cuts it (the train/serve mismatch)
    assert "c7v11" not in {c.value for c in e2.cands} and len(e2.cands) == MAX_CANDS
    assert e2.t is not None and not e2.t.vals  # v2: example kept, its value target dropped
    assert e2.t.ptr == e1.t.ptr and e2.t.sets == e1.t.sets  # structure targets unchanged
    ok = tr.make_example(_rows()[0], card, random.Random(0), order="v2")  # gold mentioned: kept, target intact
    assert ok.t.vals and ok.cands[ok.t.vals[0][2][0]].value == "Egypt"


def test_cand_v2_ranks_linked_enumerations_first():
    card = _card()
    body = "list the codes whose flag3 is set"
    cands = find_candidates(body, card)
    kept, _ = tr.select_cands(cands, order="v2", body=body)
    assert {f"c3v{j}" for j in range(12)} <= {c.value for c in kept}  # the linked column's values survive the cap


def test_cand_parity_real_bird_train():
    """BIRD-train rows of a held-out training database: the training list (stored lookup hits, twins rebuilt)
    equals the serving list (ValueIndex.match on the live database)."""
    d, db = ROOT / "data/sys1/bird_train", "movie"
    dbp = ROOT / f"data/sys1/dbs/{db}/{db}.sqlite"
    if not (d / "plans.jsonl").exists() or not dbp.exists():
        pytest.skip("no BIRD-train plans")
    import train_sq2
    card = SchemaCard.load(d / "cards" / f"{db}.json")
    con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    vix = ValueIndex.build(con, card)
    lines = [l for l in open(d / "plans.jsonl") if f'"db_id": "{db}"' in l][:300]
    n = twins = 0
    for r in train_sq2._with_bases(lines):
        r = json.loads(r) if isinstance(r, str) else r
        if r.get("outcome") != "plan":
            continue
        e = tr.make_example(r, card, random.Random(0), order="v2")
        if e is None:
            continue
        body = tr.question_body(r)
        serve, _ = tr.select_cands(find_candidates(body, card, extra=vix.match(body)), order="v2", body=body)
        assert [(c.value, c.col, c.span) for c in e.cands] == [(c.value, c.col, c.span) for c in serve], r["id"]
        n += 1
        twins += bool(r.get("twin_of"))
    assert n > 50 and twins > 10


# ---- value-side input ------------------------------------------------------------------------------------

def test_value_side_forward_backward_changes_value_logits():
    card = _card()
    exs = [tr.make_example(r, card, random.Random(0), order="v2") for r in _rows()]
    m = _tiny(value_side=True)
    st, oix, enc = tr.encode_db(m, card, exs)
    loss = sum(tr.example_loss(m, x, e.t, oix)[0] for e, x in zip(exs, enc))
    assert torch.isfinite(loss)
    loss.backward()
    assert m.W_x.weight.grad is not None and float(m.W_x.weight.grad.abs().sum()) > 0
    x, e = enc[0], exs[0]
    slot, opt, _ = e.t.vals[0]
    with torch.no_grad():
        base = m.value_logits(x, slot, oix[opt]).clone()
        lim = m.value_logits(x, "m.limit", None).clone()
        x.X = torch.zeros_like(x.X)  # the same question without the Values-entry states
        m.project(x)
        assert not torch.allclose(base, m.value_logits(x, slot, oix[opt]))
        assert not torch.allclose(lim, m.value_logits(x, "m.limit", None))  # limit uses the new vectors too
    p = m.prepare_question(e.question, e.cands, e.context)
    assert "\nValues: " not in p.text and len(p.val_tok) == len(e.cands)  # mentions read only the question part
    qt = question_text(e.question, e.context, e.cands, max_entries=m.value_entries)
    assert sum(sp is not None for sp in qt.v_spans) == min(len(e.cands), m.value_entries)
    for c, sp in zip(e.cands, qt.v_spans):  # each span is the candidate's value, after the question
        assert sp is None or (qt.text[sp[0]:sp[1]] == str(c.value)[:40] and sp[0] > qt.q_end)
    assert "Egypt" in {c.value for c, sp in zip(e.cands, qt.v_spans) if sp}  # the mentioned value is written
    hidden = [i for i, sp in enumerate(qt.v_spans) if sp is None]
    with torch.no_grad():
        X = tr.encode_db(m, card, exs[:1])[2][0].X
    assert hidden and float(X[hidden].abs().sum()) == 0 and float(X.abs().sum()) > 0  # past the budget: no Values state


def test_value_side_cached_prefix_unchanged():
    """The Values section is in the per-question part: the schema prefix (and its cache) is the same."""
    card = _card()
    m0, m1 = _tiny(False), _tiny(True)
    from luthor.sq2.serialize import schema_text
    st = schema_text(card)
    assert m0.prepare_schema(st)[0] == m1.prepare_schema(st)[0]


# ---- flags round trip / old checkpoints ---------------------------------------------------------------------

def test_args_json_roundtrip(tmp_path):
    import eval_sq2
    import train_sq2
    for order, vs in (("v2", True), ("v1", False)):
        ns = argparse.Namespace(cand_order=order, value_side=vs, joint_layers=0, col_mentions=True, lora_r=32)
        (tmp_path / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(ns).items()}, indent=1))  # as train_sq2
        ra = eval_sq2.run_args(tmp_path)
        m = eval_sq2.build_model(_tiny().backbone, ra)
        assert m.value_side is vs and hasattr(m, "W_x") is vs
        assert eval_sq2.use_cand_order(ra) == order and tr.CAND_ORDER == order and train_sq2.CAND_ORDER == order
    # a run from before the flags (sys1_full's args.json has neither): old behaviour
    old = {"col_mentions": "True", "joint_layers": "0", "lora_r": "32"}
    assert eval_sq2.use_cand_order(old) == "v1" and not eval_sq2.build_model(_tiny().backbone, old).value_side


def test_old_checkpoints_unchanged():
    card = _card()
    rows = _rows()
    for r in rows:  # v1 lists, with and without the training shuffle, are the old ones
        cands = find_candidates(tr.question_body(r), card)
        need = {len(cands) - 1}
        for rnd in (None, 7):
            a = tr.select_cands(cands, need, rnd=random.Random(rnd) if rnd else None, order="v1")
            b = _old_select(cands, need, rnd=random.Random(rnd) if rnd else None)
            assert a[1] == b[1]
    tr.CAND_ORDER = "v1"
    m = _tiny(value_side=False)
    e = tr.make_example(rows[0], card, random.Random(0))
    p = m.prepare_question(e.question, e.cands, e.context)
    assert p.text == "\nQuestion: " + e.question + "\nPlan:" and p.val_tok is None
    assert not any(k.startswith("W_x") for k in m.head_state())
    m2 = _tiny(value_side=False, seed=1)
    m2.load_state_dict(m.state_dict(), strict=True)  # an old checkpoint loads exactly into the no-flag model
    with torch.no_grad():
        _, oix, a = tr.encode_db(m, card, [e])
        _, _, b = tr.encode_db(m2, card, [e])
        slot, opt, _ = e.t.vals[0]
        assert torch.equal(m.value_logits(a[0], slot, oix[opt]), m2.value_logits(b[0], slot, oix[opt]))
        assert a[0].X is None


def test_all_questions_rows(tmp_path):
    """--all-questions: uncovered rows (no plan) and questions with no plan row become t=None examples."""
    import train_sq2
    d = tmp_path / "set"
    (d / "cards").mkdir(parents=True)
    _card().save(d / "cards" / "shop.json")
    rows = _rows()[:1] + [{"id": "bird_dev_1", "db_id": "shop", "question": "who never ordered", "context": "",
                          "outcome": "unsupported", "gold_sql": "SELECT name FROM customers"}]
    (d / "plans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    raw = [{"question_id": i, "db_id": "shop", "question": f"q{i}", "evidence": "", "SQL": "SELECT 1"} for i in range(3)]
    (tmp_path / "dev.json").write_text(json.dumps(raw))
    _, exs, gold = train_sq2.load_sets([d], workers=1, strict=False)
    assert len(exs) == 1
    _, exs, gold = train_sq2.load_sets([d], workers=1, strict=False, all_rows=True, questions=tmp_path / "dev.json")
    assert sorted(e.id for e in exs) == ["a", "bird_dev_0", "bird_dev_1", "bird_dev_2"]
    assert all(e.t is None for e in exs if e.id != "a")
    import eval_sq2
    assert eval_sq2.gold_sql(gold["bird_dev_2"]) == "SELECT 1"
