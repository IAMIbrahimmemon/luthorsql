"""Tiny random Qwen2 (OmniSQL's architecture) on grammar v3 plans: training steps lower the loss on
questions that use expression templates, and decoding returns plans that compile."""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model  # noqa: E402

import luthor.sq2.train as tr  # noqa: E402
from luthor.sq.model import Backbone  # noqa: E402
from luthor.sq.plan import compile_plan  # noqa: E402
from luthor.sq2.model import LuthorSQ2  # noqa: E402
from luthor.sq2.targets import set_vocab  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    v = ROOT / "data/plans_v3/templates.json"
    set_vocab(json.loads(v.read_text()), json.loads(v.with_name("templates_nums.json").read_text()))
    tok = AutoTokenizer.from_pretrained(ROOT / "runs/sq4b_v1/merged")
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(0)
    m = LuthorSQ2(Backbone(Qwen2Model(cfg), tok), d_proj=32, joint_layers=int(sys.argv[1]) if len(sys.argv) > 1 else 0,
                  col_mentions=len(sys.argv) > 2 and sys.argv[2] == "1",
                  mention_weight=sys.argv[3] if len(sys.argv) > 3 else "flat")
    if len(sys.argv) > 4:  # schema renaming + description dropout shares
        tr.AUG_P, tr.DESC_DROP = float(sys.argv[4]), float(sys.argv[4])
        m.mention_drop = float(sys.argv[4])
    cards, exs, gold = tr.load_dir(ROOT / "data/plans_v3/train/bird_train", limit=3000)
    tmpl = [e for e in exs if any(h == "fn" and c > 0 for _, _, h, c in e.t.pair)]
    print("examples", len(exs), "with a template", len(tmpl))
    db = tmpl[0].db_id
    group = ([e for e in tmpl if e.db_id == db] + [e for e in exs if e.db_id == db])[:8]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    rnd = random.Random(0)
    losses = []
    for _ in range(60):
        loss, _ = tr.train_db(m, cards[db], group, 16000, rnd)
        opt.step()
        opt.zero_grad()
        losses.append(loss)
    print(f"loss {losses[0]:.2f} -> {losses[-1]:.2f}")
    assert losses[-1] < losses[0] / 3
    m.eval()
    plans = tr.predict(m, cards[db], group)
    sqls = [compile_plan(p, cards[db].graph()) for p in plans if p is not None]
    print(len(sqls), "compiled;", sqls[0][:200])
    assert len(sqls) == len(group)
    from luthor.sq2.model import TorchScores
    from luthor.sq2.topk import candidates
    with torch.no_grad():
        st, oix, enc = tr.encode_db(m, cards[db], group[:4])
        for e, x in zip(group[:4], enc):
            sc = TorchScores(m, x, st.option_ids, e.cands)
            g = cards[db].graph()
            def comp(p):
                try:
                    return compile_plan(p, g)
                except Exception:
                    return None
            old = {comp(p) for p in candidates(sc, k=16)} - {None}
            new = [comp(p) for p, _ in candidates(sc, k=16, with_cost=True, distinct=True, compile_fn=lambda p: compile_plan(p, g))]
            from luthor.sq2.topk import kbest
            kb = [comp(p) for p in kbest(sc, 16, lambda p: compile_plan(p, g))]
            print(f"distinct SQL in top 16: default {len(old)}, distinct mode {len(set(new))} of {len(new)}, "
                  f"k-best alone {len(set(kb) - {None})}")
            assert len(set(new)) >= len(old)
    print("OK")


if __name__ == "__main__":
    main()
