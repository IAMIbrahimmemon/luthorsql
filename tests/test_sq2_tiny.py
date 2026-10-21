"""Tiny random Qwen3.5 end to end: v2 training steps (with schema augmentation) lower the loss, decoding
returns plans, and a batch of questions gives the same states as one question alone."""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transformers import AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

import luthor.sq2.train as tr  # noqa: E402
from luthor.sq.model import Backbone  # noqa: E402
from luthor.sq2.model import LuthorSQ2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    tok = AutoTokenizer.from_pretrained(ROOT / "runs/sq4b_v1/merged")
    cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_value_heads=4,
                            linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                            linear_conv_kernel_dim=4, full_attention_interval=4)
    torch.manual_seed(0)
    m = LuthorSQ2(Backbone(mq.Qwen3_5TextModel(cfg), tok), d_proj=32)
    cards, exs, gold = tr.load_dir(Path("/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2/spider_train"), limit=200)
    db = exs[0].db_id
    group = [e for e in exs if e.db_id == db][:8]
    tr.AUG_P = 0.5
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    rnd = random.Random(0)
    losses = []
    for _ in range(40):
        loss, _ = tr.train_db(m, cards[db], group, 16000, rnd)
        opt.step()
        opt.zero_grad()
        losses.append(loss)
    print(f"loss {losses[0]:.2f} -> {losses[-1]:.2f}")
    assert losses[-1] < losses[0] / 3
    m.eval()
    tr.AUG_P = 0.0
    plans = tr.predict(m, cards[db], group)
    assert all(p is not None for p in plans)
    with torch.no_grad():
        _, _, a = tr.encode_db(m, cards[db], group)
        _, _, b = tr.encode_db(m, cards[db], group[2:3])
    d = float((a[2].S - b[0].S).abs().max())
    print("batch vs single", d)
    assert d < 1e-4
    print("OK")


if __name__ == "__main__":
    main()
