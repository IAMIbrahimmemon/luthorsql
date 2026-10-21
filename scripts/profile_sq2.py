"""Time the parts of one LuthorSQ2 training step on the pod."""
import random, sys, time, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_sq2 as T
from luthor.sq.model import Backbone
from luthor.sq2.model import LuthorSQ2
from luthor.sq2 import train as tr
from luthor.sq2.serialize import schema_text, train_tables

if __name__ == "__main__":
    cards, exs, gold = T.load_sets([Path("/root/d/plans/spider_train")], limit=800)
    m = LuthorSQ2(Backbone.from_pretrained("AlexWortega/openjev", "qwen3.5-4b-nli-v5", grad_ckpt=False)).cuda()
    m.backbone.model.to(torch.bfloat16); m.train()
    groups = tr.db_groups(exs, 8, 0)
    rnd = random.Random(0)
    def t(): torch.cuda.synchronize(); return time.time()
    for it in range(6):
        db, es = groups[it]
        torch.cuda.reset_peak_memory_stats(); a = t()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            st = schema_text(cards[db]); prep = m.prepare_schema(st); b = t()
            se = m.encode_schema(prep); c = t()
            preps = [m.prepare_question(e.question, e.cands, e.context) for e in es]
            oix = {o: i for i, o in enumerate(st.option_ids)}
            enc = [m.project(x) for x in m.encode_questions(se, preps, [e.cands for e in es], oix)]; d = t()
            ls = [tr.example_loss(m, x, e.t, oix)[0] for e, x in zip(es, enc)]; loss = torch.stack(ls).mean(); e_ = t()
        pk = torch.cuda.max_memory_allocated() / 2**30; loss.backward(); f = t()
        print(f"schema tok {len(prep[0])} tokenize {b-a:.2f} prefix {c-b:.2f} questions {d-c:.2f} loss {e_-d:.2f} backward {f-e_:.2f} fwd_peak {pk:.1f}GB total_peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
