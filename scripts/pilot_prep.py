"""Pilot C inputs: choose 15 held-out Spider-train databases and report loadable example counts."""
import json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sq2 as T

if __name__ == "__main__":
    TR = Path("/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2")
    t = time.time()
    cards, exs, gold = T.load_sets([TR / "spider_train", TR / "spider_syn_train"])
    print(len(exs), f"{time.time() - t:.0f}s")
    dbs = sorted({e.db_id for e in exs})
    random.Random(7).shuffle(dbs)
    dev = dbs[:15]
    Path("data/pilot").mkdir(parents=True, exist_ok=True)
    json.dump(dev, open("data/pilot/dev_dbs.json", "w"))
    print("dev dbs", dev, sum(e.db_id in dev for e in exs))
    gc, gx, gg = T.load_sets([Path("data/eval_plans_v2/spider_dev")], strict=False)
    print("gate", len(gx), "outside slots", sum(e.t is None for e in gx))
