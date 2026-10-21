"""Pick the held-out TRAINING databases used for checkpoint selection and threshold
calibration, and copy their SQLite files next to the mix (the pod needs them to execute
plans). Sealed eval databases are never candidates (build_mix already excluded them).

  python scripts/pick_dev.py --mix data/mix
"""
import argparse
import json
import random
import shutil
from pathlib import Path

SOURCES = {  # where each training source's SQLite files live on the Mac
    "bird_train": ("data/raw/bird_train/train_databases", 1.0),   # the 3 databases kept while streaming
    "spider_train": ("data/raw/spider_data/database", 0.10),
    "synsql_train": ("data/raw/synsql/databases", 0.03),
}

ap = argparse.ArgumentParser()
ap.add_argument("--mix", type=Path, default=Path("data/mix"))
a = ap.parse_args()
rnd = random.Random(0)
mix_dbs: dict[str, set] = {}
for line in open(a.mix / "plans.jsonl"):
    r = json.loads(line)
    mix_dbs.setdefault(r["source"] + "_" + r["split"], set()).add(r["db_id"])
out = a.mix / "dev_sqlite"
out.mkdir(exist_ok=True)
dev = []
for src, (root, frac) in SOURCES.items():
    have = sorted(db for db in mix_dbs.get(src, ()) if (Path(root) / db / f"{db}.sqlite").exists())
    rnd.shuffle(have)
    pick = have[: max(1, int(len(have) * frac))] if have else []
    for db in pick:
        shutil.copy(Path(root) / db / f"{db}.sqlite", out / f"{db}.sqlite")
    dev += pick
    print(f"{src}: {len(pick)} dev databases (of {len(have)} with local SQLite)")
(a.mix / "dev_dbs.json").write_text(json.dumps(sorted(dev)))
size = sum(p.stat().st_size for p in out.glob("*.sqlite")) / 2**20
print(f"{len(dev)} dev databases, {size:.0f} MB of SQLite in {out}")
