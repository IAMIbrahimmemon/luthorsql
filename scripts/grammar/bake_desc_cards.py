"""Copy a plan directory with BIRD-train column descriptions baked into its card JSONs (luthor.sq.bird_desc.apply,
with a mode and a cap), so a run can use them without --bird-desc (r1r2_chain.sh: DESC=1 DESC_ARGS=none). CPU only.

  uv run --offline python scripts/grammar/bake_desc_cards.py --mode cryptic --cap 60 \
      data/plans_v4/train/bird_train data/plans_v4fix/bird_train
  uv run --offline python scripts/grammar/bake_desc_cards.py --mode cryptic --cap 60 data/plans_v4/rare1 data/plans_v4fix/rare1

Copies plans.jsonl / stats.json / manifest.json unchanged (byte for byte) and rewrites every cards/<db>.json with
the descriptions filled (only empty column descriptions; only the 69 BIRD-train databases have CSVs). Writes
bake.json beside them (mode, cap, source, per-db counts, sha256 of the plans file). Public data only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.pop("LUTHOR_BIRD_DESC", None)  # load the cards as they are on disk
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.sq import bird_desc  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402


def card_json(card: SchemaCard, raw: dict) -> dict:
    """The original JSON with each column's description replaced by the card's (all other keys untouched)."""
    out = json.loads(json.dumps(raw))
    by = {(t.name, c.name): c.description for t in card.tables for c in t.columns}
    for t in out["tables"]:
        for c in t["columns"]:
            c["description"] = by.get((t["name"], c["name"]), c.get("description", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", choices=bird_desc.MODES, default="cryptic")
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--desc-dir", default=str(bird_desc.DEFAULT_DIR))
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists():
        sys.exit(f"{dst} exists: remove it first")
    D = bird_desc.load_dir(a.desc_dir)
    (dst / "cards").mkdir(parents=True)
    for f in ("plans.jsonl", "stats.json", "manifest.json"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    counts = {}
    for p in sorted((src / "cards").glob("*.json")):
        raw = json.loads(p.read_text())
        card = SchemaCard.load(p)
        n = bird_desc.apply(card, D, cap=a.cap, mode=a.mode)
        if n:
            counts[card.db_id] = n
        (dst / "cards" / p.name).write_text(json.dumps(card_json(card, raw)))
    sha = hashlib.sha256((dst / "plans.jsonl").read_bytes()).hexdigest()[:16]
    meta = {"source": str(src), "mode": a.mode, "cap": a.cap, "desc_dir": a.desc_dir, "plans_sha256_16": sha,
            "described_columns_by_db": counts, "described_columns": sum(counts.values())}
    (dst / "bake.json").write_text(json.dumps(meta, indent=1))
    print(f"{dst}: {len(list((dst / 'cards').glob('*.json')))} cards, {meta['described_columns']} columns described "
          f"on {len(counts)} DBs (mode {a.mode}, cap {a.cap}); plans sha256 {sha}")


if __name__ == "__main__":
    main()
