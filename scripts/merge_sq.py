"""Merge the best LoRA into the full OpenJev checkpoint (the layout the MLX encoder
loads), and put the heads + serving.json next to it.  python scripts/merge_sq.py --run runs/sq4b_v1"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--base", default="AlexWortega/openjev")
ap.add_argument("--subfolder", default=None, help="default: read from the run's args.json")
a = ap.parse_args()
sub = a.subfolder or json.loads((a.run / "args.json").read_text()).get("subfolder", "qwen3.5-4b-nli-v5")
full = AutoModelForSequenceClassification.from_pretrained(a.base, subfolder=sub, dtype=torch.bfloat16)
outer = getattr(full, full.base_model_prefix)
body = getattr(outer, "language_model", outer)
merged = PeftModel.from_pretrained(body, a.run / "best_lora").merge_and_unload()
if hasattr(outer, "language_model"):
    outer.language_model = merged
else:
    setattr(full, full.base_model_prefix, merged)
out = a.run / "merged"
full.save_pretrained(out)
AutoTokenizer.from_pretrained(a.base, subfolder=sub).save_pretrained(out)
for f in ("best_heads.pt", "serving.json", "calibration.json", "best.json", "args.json"):
    if (a.run / f).exists():
        shutil.copy(a.run / f, out / f)
print("merged ->", out)
