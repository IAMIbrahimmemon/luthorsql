"""Merge a SQL-writer LoRA into its base and save a plain checkpoint (for vLLM and the next training stage).

  python scripts/sqlw_merge.py --base Qwen/Qwen2.5-Coder-7B-Instruct --lora runs/sqlw_sft/lora --out /workspace/m/sqlw_sft
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, a.lora).merge_and_unload()
m.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
print("MERGED", a.out)
