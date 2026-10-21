"""Exact weight average ("model soup") of single-LoRA System 1 runs that share a base model, as ONE LoRA adapter.

Each run i adds delta_i = s_i * B_i @ A_i to every adapted weight (s_i = lora_alpha / r_i). The average of the
fine-tuned models is base + mean_i(delta_i), which is exactly a LoRA of rank sum(r_i):

    A = [A_1; A_2; ...]  (stacked rows),   B = [c_1 B_1, c_2 B_2, ...]  (stacked columns),   c_i = s_i / (n * s)

where s = lora_alpha_out / r_out is the scale the loader applies (Backbone.from_pretrained uses lora_alpha = 64).
Nothing is approximated: the result is still one adapter, with twice the rank for two runs. The heads (slot heads,
rerank head) are linear layers on top and are averaged elementwise. Output dir = a single-LoRA run dir
(rerank_sq2.py gen --run / score --rr, predict.py --run/--rr).

  python scripts/soup_lora.py --runs runs/phase1/single_lora_s/single/run runs/phase1/single_lora2/single/run \\
      --out data/single_soup/run
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LOADER_ALPHA = 64  # luthor.sq.model.Backbone.from_pretrained default; load_model / load_validator never override it


def soup_lora(states: list[dict], scales: list[float], out_scale: float) -> dict:
    """states: each run's adapter tensors (lora_A: (r, d_in), lora_B: (d_out, r)) -> stacked rank-sum(r) tensors."""
    keys = sorted(states[0])
    if any(sorted(s) != keys for s in states[1:]):
        raise SystemExit("runs adapt different modules")
    n, out = len(states), {}
    for k in keys:
        if ".lora_A." in k:
            out[k] = torch.cat([s[k] for s in states], dim=0)
        elif ".lora_B." in k:
            out[k] = torch.cat([s[k].float() * (sc / (n * out_scale)) for s, sc in zip(states, scales)],
                               dim=1).to(states[0][k].dtype)
        else:
            raise SystemExit(f"unexpected adapter tensor {k}")
    return out


def average_heads(paths: list[Path]) -> dict:
    sds = [torch.load(p, map_location="cpu") for p in paths]
    if any(sorted(sd) != sorted(sds[0]) or any(sd[k].shape != sds[0][k].shape for k in sd) for sd in sds[1:]):
        raise SystemExit(f"heads differ in keys or shapes: {paths}")
    return {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) for k in sds[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if len(a.runs) < 2:
        raise SystemExit("need at least two runs")
    cfgs = [json.loads((r / "best_lora" / "adapter_config.json").read_text()) for r in a.runs]
    if len({c["base_model_name_or_path"] for c in cfgs}) != 1 or any(c.get("use_rslora") for c in cfgs):
        raise SystemExit("runs must share the base model and use plain LoRA scaling")
    vocabs = [(r / "vocab.json").read_text() if (r / "vocab.json").exists() else None for r in a.runs]
    if len(set(vocabs)) != 1:
        raise SystemExit("runs have different (or missing) template vocabularies")
    r_out = sum(int(c["r"]) for c in cfgs)
    scales = [c["lora_alpha"] / c["r"] for c in cfgs]
    out_scale = LOADER_ALPHA / r_out
    lora = soup_lora([load_file(str(r / "best_lora" / "adapter_model.safetensors")) for r in a.runs], scales, out_scale)
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "best_lora").mkdir(parents=True)
    save_file(lora, str(a.out / "best_lora" / "adapter_model.safetensors"))
    (a.out / "best_lora" / "adapter_config.json").write_text(json.dumps({**cfgs[0], "r": r_out,
                                                                          "lora_alpha": LOADER_ALPHA}, indent=1))
    torch.save(average_heads([r / "best_heads.pt" for r in a.runs]), a.out / "best_heads.pt")
    torch.save(average_heads([r / "rerank_head.pt" for r in a.runs]), a.out / "rerank_head.pt")
    ra = json.loads((a.runs[-1] / "args.json").read_text())
    (a.out / "args.json").write_text(json.dumps({**ra, "lora_r": str(r_out), "out": str(a.out),
                                                 "soup_of": [str(r) for r in a.runs]}, indent=1))
    if vocabs[0] is not None:
        (a.out / "vocab.json").write_text(vocabs[0])
    st = json.loads((a.runs[-1] / "settings.json").read_text())
    (a.out / "settings.json").write_text(json.dumps({**st, "trainer": "soup_lora", "shared_adapter": True,
                                                     "lora_r": r_out, "soup_of": [str(r) for r in a.runs]}, indent=1))
    print(f"SOUP {len(a.runs)} runs -> {a.out}: one LoRA, rank {r_out}, {len(lora)} tensors; heads averaged")


if __name__ == "__main__":
    main()
