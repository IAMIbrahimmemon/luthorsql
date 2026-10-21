"""Research problem 08-03, attack 2: one model from two runs of one recipe (a "LoRA soup" of R1 and R2).

  python scripts/analysis/soup_lora.py runs/r1r2/r1 runs/r1r2/r2 runs/soup_r1r2

The LoRA weight update of each run is d_i = (alpha/r) B_i A_i with r=32, alpha=64. The soup is their exact mean,
(d_1 + d_2)/2, written as one rank-64 adapter: A = [A_1; A_2], B = [B_1, B_2]. Backbone.from_pretrained always uses
alpha=64, so at r=64 the scale is 1 and B A = B_1 A_1 + B_2 A_2 = (d_1 + d_2)/2. Slot heads are averaged parameter by
parameter (both runs start from the same seed-0 head init). args.json gets lora_r=64; vocab.json must be identical.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main(r1: Path, r2: Path, out: Path):
    a1, a2 = json.loads((r1 / "args.json").read_text()), json.loads((r2 / "args.json").read_text())
    assert (r1 / "vocab.json").read_bytes() == (r2 / "vocab.json").read_bytes(), "different vocabularies"
    assert int(a1["lora_r"]) == int(a2["lora_r"]) == 32
    w1 = load_file(str(r1 / "best_lora" / "adapter_model.safetensors"))
    w2 = load_file(str(r2 / "best_lora" / "adapter_model.safetensors"))
    assert w1.keys() == w2.keys()
    soup = {}
    for k in w1:
        if "lora_A" in k:
            soup[k] = torch.cat([w1[k], w2[k]], 0).contiguous()
        elif "lora_B" in k:
            soup[k] = torch.cat([w1[k], w2[k]], 1).contiguous()
        else:
            raise ValueError(f"unexpected adapter tensor {k}")
    (out / "best_lora").mkdir(parents=True, exist_ok=True)
    save_file(soup, str(out / "best_lora" / "adapter_model.safetensors"))
    cfg = json.loads((r1 / "best_lora" / "adapter_config.json").read_text())
    cfg["r"] = 64  # lora_alpha stays 64: scale 1, see the module docstring
    (out / "best_lora" / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    h1 = torch.load(r1 / "best_heads.pt", map_location="cpu")
    h2 = torch.load(r2 / "best_heads.pt", map_location="cpu")
    assert h1.keys() == h2.keys()
    torch.save({k: (h1[k].float() + h2[k].float()).div(2).to(h1[k].dtype) if h1[k].is_floating_point() else h1[k]
                for k in h1}, out / "best_heads.pt")
    a = dict(a1, lora_r="64", out=str(out), soup_of=[str(r1), str(r2)])
    (out / "args.json").write_text(json.dumps(a, indent=1))
    shutil.copy(r1 / "vocab.json", out / "vocab.json")
    print(f"soup of {len(w1) // 2} LoRA modules and {len(h1)} head tensors -> {out}")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:4]))
