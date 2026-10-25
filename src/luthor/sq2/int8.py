"""8-bit weight-only quantization of the backbone's linear layers, for small machines (e.g. 24 GB of unified memory).

Each base linear layer under the LoRA wrappers keeps int8 weights with one bf16 scale per output channel (symmetric,
absmax) and multiplies with torch._weight_int8pack_mm; LoRA weights, embeddings, norms and the slot and rerank heads
stay in bf16. The 7B backbone drops from about 15 GB to about 8 GB. Quantizing layer by layer and moving each one to
the device right away keeps the peak low while loading.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Int8Linear(nn.Module):
    def __init__(self, lin: nn.Linear, device):
        super().__init__()
        w = lin.weight.data.float()
        scale = (w.abs().amax(1).clamp(min=1e-8) / 127.0)
        self.register_buffer("qweight", (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8).to(device))
        self.register_buffer("scale", scale.to(torch.bfloat16).to(device))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.to(torch.bfloat16).to(device),
                                                               requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    @property
    def weight(self):  # for code that only reads dtype/device/shape
        return self.qweight

    def forward(self, x):
        shape = x.shape
        y = torch._weight_int8pack_mm(x.reshape(-1, shape[-1]).to(torch.bfloat16), self.qweight, self.scale)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)


def quantize_backbone_(body: nn.Module, device) -> int:
    """Replace every LoRA-wrapped base linear layer of a PEFT model in place. Returns the number replaced."""
    n = 0
    for mod in list(body.modules()):
        base = getattr(mod, "base_layer", None)
        if isinstance(base, nn.Linear):
            mod.base_layer = Int8Linear(base, device)
            del base
            n += 1
    return n
