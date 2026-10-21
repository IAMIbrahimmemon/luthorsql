"""Share one database's schema prefix across many questions (Qwen3.5 hybrid cache).

Qwen3.5 mixes full-attention layers (key/value cache) with linear-attention layers (a conv
state and a recurrent state). After running the schema once, `expand_cache` repeats every
state along the batch dimension so N questions continue from it, with gradients flowing
back into the prefix (the schema encoding trains too).

HF updates the linear-attention recurrent state with an in-place copy_, which breaks
autograd when that state came from the prefix graph; for the question pass we swap in a
functional update (plain assignment). The conv state already updates functionally when
record_past is True.
"""
from __future__ import annotations

import types

import torch


def _rep(t, n: int):
    return t.repeat_interleave(n, dim=0) if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == 1 else t


def _functional_recurrent_update(self, recurrent_states, state_idx: int = 0, **kwargs):
    self.recurrent_states[state_idx] = recurrent_states
    self.is_recurrent_states_initialized[state_idx] = True
    return recurrent_states


def expand_cache(cache, n: int):
    for layer in cache.layers:
        if torch.is_tensor(getattr(layer, "keys", None)):
            layer.keys = _rep(layer.keys, n)
            layer.values = _rep(layer.values, n)
        for attr in ("conv_states", "recurrent_states"):
            d = getattr(layer, attr, None)
            if isinstance(d, dict):
                for k in list(d):
                    d[k] = _rep(d[k], n)
        if hasattr(layer, "update_recurrent_state"):
            layer.record_past = True
            layer.update_recurrent_state = types.MethodType(_functional_recurrent_update, layer)
    return cache


# ---- gradient checkpointing with a shared prefix (full-attention backbones: OmniSQL / Qwen2.5) --------------
# HF drops past_key_values inside a checkpointed layer unless the layer may keep it, because a backward replay
# would run cache.update a second time and append the keys twice. Here the prefix pass writes into a cache that
# overwrites instead of appending (the replay rewrites the same states), and the question pass reads a separate
# cache that never changes (the replay concatenates the same prefix again). Gradients match the plain run
# (tests/test_sq2_ckpt.py); activations saved for backward drop ~24x, which is what lets 14B/32B and the largest
# BIRD schemas fit.

def _base_layers(body):
    from transformers.modeling_layers import GradientCheckpointingLayer
    return [m for m in body.modules() if isinstance(m, GradientCheckpointingLayer)]


def checkpointing(body) -> bool:
    """True when the backbone's decoder layers are gradient-checkpointed (train_sq2.py --grad-ckpt)."""
    return any(getattr(m, "gradient_checkpointing", False) for m in _base_layers(body))


def allow_cache_under_checkpointing(body) -> None:
    """Let checkpointed layers keep past_key_values (safe only with the caches below). Raises for backbones with
    linear-attention or sliding-window layers, whose cache state these classes do not cover."""
    cfg = body.config.get_text_config() if hasattr(body.config, "get_text_config") else body.config
    kinds = set(getattr(cfg, "layer_types", None) or ["full_attention"])
    if getattr(cfg, "use_sliding_window", False) or kinds - {"full_attention"}:
        raise NotImplementedError(f"--grad-ckpt with the schema prefix cache needs full-attention layers only; "
                                  f"this backbone has {sorted(kinds)} (sliding window: "
                                  f"{getattr(cfg, 'use_sliding_window', False)})")
    for m in _base_layers(body):
        m._can_checkpoint_with_cache = True


def _layer_classes():
    from transformers.cache_utils import DynamicLayer

    class OverwriteLayer(DynamicLayer):
        """update() stores this call's states and returns them: a checkpoint replay rewrites, never appends.
        frozen: a read-only prefix; update() returns prefix + new states and stores nothing."""
        frozen = False

        def update(self, key_states, value_states, *args, **kwargs):
            if self.frozen:
                return torch.cat([self.keys, key_states], -2), torch.cat([self.values, value_states], -2)
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            self.keys, self.values = key_states, value_states
            return key_states, value_states

    return OverwriteLayer


def prefix_cache(config):
    """An empty cache for the schema prefix pass under gradient checkpointing."""
    from transformers import DynamicCache
    cache = DynamicCache(config=config)
    cls = _layer_classes()
    cache.layers = [cls() for _ in range(len(cache.layers) or config.num_hidden_layers)]
    return cache


def question_cache(prefix, n: int, config):
    """A read-only copy of the prefix cache repeated n times, for the batched question pass (gradients still flow
    into the prefix states)."""
    from transformers import DynamicCache
    cache = DynamicCache(config=config)
    cls = _layer_classes()
    layers = []
    for lp in prefix.layers:
        lq = cls()
        lq.lazy_initialization(lp.keys, lp.values)
        lq.keys, lq.values = _rep(lp.keys, n), _rep(lp.values, n)
        lq.frozen = True
        layers.append(lq)
    cache.layers = layers
    return cache


def release_cache(cache) -> None:
    """Break the layer -> bound method -> layer cycle expand_cache creates, so the cache (and
    the GPU tensors it holds) is freed by reference counting, not whenever the cyclic GC runs."""
    for layer in cache.layers:
        layer.__dict__.pop("update_recurrent_state", None)
