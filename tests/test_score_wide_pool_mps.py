"""CPU numerical checks for the cached-schema research scorer; no checkpoint download."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from luthor.sq2.rerank import Reranker

spec = importlib.util.spec_from_file_location('wide_score', Path(__file__).resolve().parents[1]/'scripts/analysis/score_wide_pool_mps.py')
wide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wide)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {'input_ids': [1 + ord(c) % 29 for c in text] or [1]}


def tiny_rr():
    from transformers import Qwen2Config, Qwen2Model
    torch.manual_seed(113)
    cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      attention_dropout=0.0)
    body = Qwen2Model(cfg).eval()
    rr = SimpleNamespace(body=body, device=torch.device('cpu'),
                         backbone=SimpleNamespace(tok=TinyTokenizer()), score=torch.nn.Linear(16, 1).eval())
    return rr


@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_cached_prefix_equals_original_batches(monkeypatch, batch_size):
    import luthor.sq2.rerank as rerank
    import luthor.sq2.serialize as serialize
    schema = lambda *args: SimpleNamespace(text='schema prefix')
    monkeypatch.setattr(rerank, 'schema_text', schema)
    monkeypatch.setattr(serialize, 'schema_text', schema)
    rr = tiny_rr()
    items = [('count?', '', 'SELECT 1'), ('longer question?', 'context', 'SELECT COUNT(*)'),
             ('last?', '', 'SELECT 2'), ('short?', '', 'SELECT 3'), ('uneven?', '', 'SELECT 44')]
    with torch.inference_mode():
        expected = []
        for i in range(0, len(items), batch_size):
            expected.extend(Reranker.scores(rr, None, items[i:i+batch_size], max_tok=48).tolist())
    calls = []
    handle = rr.body.register_forward_pre_hook(lambda model, args, kw: calls.append(kw['input_ids'].shape), with_kwargs=True)
    actual = wide.cached_scores(rr, None, items, None, batch_size, max_tok=48)
    handle.remove()
    assert actual == pytest.approx(expected, abs=1e-6, rel=1e-6)
    assert len(calls) == 1 + (len(items)+batch_size-1)//batch_size
    assert calls[0] == (1, len('schema prefix'))
    # A second call verifies candidate continuations have not leaked state.
    assert wide.cached_scores(rr, None, items, None, batch_size, max_tok=48) == actual


def test_reject_unsupported_cache_and_empty():
    rr = tiny_rr()
    assert wide.cached_scores(rr, None, [], None) == []
    rr.body.config.use_sliding_window = True
    with pytest.raises(ValueError, match='full-attention'):
        wide.cached_scores(rr, None, [('q','','sql')], None)
