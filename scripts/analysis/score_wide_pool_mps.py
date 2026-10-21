"""Score research 01-01 held-out wide pools with the frozen R2 validator.

The local 24GB Mac uses explicitly labelled weight-only int8 base weights and
bf16 activations/LoRA; this is a numerical variant, not the published GPU run.
Inputs must be prepared held-out candidates; output is append-only/resumable.
A single table context from the complete pool is fixed across all prefixes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'reports/fanout_r2/zero_gpu')]



def cached_scores(rr, card, items, tables, batch_size=8, max_tok=384):
    """Equivalent to Reranker.scores in batches, encoding fixed schema only once.

    Only full-attention backbones are accepted. Each continuation receives its
    own frozen cache, so appending a candidate cannot mutate the saved prefix.
    """
    import torch
    from luthor.sq2.prefix_cache import question_cache, release_cache
    from luthor.sq2.rerank import candidate_text
    from luthor.sq2.serialize import schema_text
    if not items:
        return []
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    cfg = rr.body.config
    cfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
    if getattr(cfg, 'use_sliding_window', False) or set(getattr(cfg, 'layer_types', None) or ['full_attention']) - {'full_attention'}:
        raise ValueError('Cached scorer requires full-attention layers')
    tok = rr.backbone.tok
    ids = tok(schema_text(card, tables).text, add_special_tokens=False)['input_ids']
    result = []
    with torch.inference_mode():
        prefix = rr.body(input_ids=torch.tensor([ids], device=rr.device), use_cache=True).past_key_values
        try:
            for start in range(0, len(items), batch_size):
                rows = [tok(candidate_text(*it), add_special_tokens=False)['input_ids'][-max_tok:]
                        for it in items[start:start+batch_size]]
                n, width = len(rows), max(map(len, rows))
                x = torch.full((n, width), tok.pad_token_id, dtype=torch.long, device=rr.device)
                for i, row in enumerate(rows):
                    x[i, :len(row)] = torch.tensor(row, device=rr.device)
                continuation = question_cache(prefix, n, cfg)
                pos = torch.arange(len(ids), len(ids)+width, device=rr.device)[None].expand(n, -1)
                try:
                    hidden = rr.body(input_ids=x, past_key_values=continuation, position_ids=pos,
                                     use_cache=True).last_hidden_state
                    last = torch.tensor([len(row)-1 for row in rows], device=rr.device)
                    result.extend(rr.score(hidden[torch.arange(n, device=rr.device), last].float()).squeeze(-1).tolist())
                finally:
                    release_cache(continuation)
                    del continuation
                del hidden
                if rr.device.type == 'mps':
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
        finally:
            release_cache(prefix)
    return result


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from luthor.sq.model import Backbone
    from luthor.sq.schema_card import SchemaCard
    from luthor.sq2.rerank import Reranker
    from rerank_sq2 import tables_for
    from mac_gen import quantize_int8

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-k', type=int, default=32)
    ap.add_argument('--inference-manifest', type=Path, help='Optional JSON object of required run-metadata values')
    ap.add_argument('--parity-reference', type=Path, help='Optional existing held-out flip scores for smoke parity')
    a = ap.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required for this explicitly labelled scorer')
    rows = [json.loads(line) for line in a.input.read_text().splitlines()]
    source_rows = {r['id']:r for r in rows}
    if len(source_rows) != len(rows):
        raise ValueError('Duplicate input question IDs')
    if a.batch_size < 1 or a.max_k < 1 or a.limit < 0:
        raise ValueError('Batch size and max-k must be positive; limit must be nonnegative')
    input_hash = hashlib.sha256(a.input.read_bytes()).hexdigest()
    rr_dir = ROOT/'runs/r1r2/r2/rr'
    head_hash = hashlib.sha256((rr_dir/'rerank_head.pt').read_bytes()).hexdigest()
    adapter_hash = hashlib.sha256((rr_dir/'rerank_lora/rerank/adapter_model.safetensors').read_bytes()).hexdigest()
    allowed = set(json.loads((ROOT/'data/sys1/dev_dbs_bird.json').read_text()))
    if any(r['db_id'] not in allowed or not r['id'].startswith('bird_train_') for r in rows):
        raise ValueError('Only held-out BIRD training databases are accepted')
    done = set()
    manifest_path = a.output.with_suffix(a.output.suffix+'.manifest.json')
    expected = {'input_sha256':input_hash, 'head_sha256':head_hash, 'adapter_sha256':adapter_hash,
                'max_k':a.max_k, 'batch_size':a.batch_size,
                'scorer': 'fixed-context-frozen-schema-cache-v1',
                'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if a.output.exists() and a.output.stat().st_size and not manifest_path.exists():
        raise ValueError('Cannot resume output without verified inference manifest')
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text())
        if any(prior_manifest.get(k) != v for k,v in expected.items()):
            raise ValueError('Resume inference manifest mismatch')
        if a.inference_manifest and any(prior_manifest.get(k) != v for k,v in json.loads(a.inference_manifest.read_text()).items()):
            raise ValueError('Requested inference manifest differs from recorded run')
    if a.output.exists():
        existing = [json.loads(l) for l in a.output.read_text().splitlines()]
        if len({r['id'] for r in existing}) != len(existing):
            raise ValueError('Duplicate output question IDs')
        for old in existing:
            current = source_rows.get(old['id'])
            if current is None or any(old['scoring'].get(k) != v for k,v in expected.items()):
                raise ValueError('Resume provenance mismatch: '+old['id'])
            if [c['sql'] for c in old['cands']] != [c['sql'] for c in current['cands']]:
                raise ValueError('Resume candidate order mismatch: '+old['id'])
        done = {r['id'] for r in existing}
    rows = [r for r in rows if r['id'] not in done]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        print('No remaining rows', flush=True)
        return
    rr_dir = ROOT/'runs/r1r2/r2/rr'
    if json.loads((rr_dir/'settings.json').read_text()).get('result_preview', False):
        raise ValueError('This harness expects the R2 no-preview validator')
    base = 'seeklhy/OmniSQL-7B'
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    body = AutoModel.from_pretrained(base, dtype=torch.bfloat16, device_map='cpu', local_files_only=True)
    body = get_peft_model(body, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules='all-linear'))
    rr = Reranker(Backbone(body, tok), lora_r=16)
    adapter = rr_dir/'rerank_lora/rerank/adapter_model.safetensors'
    adapter_state = load_file(str(adapter))
    adapter_result = set_peft_model_state_dict(body, adapter_state, adapter_name='rerank')
    if adapter_result.unexpected_keys or any('.rerank.' in k for k in adapter_result.missing_keys):
        raise RuntimeError('Reranker adapter state mismatch: '+str(adapter_result))
    state = torch.load(rr_dir/'rerank_head.pt', map_location='cpu', weights_only=True)
    if set(state) != set(rr.head_state()):
        raise RuntimeError('Reranker head state keys mismatch')
    loaded = rr.load_state_dict(state, strict=False)
    if loaded.unexpected_keys or any(not k.startswith('backbone.') for k in loaded.missing_keys):
        raise RuntimeError('Reranker head missing keys')
    body.to(torch.bfloat16)
    count = quantize_int8(body, 'mps')
    rr.to('mps').eval()
    rr.activate(train=False)
    meta = {'base': base, 'base_precision': 'weight-only-int8-per-output-channel',
            'activation_precision': 'bfloat16', 'head_precision': str(rr.score.weight.dtype),
            'device': 'mps', 'adapter_sha256': adapter_hash, 'head_sha256': head_hash, 'input_sha256': input_hash,
            'base_revision': getattr(body.config, '_commit_hash', None),
            'schema_policy': 'fixed-from-full-input-pool-budget9000', 'max_k': a.max_k,
            'quantized_linears': count, 'torch': torch.__version__, 'batch_size': a.batch_size,
            'scorer': 'fixed-context-frozen-schema-cache-v1',
            'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if a.inference_manifest:
        required = json.loads(a.inference_manifest.read_text())
        mismatched = [k for k,v in required.items() if meta.get(k) != v]
        if mismatched:
            raise ValueError('Inference manifest mismatch: '+str(mismatched))
    manifest_path = a.output.with_suffix(a.output.suffix+'.manifest.json')
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != meta:
            raise ValueError('Existing run manifest differs from current inference configuration')
    elif a.output.exists() and a.output.stat().st_size:
        raise ValueError('Cannot resume output without verified inference manifest')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'loaded_seconds': time.monotonic()-start, **meta}), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cards = {}
    refs = {}
    if a.parity_reference:
        if a.parity_reference.resolve() != (ROOT/'runs/r1r2/r2/rr/dev_scored.jsonl').resolve():
            raise ValueError('Parity input must be the existing held-out flip scores')
        refs = {r['id']: r for r in map(json.loads, a.parity_reference.read_text().splitlines())}
        if any(r['db_id'] not in allowed for r in refs.values()):
            raise ValueError('Parity database is not held out training data')
    for n, r in enumerate(rows, 1):
        t = time.monotonic()
        db = r['db_id']
        if db not in cards:
            cards[db] = SchemaCard.load(ROOT/'data/sys1/bird_train/cards'/f'{db}.json')
        card = cards[db]
        tabs = tables_for(card, [c['sql'] for c in r['cands'] if c.get('sql')], 9000)
        # Fix context using full input, then limit scored candidates to requested k.
        positions = [i for i,c in enumerate(r['cands'][:a.max_k]) if c.get('sql')]
        items = [(r['question'], r.get('context',''), r['cands'][i]['sql']) for i in positions]
        scores = cached_scores(rr, card, items, tabs, a.batch_size)
        for i, value in zip(positions, scores):
            r['cands'][i]['rr'] = value
        if r['id'] in refs:
            ref = refs[r['id']]
            unique_ref = list({c['sql']:c for c in reversed(ref['cands']) if c.get('sql') and 'rr' in c}.values())
            ref_tabs = tables_for(card, [c['sql'] for c in unique_ref], 9000)
            measured = []
            with torch.inference_mode():
                for i in range(0, len(unique_ref), a.batch_size):
                    batch = unique_ref[i:i+a.batch_size]
                    measured.extend(rr.scores(card, [(ref['question'],ref.get('context',''),c['sql']) for c in batch], ref_tabs).float().tolist())
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
            r['parity_smoke'] = {'scope': 'same flip SQL, per-question schema; original GPU schema batching may differ',
                                 'saved_scores': [c['rr'] for c in unique_ref], 'mps_scores': measured,
                                 'sqls': [c['sql'] for c in unique_ref]}
        r['scoring'] = {**meta, 'tables': tabs, 'seconds': time.monotonic()-t}
        with a.output.open('a') as out:
            out.write(json.dumps(r)+'\n')
        print(json.dumps({'done':n,'total':len(rows),'id':r['id'],'candidates':len(items),'seconds':r['scoring']['seconds']}), flush=True)


if __name__ == '__main__':
    main()
