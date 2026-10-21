"""Pre-register and generate a fresh-question, same-held-out-DB confirmation pool.

--prepare performs only deterministic selection and read-only gold execution.
--generate uses the local R2 MPS int8 model with all Hugging Face access offline.
It appends a resumable progress log and publishes the scoring input only once
all frozen questions finish. Original BIRD dev and sealed data are never read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / 'data/sys1/bird_train/plans.jsonl'
ALLOWLIST = ROOT / 'data/sys1/dev_dbs_bird.json'
EXCLUDED = ROOT / 'reports/fanout_r3/rl/ho200.jsonl'
RUN = ROOT / 'runs/r1r2/r2'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def rank_key(row, seed):
    return hashlib.sha256(f'{seed}\0{row["db_id"]}\0{row["id"]}'.encode()).hexdigest()


def source_fingerprints():
    paths = [PLANS, ALLOWLIST, EXCLUDED, RUN/'args.json', RUN/'vocab.json',
             RUN/'best_heads.pt', RUN/'best_lora/adapter_model.safetensors']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def execute_sql(con, sql, seconds):
    deadline = time.monotonic() + seconds
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    return con.execute(sql).fetchall()


def open_db(db):
    path = ROOT / 'data/sys1/dbs' / db / f'{db}.sqlite'
    con = sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)
    con.execute('PRAGMA query_only=ON')
    con.text_factory = lambda b: b.decode('utf-8', 'replace')
    return con


def select_cohort(rows, allowed, excluded, n, seed, gold_check):
    if n < 1:
        raise ValueError('n must be positive')
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row['db_id'] not in allowed or row['id'] in excluded or row.get('twin_of') or '_twin' in row['id']:
            continue
        if row.get('source') != 'bird' or row.get('split') != 'train' or not row['id'].startswith('bird_train_'):
            raise ValueError('Unexpected non-training provenance')
        if row['id'] in seen:
            raise ValueError('Duplicate question ID: ' + row['id'])
        seen.add(row['id'])
        groups[row['db_id']].append(row)
    groups = {db: iter(sorted(groups[db], key=lambda r: rank_key(r, seed))) for db in sorted(allowed)}
    selected, rejected = [], []
    while len(selected) < n and groups:
        for db in list(groups):
            while True:
                row = next(groups[db], None)
                if row is None:
                    del groups[db]
                    break
                error = gold_check(row)
                if error:
                    rejected.append({'id': row['id'], 'db_id': db, 'error': error})
                else:
                    selected.append(row)
                    break
            if len(selected) >= n:
                break
    if len(selected) != n:
        raise ValueError(f'Only {len(selected)} executable-gold fresh questions; requested {n}')
    return selected, rejected


def prepare(a):
    if a.manifest.exists():
        raise ValueError('Manifest already exists; frozen cohorts cannot be overwritten')
    allowed = set(json.loads(ALLOWLIST.read_text()))
    excluded = {r['id'] for r in read_rows(EXCLUDED)}
    connections = {}
    def gold_check(row):
        db = row['db_id']
        try:
            if db not in connections:
                connections[db] = open_db(db)
            execute_sql(connections[db], row['gold_sql'], a.seconds)
            return None
        except sqlite3.Error as exc:
            return str(exc)
    try:
        rows, rejected = select_cohort(read_rows(PLANS), allowed, excluded, a.n, a.seed, gold_check)
    finally:
        for con in connections.values():
            con.close()
    manifest = {'version': 1, 'scope': 'fresh questions on the same nine held-out TRAIN databases; not new-DB confirmation',
                'selection': 'SHA256 seed/db/id order, DB round robin; executable gold only; no label-based selection',
                'seed': a.seed, 'n': len(rows), 'seconds': a.seconds, 'requested_k': 64,
                'decoder': 'kbest only, max_pops=512; fixed64 prefixes; no samples',
                'database_counts': dict(sorted(Counter(r['db_id'] for r in rows).items())),
                'excluded_ho200_ids': sorted(excluded), 'gold_preflight_rejections': rejected,
                'sources_sha256': source_fingerprints(),
                'cohort': [{'id': r['id'], 'db_id': r['db_id']} for r in rows]}
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    with a.manifest.open('x') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print(json.dumps({'manifest': str(a.manifest), 'manifest_sha256': sha(a.manifest),
                      'n': len(rows), 'database_counts': manifest['database_counts'],
                      'gold_preflight_rejections': len(rejected)}), flush=True)


def verify_manifest(manifest):
    if manifest['sources_sha256'] != source_fingerprints():
        raise ValueError('Frozen source hashes changed; do not silently change the cohort/model')
    allowed = set(json.loads(ALLOWLIST.read_text()))
    excluded = set(manifest['excluded_ho200_ids'])
    ids = [r['id'] for r in manifest['cohort']]
    if len(ids) != manifest['n'] or len(set(ids)) != len(ids):
        raise ValueError('Invalid cohort IDs')
    if any(r['db_id'] not in allowed or r['id'] in excluded or not r['id'].startswith('bird_train_')
           for r in manifest['cohort']):
        raise ValueError('Invalid confirmation provenance')


def load_progress(path, manifest_hash, cohort):
    if not path.exists():
        return []
    # Recover a process interrupted during the final append; reject corruption
    # before the tail rather than discarding completed questions.
    content = path.read_bytes()
    if content and not content.endswith(b'\n'):
        tail = content.rfind(b'\n') + 1
        partial = path.with_suffix(path.suffix + f'.partial-{time.time_ns()}')
        partial.write_bytes(content[tail:])
        with path.open('r+b') as f:
            f.truncate(tail)
    rows = read_rows(path)
    expected = {r['id']: r['db_id'] for r in cohort}
    seen = set()
    for r in rows:
        if r['id'] in seen or expected.get(r['id']) != r['db_id'] or r.get('manifest_sha256') != manifest_hash:
            raise ValueError('Progress provenance mismatch: ' + r['id'])
        seen.add(r['id'])
    return rows


def execute_candidates(row, sqls, con, seconds):
    try:
        gold = set(map(tuple, execute_sql(con, row['gold_sql'], seconds)))
    except sqlite3.Error as exc:
        return [], str(exc)
    cands = []
    for rank, sql in enumerate(sqls):
        c = {'rank': rank, 'sql': sql}
        try:
            rows = execute_sql(con, sql, seconds)
            results = set(map(tuple, rows))
            sig = hashlib.sha1(repr(sorted(results, key=repr)).encode()).hexdigest()[:12]
            c.update(runs=True, nonnull=any(v is not None for r in rows for v in r),
                     ok=results == gold, sig=sig, result_rows=len(rows))
        except sqlite3.Error:
            c.update(runs=False, nonnull=False, ok=False, sig=None, result_rows=0)
        cands.append(c)
    return cands, None


def publish_output(path, progress, manifest):
    if len(progress) != manifest['n']:
        return False
    by_id = {r['id']: r for r in progress}
    valid = [by_id[r['id']] for r in manifest['cohort'] if not by_id[r['id']].get('gold_error')]
    content = ''.join(json.dumps(r) + '\n' for r in valid)
    if path.exists():
        if path.read_text() != content:
            raise ValueError('Existing final output disagrees with completed progress')
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content)
        temporary.replace(path)
    print(json.dumps({'complete': True, 'cohort': len(progress), 'valid_gold': len(valid),
                      'gold_errors': len(progress)-len(valid), 'output': str(path)}), flush=True)
    return True


def generate(a):
    if a.out is None:
        raise ValueError('--out is required for generation')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    manifest = json.loads(a.manifest.read_text())
    verify_manifest(manifest)
    manifest_hash = sha(a.manifest)
    progress_path = a.out.with_suffix('.progress.jsonl')
    progress = load_progress(progress_path, manifest_hash, manifest['cohort'])
    if publish_output(a.out, progress, manifest):
        return
    done = {r['id'] for r in progress}
    remaining = [r for r in manifest['cohort'] if r['id'] not in done]
    if a.limit:
        remaining = remaining[:a.limit]
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'reports/fanout_r2/zero_gpu')]
    import torch
    import train_sq2
    from luthor.sq.candidates import ValueIndex, find_candidates
    from luthor.sq.plan import compile_plan
    from luthor.sq2.model import TorchScores
    from luthor.sq2.topk import kbest
    from luthor.sq2.train import encode_db, question_body, select_cands, serve_chunks
    spec = importlib.util.spec_from_file_location('confirmation_sample_k', ROOT/'reports/fanout_r3/rl/sample_k.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    order = module.setup_vocab()
    # Offline env is set before importing transformers; the reused loader may
    # resolve only installed/cache artifacts and cannot download checkpoints.
    model = module.load_r2('mps')
    cards, examples, gold = train_sq2.load_sets([ROOT/'data/sys1/bird_train'], workers=4, strict=False, all_rows=True)
    examples = {e.id: e for e in examples}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    connections, value_indexes = {}, {}
    torch.manual_seed(10101)
    try:
        for wanted in remaining:
            t0 = time.monotonic()
            e = examples[wanted['id']]
            db = e.db_id
            if db != wanted['db_id']:
                raise ValueError('Example/manifest database mismatch')
            card = cards[db]
            if db not in connections:
                connections[db] = open_db(db)
                value_indexes[db] = ValueIndex.build(connections[db], card)
            row = gold[e.id]
            body = question_body(row)
            e.cands, _ = select_cands(find_candidates(body, card, extra=value_indexes[db].match(body)),
                                      order=order, body=body)
            graph = card.graph()
            compile_fn = lambda p: compile_plan(p, graph)
            decode_error = None
            with torch.inference_mode():
                tabs, chunk = next(iter(serve_chunks(card, [e], 1)))
                st, _, enc = encode_db(model, card, chunk, tabs)
                sc = TorchScores(model, enc[0], st.option_ids, e.cands)
                try:
                    sqls = [compile_fn(p) for p in kbest(sc, 64, compile_fn, max_pops=512)]
                except Exception as exc:
                    sqls, decode_error = [], f'{type(exc).__name__}: {exc}'
                torch.mps.synchronize()
                torch.mps.empty_cache()
            cands, gold_error = execute_candidates(row, sqls, connections[db], manifest['seconds'])
            result = {'id': e.id, 'db_id': db, 'question': row['question'], 'context': row.get('context', ''),
                      'gold_sql': row['gold_sql'], 'cands': cands, 'covered': row.get('outcome') == 'plan',
                      'pool_generator': 'r2_local_mps_int8_kbest', 'pool_status': 'cached_bounded_search',
                      'requested_k': 64, 'cost_available': False, 'manifest_sha256': manifest_hash,
                      'generation': {'device': 'mps', 'base_precision': 'weight-only-int8',
                                     'schema_context': 'one-question serve_chunks', 'seconds': time.monotonic()-t0}}
            if decode_error:
                result['decode_error'] = decode_error
            if gold_error:
                result.update(gold_error=True, gold_error_message=gold_error)
            with progress_path.open('a') as f:
                f.write(json.dumps(result) + '\n')
                f.flush()
                os.fsync(f.fileno())
            progress.append(result)
            print(json.dumps({'done': len(progress), 'total': manifest['n'], 'id': e.id,
                              'candidates': len(cands), 'seconds': time.monotonic()-t0}), flush=True)
    finally:
        for con in connections.values():
            con.close()
    if not publish_output(a.out, progress, manifest):
        print(json.dumps({'complete': False, 'done': len(progress), 'remaining': manifest['n']-len(progress),
                          'resume': str(progress_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument('--prepare', action='store_true')
    stages.add_argument('--generate', action='store_true')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--seed', type=int, default=10101)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--limit', type=int, default=0, help='Generate at most this many additional questions; resume later')
    a = parser.parse_args()
    if a.seconds <= 0 or a.limit < 0:
        parser.error('seconds must be positive and limit nonnegative')
    (prepare if a.prepare else generate)(a)


if __name__ == '__main__':
    main()
