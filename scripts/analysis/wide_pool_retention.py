"""Held-out TRAINING-DB input conversion and analysis for research problem 01-01.

No model loads, network requests, or original-dev/sealed-set evaluation. `prepare`
converts the saved R2 k-best pool without mixing in samples or the flip pool.
`analyze` consumes validator scores produced by a separate local scorer.

  python scripts/analysis/wide_pool_retention.py prepare --out reports/research/01-01/heldout_kbest_candidates.jsonl
  python scripts/analysis/wide_pool_retention.py analyze --scored SCORED.jsonl --reference CANDIDATES.jsonl --out ANALYSIS.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = ROOT / 'data/sys1/dev_dbs_bird.json'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def heldout_dbs(path: Path = HELDOUT) -> set[str]:
    allowed = set(json.loads(path.read_text()))
    canonical = set(json.loads(HELDOUT.read_text()))
    if not allowed or not allowed <= canonical:
        raise ValueError('DB allowlist must be a nonempty subset of held-out BIRD training DBs')
    return allowed


def check_rows(rows: list[dict], allowed: set[str]) -> None:
    seen = set()
    for r in rows:
        key = (r['db_id'], r['id'])
        if r['db_id'] not in allowed or not r['id'].startswith('bird_train_'):
            raise ValueError(f'Non-held-out-training question: {key}')
        if key in seen:
            raise ValueError(f'Duplicate question: {key}')
        seen.add(key)
    if not rows:
        raise ValueError('Empty evaluation cohort')


def prepare_rows(rows: list[dict], metadata: list[dict], signatures: dict,
                 allowed: set[str]) -> tuple[list[dict], dict]:
    check_rows(rows, allowed)
    meta = {}
    wanted = {r['id'] for r in rows}
    for r in metadata:
        if r['id'] in wanted:
            if r['id'] in meta:
                raise ValueError(f'Duplicate metadata: {r["id"]}')
            meta[r['id']] = r
    converted, excluded = [], []
    for r in rows:
        if r.get('gold_error'):
            excluded.append(r['id'])
            continue
        m = meta[r['id']]
        if m['db_id'] != r['db_id'] or m.get('source') != 'bird' or m.get('split') != 'train':
            raise ValueError(f'Metadata provenance mismatch: {r["id"]}')
        if m['gold_sql'] != r['gold_sql']:
            raise ValueError(f'Gold SQL mismatch: {r["id"]}')
        candidates = []
        for rank, sql in enumerate(r['kbest']):
            if not sql:
                candidates.append(dict(rank=rank, sql=sql, runs=False, nonnull=False, ok=False, sig=None))
                continue
            sig, nrows, nonnull = signatures[r['id']][sql]
            runs = bool(r['runs'][sql])
            if runs != (sig != 'ERR'):
                raise ValueError(f'Execution-cache disagreement: {r["id"]}, rank {rank}')
            correct = bool(r['ok'][sql])
            if correct and not runs:
                raise ValueError(f'Correct candidate cannot fail execution: {r["id"]}')
            candidates.append(dict(rank=rank, sql=sql, runs=runs, nonnull=bool(nonnull),
                                   ok=correct, sig=sig if runs else None, result_rows=nrows))
        converted.append(dict(id=r['id'], db_id=r['db_id'], question=m['question'],
                              context=m.get('context', ''), gold_sql=r['gold_sql'],
                              covered=r['covered'], in_slots=r['covered'], cands=candidates,
                              pool_generator='r2_cached_kbest', cost_available=False, pool_status="cached_bounded_search",
                              requested_k=64))
    check_rows(converted, allowed)
    return converted, dict(input_questions=len(rows), valid_gold_questions=len(converted),
                           excluded_gold_error_ids=excluded, gold_error_policy='exclude',
                           candidate_order='original kbest order; no resampling or SQL deduplication',
                           policy_costs='unavailable in source; omitted',
                           pool_semantics='cached kbest64 bounded-search prefixes; short pools do not prove exhaustive generation',
                           database_counts={db: sum(r['db_id'] == db for r in converted)
                                            for db in sorted({r['db_id'] for r in converted})})


def validate_scored(rows: list[dict], reference: list[dict], allowed: set[str],
                    *, allow_subset: bool = False) -> None:
    """Reject partial jobs and changed pools before reporting an experiment."""
    check_rows(reference, allowed)
    check_rows(rows, allowed)
    expected = {r['id']: r for r in reference}
    actual = {r['id']: r for r in rows}
    if (not allow_subset and actual.keys() != expected.keys()) or not actual.keys() <= expected.keys():
        raise ValueError('Scored cohort differs from reference; --allow-subset is required for a pilot')
    fields = ('sql', 'rank', 'ok', 'runs', 'nonnull', 'sig')
    for qid, r in actual.items():
        ref = expected[qid]
        if r['db_id'] != ref['db_id'] or len(r['cands']) != len(ref['cands']):
            raise ValueError(f'Scored pool differs from reference: {qid}')
        for c, rc in zip(r['cands'], ref['cands']):
            if any(c.get(f) != rc.get(f) for f in fields):
                raise ValueError(f'Scored candidate differs from reference: {qid}')


def fingerprint(path: Path) -> dict:
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--pool', type=Path, default=ROOT / 'reports/fanout_r3/rl/ho200.jsonl')
    prep.add_argument('--metadata', type=Path, default=ROOT / 'data/sys1/bird_train/plans.jsonl')
    prep.add_argument('--signatures', type=Path, default=ROOT / 'docs/research_problems/scripts/sigs_ho200.json')
    ana = sub.add_parser('analyze')
    ana.add_argument('--scored', type=Path, required=True)
    ana.add_argument('--reference', type=Path, required=True, help='Full prepared candidate file')
    ana.add_argument('--allow-subset', action='store_true', help='Explicit pilot only; not full-cohort evidence')
    ana.add_argument('--ks', type=int, nargs='+', default=[8, 16, 32])
    for p in (prep, ana):
        p.add_argument('--heldout-dbs', type=Path, default=HELDOUT)
        p.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    allowed = heldout_dbs(a.heldout_dbs)
    if a.out.exists():
        parser.error(f'Output exists; choose a new path: {a.out}')
    if a.stage == 'prepare':
        manifest_path = a.out.with_suffix('.manifest.json')
        if manifest_path.exists():
            parser.error(f'Manifest exists; choose a new path: {manifest_path}')
        rows, manifest = prepare_rows(read_jsonl(a.pool), read_jsonl(a.metadata),
                                      json.loads(a.signatures.read_text()), allowed)
        manifest['sources'] = {k: fingerprint(getattr(a, k)) for k in
                               ('pool', 'metadata', 'signatures', 'heldout_dbs')}
        manifest['status'] = 'prepared_unscored; no selector accuracy claimed'
        write_new(a.out, ''.join(json.dumps(r) + '\n' for r in rows))
        manifest['output'] = fingerprint(a.out)
        write_new(manifest_path, json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest, indent=2))
    else:
        sys.path.insert(0, str(ROOT / "src"))
        from luthor.retention import evaluate_heldout
        rows = read_jsonl(a.scored)
        check_rows(rows, allowed)
        reference = read_jsonl(a.reference)
        validate_scored(rows, reference, allowed, allow_subset=a.allow_subset)
        result = evaluate_heldout(rows, ks=tuple(a.ks), allowed_dbs=allowed)
        from luthor.stats import cluster_signflip, question_signflip
        base_k = min(a.ks)
        outcomes = result['question_outcomes']
        base = {qid: q['pick'] for qid, q in outcomes[str(base_k)].items()}
        db = {qid: q['db_id'] for qid, q in outcomes[str(base_k)].items()}
        result['paired_vs_smallest_k'] = {
            str(k): {'baseline_k': base_k,
                     'cluster_signflip': cluster_signflip(base, {i:q['pick'] for i,q in outcomes[str(k)].items()}, db),
                     'question_signflip': question_signflip(base, {i:q['pick'] for i,q in outcomes[str(k)].items()}),
                     'delta_units': 'percentage_points'}
            for k in sorted(a.ks) if k != base_k}
        result['cohort_status'] = 'pilot_subset' if a.allow_subset else 'complete_reference_cohort'
        result['reference'] = fingerprint(a.reference)
        result['input'] = fingerprint(a.scored)
        result['heldout_allowlist'] = fingerprint(a.heldout_dbs)
        write_new(a.out, json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
