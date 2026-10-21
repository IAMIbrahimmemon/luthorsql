"""Provenance and fixed-cohort checks for the held-out experiment runner."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('wide_pool_retention',
        Path(__file__).resolve().parents[1] / 'scripts/analysis/wide_pool_retention.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def inputs():
    row = dict(id='bird_train_train_1', db_id='university', gold_sql='SELECT 1',
               kbest=['SELECT 1', 'SELECT 2'], covered=True,
               runs={'SELECT 1': True, 'SELECT 2': True},
               ok={'SELECT 1': True, 'SELECT 2': False})
    metadata = dict(id=row['id'], db_id='university', gold_sql='SELECT 1', source='bird',
                    split='train', question='one?', context='')
    sigs = {row['id']: {'SELECT 1': ['one', 1, True], 'SELECT 2': ['two', 1, True]}}
    return [row], [metadata], sigs, {'university'}


def test_prepare_preserves_order_without_inventing_costs():
    records, manifest = runner.prepare_rows(*inputs())
    assert [c['sql'] for c in records[0]['cands']] == ['SELECT 1', 'SELECT 2']
    assert 'cost' not in records[0]['cands'][0]
    assert records[0]['pool_status'] == 'cached_bounded_search'
    assert manifest['valid_gold_questions'] == 1


def test_reject_original_dev_even_with_train_database_name():
    rows, metadata, sigs, allowed = inputs()
    rows[0]['id'] = 'bird_dev_1'
    with pytest.raises(ValueError, match='Non-held-out-training'):
        runner.prepare_rows(rows, metadata, sigs, allowed)


def test_reject_stale_signature_execution():
    rows, metadata, sigs, allowed = inputs()
    sigs[rows[0]['id']]['SELECT 1'][0] = 'ERR'
    with pytest.raises(ValueError, match='Execution-cache disagreement'):
        runner.prepare_rows(rows, metadata, sigs, allowed)


def test_excluded_gold_errors_are_auditable():
    rows, metadata, sigs, allowed = inputs()
    rows.append({**rows[0], 'id': 'bird_train_train_2', 'gold_error': True})
    records, manifest = runner.prepare_rows(rows, metadata, sigs, allowed)
    assert len(records) == 1
    assert manifest['excluded_gold_error_ids'] == ['bird_train_train_2']


def test_missing_score_rows_require_explicit_pilot():
    records, _ = runner.prepare_rows(*inputs())
    reference = records + [{**records[0], 'id': 'bird_train_train_2'}]
    with pytest.raises(ValueError, match='cohort differs'):
        runner.validate_scored(records, reference, {'university'})
    runner.validate_scored(records, reference, {'university'}, allow_subset=True)


def test_scoring_cannot_reorder_or_change_candidates():
    records, _ = runner.prepare_rows(*inputs())
    altered = [{**records[0], 'cands': list(reversed(records[0]['cands']))}]
    with pytest.raises(ValueError, match='candidate differs'):
        runner.validate_scored(altered, records, {'university'})
