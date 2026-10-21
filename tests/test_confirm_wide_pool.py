import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

spec = importlib.util.spec_from_file_location('confirm_wide_pool', Path(__file__).resolve().parents[1]/'scripts/analysis/confirm_wide_pool.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def rows():
    return [dict(id=f'bird_train_train_{db}_{i}', db_id=db, source='bird', split='train', gold_sql='SELECT 1')
            for db in ('a','b') for i in range(8)]


def test_deterministic_balancing_exclusions_and_gold_filter():
    source = rows()
    excluded = {source[0]['id']}
    check = lambda r: 'failure' if r['id'].endswith('_1') else None
    first, rejected = m.select_cohort(source, {'a','b'}, excluded, 6, 10101, check)
    again, _ = m.select_cohort(source[::-1], {'b','a'}, excluded, 6, 10101, check)
    assert first == again
    assert [r['db_id'] for r in first] == ['a','b']*3
    assert not {r['id'] for r in first} & excluded
    assert all(not r['id'].endswith('_1') for r in first)


def test_insufficient_executable_gold_fails():
    with pytest.raises(ValueError, match='Only 0'):
        m.select_cohort(rows(), {'a','b'}, set(), 3, 0, lambda r: 'bad')


def test_execution_keeps_null_correct_result_and_errors():
    con = sqlite3.connect(':memory:')
    cands, error = m.execute_candidates({'gold_sql':'SELECT NULL'}, ['SELECT NULL','SELECT 1','SELECT absent'],con,1)
    assert error is None
    assert cands[0]['ok'] and cands[0]['runs'] and not cands[0]['nonnull']
    assert not cands[1]['ok'] and not cands[2]['runs']
    con.close()


def test_progress_preserves_completed_rows_and_recovers_torn_tail(tmp_path):
    path = tmp_path/'progress.jsonl'
    row = {'id':'bird_train_train_1','db_id':'a','manifest_sha256':'hash'}
    path.write_text(json.dumps(row)+'\n'+ '{"id":')
    assert m.load_progress(path, 'hash', [{'id':row['id'],'db_id':'a'}]) == [row]
    assert path.read_text().endswith('\n')
    assert list(tmp_path.glob('*.partial-*'))
    with pytest.raises(ValueError, match='provenance mismatch'):
        m.load_progress(path,'wrong',[{'id':row['id'],'db_id':'a'}])


def test_publish_only_complete_cohort_and_exclude_failed_gold(tmp_path):
    rows = [{'id':'a','cands':[]}, {'id':'b','gold_error':True,'cands':[]}]
    manifest = {'n':2,'cohort':[{'id':'a'},{'id':'b'}]}
    out = tmp_path/'cands.jsonl'
    assert not m.publish_output(out, rows[:1], manifest)
    assert not out.exists()
    assert m.publish_output(out,rows,manifest)
    assert m.read_rows(out) == rows[:1]
