"""Tests cover production pick semantics, malformed measurements and label isolation."""
from copy import deepcopy

import pytest

from luthor.retention import evaluate_heldout, nested_curve


def cand(sql, score, ok=False, *, live=True, sig=None):
    c = dict(sql=sql, ok=ok, runs=True, nonnull=live, sig=sig or sql)
    if score is not None:
        c["rr"] = score
    return c


def record(q, db, cands, **kw):
    return dict(id=q, db_id=db, cands=cands, **kw)


def test_nested_curves_production_filter_and_duplicate_ties():
    rs = [record("a", "db", [cand("g", 2, True), cand("g", None, True),
                              cand("dead", 100, live=False), cand("wrong", 2)])]
    out = nested_curve(rs, (1, 2, 4))
    assert out["score_protocol"] == "frozen_full_pool"
    assert [out["curve"][str(k)]["accuracy"] for k in (1, 2, 4)] == [1, 1, 1]
    assert out["curve"]["4"]["rho"] is None


def test_negative_retention_not_clipped():
    rs = [record("a", "db", [cand("g", 0, True), cand("w", 2)]),
          record("b", "db", [cand("g", 2), cand("right", 0, True)])]
    result = nested_curve(rs, (2,))["curve"]["2"]
    assert result["rho"] == -1
    assert result["wins_vs_greedy"] == 0
    assert result["losses_vs_greedy"] == 1


def test_prefix_scores_change_pick_without_mutating_input():
    rs = [record("a", "db", [cand("g", 0, True), cand("w", 10)],
                 prefix_scores={"1": [2], "2": [20, 1]})]
    before = deepcopy(rs)
    assert nested_curve(rs, (1, 2))["curve"]["2"]["accuracy"] == 1
    assert rs == before


def test_incomplete_scores_fail_instead_of_inflating_accuracy():
    rs = [record("a", "db", [cand("g", 10, True), cand("w", None)])]
    with pytest.raises(ValueError, match="missing/nonfinite"):
        nested_curve(rs, (2,))
    rs[0]["cands"][1]["rr"] = float("nan")
    with pytest.raises(ValueError, match="missing/nonfinite"):
        nested_curve(rs, (2,))


def test_exhausted_pools_retained_and_unknown_truncation_rejected():
    rs = [record("a", "db", [cand("g", 1)])]
    with pytest.raises(ValueError, match="incomplete pool"):
        nested_curve(rs, (2,))
    rs[0]["pool_exhausted"] = True
    assert nested_curve(rs, (2,))["curve"]["2"]["mean_realized_k"] == 1


def _dataset():
    return [record(f"{db}-{i}", db, [cand("g", i / 4, bool(i % 2)),
                                    cand("w", 1 - i / 4, bool((i + 1) % 2))])
            for db in ("train_a", "train_b", "train_c") for i in range(4)]


def test_database_manifest_rejects_foreign_splits():
    with pytest.raises(ValueError, match="outside held-out"):
        evaluate_heldout(_dataset(), (1, 2), allowed_dbs=["train_a", "train_b"])


def test_heldout_labels_do_not_affect_own_fold_predictions_or_k():
    rs = _dataset()
    manifest = ["train_a", "train_b", "train_c"]
    original = evaluate_heldout(rs, (1, 2), allowed_dbs=manifest)
    for r in rs:
        if r["db_id"] == "train_a":
            for c in r["cands"]:
                c["ok"] = not c["ok"]
    changed = evaluate_heldout(rs, (1, 2), allowed_dbs=manifest)
    a, b = original["folds"][0], changed["folds"][0]
    assert a["test_db"] == "train_a"
    assert a["train_dbs"] == ["train_b", "train_c"]
    assert a["predicted_accuracy"] == b["predicted_accuracy"]
    assert a["chosen_k"] == b["chosen_k"]
    assert "first_right_rank" not in original["features"]
    for point in original["curve"].values():
        assert 0 <= point["predicted_accuracy"] <= 1
        assert point["per_db_max_absolute_error"] >= point["per_db_mean_absolute_error"]


def test_missing_labels_and_duplicate_questions_rejected():
    rs = _dataset()
    del rs[0]["cands"][0]["ok"]
    with pytest.raises(ValueError, match="execution labels"):
        nested_curve(rs, (2,))
    rs = _dataset()
    with pytest.raises(ValueError, match="unique question"):
        nested_curve(rs + [rs[0]], (2,))


def test_bounded_search_keeps_empty_failure_and_short_pool_in_denominator():
    rs = [record("a", "db", [], pool_status="cached_bounded_search", requested_k=64),
          record("b", "db", [cand("g", 1, True)],
                 pool_status="cached_bounded_search", requested_k=64)]
    out = nested_curve(rs, (8, 16, 32))
    for point in out["curve"].values():
        assert point["n"] == 2
        assert point["accuracy"] == 0.5
        assert point["mean_realized_k"] == 0.5
    with pytest.raises(ValueError, match="empty pool"):
        nested_curve(rs, (128,))


def test_reachable_oracle_exposes_correct_empty_result_exclusion():
    rs = [record("a", "db", [cand("correct_empty", 100, True, live=False), cand("wrong_live", 0)])]
    point = nested_curve(rs, (2,))["curve"]["2"]
    assert point["oracle"] == 1
    assert point["reachable_oracle"] == point["accuracy"] == 0


def test_constant_label_folds_converge_and_outcomes_available():
    rs = _dataset()
    for r in rs:
        for c in r["cands"]:
            c["ok"] = False
    out = evaluate_heldout(rs, (1, 2), allowed_dbs=["train_a", "train_b", "train_c"])
    assert "converged" in out["fit_status"]
    assert out["curve"]["1"]["predicted_accuracy"] < 0.001
    assert len(out["question_outcomes"]["2"]) == len(rs)


def test_frozen_model_json_roundtrip_and_unlabeled_prediction():
    import json
    from luthor.retention import fit_retention_model, predict_retention
    rs = _dataset()
    model = fit_retention_model(rs, (1, 2), allowed_dbs={r['db_id'] for r in rs})
    model = json.loads(json.dumps(model))
    frozen = deepcopy(model)
    fresh = deepcopy(rs[:4])
    for r in fresh:
        r['id'] = 'fresh-' + r['id']
        r['db_id'] = 'new_db'
    original = predict_retention(model, fresh, allowed_dbs=['new_db'])
    for r in fresh:
        r['gold_valid'] = False
        r['greedy_ok'] = 'not a label'
        for c in r['cands']:
            del c['ok']
    unlabeled = predict_retention(model, fresh, allowed_dbs=['new_db'])
    assert original == unlabeled
    assert model == frozen
    assert original['scope'] == 'new_databases'
    assert set(original['predicted_accuracy']) == {'1', '2'}


def test_frozen_confirmation_overlap_and_protocol_guards():
    from luthor.retention import fit_retention_model, predict_retention
    rs = _dataset()
    allowed = {r['db_id'] for r in rs}
    model = fit_retention_model(rs, (1, 2), allowed_dbs=allowed)
    with pytest.raises(ValueError, match='questions overlap'):
        predict_retention(model, rs, allowed_dbs=allowed)
    fresh = deepcopy(rs)
    for r in fresh:
        r['id'] = 'fresh-' + r['id']
    with pytest.raises(ValueError, match='databases overlap'):
        predict_retention(model, fresh, allowed_dbs=allowed)
    out = predict_retention(model, fresh, allowed_dbs=allowed, allow_same_dbs=True)
    assert out['scope'] == 'fresh_questions_with_shared_databases'
    for r in fresh:
        r['prefix_scores'] = {'1': [1], '2': [1, 2]}
    with pytest.raises(ValueError, match='score_protocol differs'):
        predict_retention(model, fresh, allowed_dbs=allowed, allow_same_dbs=True)
