"""CPU retention diagnostics and a fixed out-of-database prediction baseline.

Only held-out *training* databases belong in ``evaluate_heldout``. This is an
experimental baseline, not evidence that problem 01-01 is solved. Correct-class
rank is a label-dependent diagnostic and deliberately never a model feature.
Scores can depend on the prefix's schema context: supply ``prefix_scores`` as
{str(k): [score_or_None, ...]} to reproduce independently scored prefixes.
Without that field the curve explicitly uses frozen full-pool scores.
"""
from __future__ import annotations

import math

import numpy as np


FEATURE_NAMES = ("log_k", "log_unique_sql", "log_live_classes", "winner_margin",
                 "winner_score", "winner_rank_fraction", "greedy_selected", "has_live")


def _check(records, ks):
    if not records or not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("nonempty records and positive integer pool sizes required")
    if len(ks) != len(set(ks)):
        raise ValueError("pool sizes must be unique")
    seen = set()
    for r in records:
        if not r.get("id") or not r.get("db_id") or r["id"] in seen:
            raise ValueError("unique question ids and database ids required")
        seen.add(r["id"])
        if r.get("gold_valid") is False or r.get("gold_ok") is False:
            raise ValueError(f"{r['id']}: invalid gold must be excluded explicitly before analysis")
        bounded = (r.get("pool_status") == "cached_bounded_search"
                   and type(r.get("requested_k")) is int and r["requested_k"] >= max(ks))
        if "cands" not in r or (not r["cands"] and not bounded):
            raise ValueError(f"{r['id']}: empty pool without explicit bounded-search provenance")
        if "greedy_ok" in r and (type(r["greedy_ok"]) not in (bool, int) or r["greedy_ok"] not in (0, 1)):
            raise ValueError("greedy_ok must be binary")
        if len(r["cands"]) < max(ks) and not (r.get("pool_exhausted", False) or bounded):
            raise ValueError(f"{r['id']}: incomplete pool; mark pool_exhausted only if generation exhausted")
        for c in r["cands"]:
            if type(c.get("ok")) not in (bool, int) or c["ok"] not in (0, 1):
                raise ValueError(f"{r['id']}: explicit binary execution labels required")
            for key in ("runs", "nonnull"):
                if type(c.get(key)) is not bool:
                    raise ValueError(f"{r['id']}: explicit {key} boolean required")


def _prefix(r, k):
    cands = r["cands"][:k]
    if "prefix_scores" in r:
        if str(k) not in r["prefix_scores"]:
            raise ValueError(f"{r['id']}: missing independently scored prefix {k}")
        scores = r["prefix_scores"][str(k)]
        if len(scores) != len(cands):
            raise ValueError("prefix score length differs from candidate prefix")
        cands = [{**c, "rr": s} for c, s in zip(cands, scores)]
    if not cands:
        return {"id": r["id"], "db_id": r["db_id"], "k": k,
                "features": [math.log1p(k)] + [0.0] * (len(FEATURE_NAMES) - 1),
                "greedy": int(r.get("greedy_ok", False)), "pick": 0, "oracle": 0, "reachable_oracle": 0,
                "realized_k": 0, "first_right_rank": None}
    unique, seen = [], set()
    for c in cands:
        sql = c.get("sql")
        if not sql or sql in seen:
            continue
        seen.add(sql)
        if c.get("rr") is None or not math.isfinite(float(c["rr"])):
            raise ValueError(f"{r['id']}: missing/nonfinite score on first copy of SQL at k={k}")
        unique.append(c)
    live = [c for c in unique if c["runs"] and c["nonnull"]]
    for c in live:
        if not c.get("sig"):
            raise ValueError(f"{r['id']}: live result signature required")
    eligible = live or unique
    # Stable max agrees with production's first-candidate tie breaking.
    pick = max(eligible, key=lambda c: c["rr"]) if eligible else cands[0]
    scores = sorted((float(c["rr"]) for c in eligible), reverse=True)
    margin = scores[0] - scores[1] if len(scores) > 1 else 0.0
    winner_rank = next(i for i, c in enumerate(cands) if c.get("sql") == pick.get("sql"))
    features = [math.log1p(k), math.log1p(len(unique)),
                math.log1p(len({c["sig"] for c in live})), margin,
                scores[0] if scores else 0.0, winner_rank / max(1, len(cands) - 1),
                float(winner_rank == 0), float(bool(live))]
    return {"id": r["id"], "db_id": r["db_id"], "k": k, "features": features,
            "greedy": int(r.get("greedy_ok", cands[0]["ok"])), "pick": int(pick["ok"]),
            "oracle": int(any(c["ok"] for c in cands)),
            "reachable_oracle": int(any(c["ok"] for c in eligible)) if eligible else int(pick["ok"]),
            "realized_k": len(cands),
            "first_right_rank": next((i for i, c in enumerate(cands) if c["ok"]), None)}


def _summarize(rows):
    n = len(rows)
    g, a, o = (sum(r[key] for r in rows) / n for key in ("greedy", "pick", "oracle"))
    return {"n": n, "greedy": g, "accuracy": a, "oracle": o,
            "reachable_oracle": sum(r["reachable_oracle"] for r in rows) / n,
            "rho": (a - g) / (o - g) if o > g else None,
            "wins_vs_greedy": sum(r["pick"] > r["greedy"] for r in rows),
            "losses_vs_greedy": sum(r["pick"] < r["greedy"] for r in rows),
            "mean_realized_k": sum(r["realized_k"] for r in rows) / n}


def nested_curve(records, ks=(8, 16, 32)):
    """Same-question curves, in fractions; undefined rho is None, negative rho is retained.

    Lists are generation-order prefixes, not sorted by selector scores. Short
    pools require ``pool_exhausted`` or ``pool_status="cached_bounded_search"``
    plus ``requested_k >= max(ks)`` and remain in every curve. The latter means
    cached-search replay, not fresh width-specific generation. An explicit
    ``greedy_ok`` may provide the actual baseline when generation fails.
    Gold validity/filtering must be established before supplying these records.
    """
    _check(records, ks)
    rows = [_prefix(r, k) for r in records for k in sorted(ks)]
    modes = {"independent_prefix" if "prefix_scores" in r else "frozen_full_pool" for r in records}
    if len(modes) != 1:
        raise ValueError("cannot mix frozen and independently scored prefix protocols")
    return {"score_protocol": next(iter(modes)), "units": "fractions",
            "pool_protocols": sorted({r.get("pool_status", "nested_candidates") for r in records}),
            "curve": {str(k): _summarize([r for r in rows if r["k"] == k]) for k in sorted(ks)},
            "rows": rows}


def _fit(train, ridge):
    """Fixed penalized logistic model; fit scaling on training DBs only."""
    x = np.asarray([r["features"] for r in train], dtype=float)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.0
    x = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    y = np.asarray([r["pick"] for r in train], dtype=float)
    beta = np.zeros(x.shape[1])
    # A weak intercept penalty keeps constant-label folds finite.
    penalty = np.diag([1e-6] + [ridge] * (x.shape[1] - 1))
    def objective(b):
        logits = x @ b
        return float(np.sum(np.logaddexp(0, logits) - y * logits) + 0.5 * b @ penalty @ b)

    converged = False
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -35, 35)))
        w = np.maximum(p * (1 - p), 1e-8)
        step = np.linalg.solve((x.T * w) @ x + penalty, x.T @ (p - y) + penalty @ beta)
        old_objective = objective(beta)
        rate = 1.0
        while rate > 2 ** -30 and objective(beta - rate * step) > old_objective + 1e-12:
            rate *= 0.5
        if rate <= 2 ** -30 or not np.all(np.isfinite(step)):
            raise ValueError("logistic fitting failed to find a finite descent step")
        beta -= rate * step
        if np.max(np.abs(rate * step)) < 1e-8:
            converged = True
            break
    if not converged:
        raise ValueError("logistic fitting did not converge; do not report predictions")
    return {"mean": mean.tolist(), "scale": scale.tolist(), "beta": beta.tolist()}


def _apply(model, rows):
    x = np.asarray([r["features"] for r in rows], dtype=float)
    z = np.column_stack((np.ones(len(x)), (x - model["mean"]) / model["scale"]))
    return (1 / (1 + np.exp(-np.clip(z @ model["beta"], -35, 35)))).tolist()


def _predict(train, test, ridge):
    return _apply(_fit(train, ridge), test)


def evaluate_heldout(records, ks=(8, 16, 32), *, allowed_dbs, ridge=1.0):
    """Leave-one-DB-out accuracy prediction and pool choice with a fixed model.

    All hyperparameters must be fixed before running this evaluation. Choosing a
    model/ridge after seeing these errors requires another outer evaluation.
    Each fold chooses k from predicted accuracies on its unlabeled pool features;
    its outcome labels are used solely for measurement. This evaluates selection
    of a pool size for a new DB, not a globally frozen deployment pool size.
    ±1 point is an observed error criterion, NOT a statistical guarantee.
    """
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    allowed = set(allowed_dbs)
    actual = {r.get("db_id") for r in records}
    if not actual <= allowed:
        raise ValueError(f"databases outside held-out training manifest: {actual - allowed}")
    if len(actual) < 3:
        raise ValueError("at least three held-out databases required")
    result = nested_curve(records, ks)
    rows = result.pop("rows")
    folds, predictions = [], []
    for db in sorted(actual):
        train = [r for r in rows if r["db_id"] != db]
        test = [r for r in rows if r["db_id"] == db]
        pred = _predict(train, test, ridge)
        pr = [{**r, "predicted_accuracy": p} for r, p in zip(test, pred)]
        predictions.extend(pr)
        by_k = {k: [r for r in pr if r["k"] == k] for k in sorted(ks)}
        p_by_k = {k: float(np.mean([r["predicted_accuracy"] for r in rs])) for k, rs in by_k.items()}
        chosen = max(sorted(ks), key=lambda k: p_by_k[k])
        observed = {k: _summarize(rs)["accuracy"] for k, rs in by_k.items()}
        folds.append({"test_db": db, "train_dbs": sorted(actual - {db}), "chosen_k": chosen,
                      "predicted_accuracy": p_by_k, "observed_accuracy": observed,
                      "prediction_error": {k: p_by_k[k] - observed[k] for k in sorted(ks)},
                      "chosen_accuracy": observed[chosen],
                      "regret_vs_hindsight_best": max(observed.values()) - observed[chosen]})
    for k in sorted(ks):
        rs = [r for r in predictions if r["k"] == k]
        predicted = float(np.mean([r["predicted_accuracy"] for r in rs]))
        point = result["curve"][str(k)]
        point["predicted_accuracy"] = predicted
        point["prediction_error"] = predicted - point["accuracy"]
        errors = [f["prediction_error"][k] for f in folds]
        point["per_db_mean_absolute_error"] = float(np.mean(np.abs(errors)))
        point["per_db_max_absolute_error"] = float(np.max(np.abs(errors)))
        point["within_one_point_observed"] = abs(point["prediction_error"]) <= 0.01
        # Conditional retention uses measured oracle/greedy; not an ex-ante rho forecast.
        gap = point["oracle"] - point["greedy"]
        point["predicted_rho_conditional_on_observed_oracle"] = ((predicted - point["greedy"]) / gap if gap > 0 else None)
    result["question_outcomes"] = {str(k): {r["id"]: {
        field: r[field] for field in ("db_id", "greedy", "pick", "oracle", "reachable_oracle")
    } for r in rows if r["k"] == k} for k in sorted(ks)}
    result.update(model="fixed ridge logistic, leave-one-database-out", ridge=ridge,
                  fit_status="all folds converged with finite descent steps",
                  features=list(FEATURE_NAMES), folds=folds,
                  note="Observed error only; no claim of a one-point generalization guarantee.")
    return result



def fit_retention_model(records, ks=(8, 16, 32), *, allowed_dbs, ridge=1.0):
    """Freeze the prespecified baseline on a pilot; returns JSON-serializable state.

    Does not measure training accuracy or select hyperparameters. Persist this
    object before observing confirmation outcomes; predictions never refit it.
    """
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if not {r.get("db_id") for r in records} <= set(allowed_dbs):
        raise ValueError("databases outside held-out training manifest")
    data = nested_curve(records, ks)
    model = _fit(data["rows"], ridge)
    model.update(version=1, features=list(FEATURE_NAMES), ks=sorted(ks), ridge=ridge,
                 train_ids=sorted(r["id"] for r in records),
                 train_dbs=sorted({r["db_id"] for r in records}),
                 score_protocol=data["score_protocol"], pool_protocols=data["pool_protocols"])
    return model


def predict_retention(model, records, *, allowed_dbs, allow_same_dbs=False):
    """Predict fresh unlabeled pools with frozen coefficients and normalization.

    Correctness, gold validity, gold SQL and greedy labels are never accessed.
    Requires a disjoint question cohort, compatible protocols and an explicit
    database allowlist. By default the databases must also be new; setting
    ``allow_same_dbs=True`` is a within-database question confirmation, not an
    out-of-database replication. Caller must exclude pilot questions by stable
    identity, including aliases: arbitrary renamed IDs cannot be detected here.
    """
    if model.get("version") != 1 or model.get("features") != list(FEATURE_NAMES):
        raise ValueError("incompatible frozen model version or feature schema")
    if {r.get("id") for r in records} & set(model["train_ids"]):
        raise ValueError("confirmation questions overlap pilot")
    dbs = {r.get("db_id") for r in records}
    if not dbs <= set(allowed_dbs):
        raise ValueError("databases outside confirmation allowlist")
    overlap = dbs & set(model["train_dbs"])
    if overlap and not allow_same_dbs:
        raise ValueError("confirmation databases overlap pilot; use explicit within-database scope")
    # Only inference fields cross this boundary. Dummy labels let the common
    # strict feature extractor be reused; none escape as reported outcomes.
    clean = []
    for r in records:
        row = {key: r[key] for key in ("id", "db_id", "pool_status", "requested_k",
               "pool_exhausted", "prefix_scores") if key in r}
        row["cands"] = [{**{key: c[key] for key in ("sql", "rr", "runs", "nonnull", "sig")
                                  if key in c}, "ok": False} for c in r["cands"]]
        clean.append(row)
    data = nested_curve(clean, model["ks"])
    for field in ("score_protocol", "pool_protocols"):
        if data[field] != model[field]:
            raise ValueError(f"confirmation {field} differs from frozen pilot protocol")
    probabilities = _apply(model, data["rows"])
    rows = [{"id": r["id"], "db_id": r["db_id"], "k": r["k"], "predicted_accuracy": p}
            for r, p in zip(data["rows"], probabilities)]
    curve = {str(k): float(np.mean([r["predicted_accuracy"] for r in rows if r["k"] == k]))
             for k in model["ks"]}
    return {"units": "fractions", "predicted_accuracy": curve,
            "chosen_k": max(model["ks"], key=lambda k: curve[str(k)]),
            "rows": rows, "score_protocol": model["score_protocol"],
            "scope": "fresh_questions_with_shared_databases" if overlap else "new_databases",
            "note": "Frozen prediction; no confirmation labels accessed and no refitting."}
