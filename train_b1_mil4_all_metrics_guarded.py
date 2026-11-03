"""Strict exploratory test: B1+MIL4 must improve all four inner OOF metrics.

Selection uses only inner OOF data. Calibration predictions used to assess
candidates are cross-fitted by inner_fold. If no candidate strictly improves
ACC, balanced accuracy, AUC, and Brier over raw B1, that outer fold falls back
to raw B1 rather than relaxing the rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)

LAMBDAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
THRESHOLDS = np.arange(0.05, 0.951, 0.01)


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def logit(p):
    p = clip(p)
    return np.log(p / (1.0 - p))


def fit_platt(p, y):
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(logit(p).reshape(-1, 1), np.asarray(y, dtype=int))
    return model


def apply_platt(model, p):
    return model.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def crossfit_platt(p, y, folds):
    result = np.empty(len(y), dtype=float)
    for fold in sorted(np.unique(folds)):
        tr, va = folds != fold, folds == fold
        if len(np.unique(y[tr])) < 2:
            raise ValueError(f"Calibration train partition lacks a class: {fold}")
        result[va] = apply_platt(fit_platt(p[tr], y[tr]), p[va])
    return result


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
    }


def choose_threshold(y, p):
    rows = [metrics(y, p, t) for t in THRESHOLDS]
    return max(rows, key=lambda x: (x["balanced_accuracy"], x["accuracy"],
                                    x["auc"], -x["brier"],
                                    -abs(x["threshold"] - 0.5)))


def paired(b1_path, mil_path, has_inner):
    b = pd.read_csv(b1_path, encoding="utf-8-sig")
    m = pd.read_csv(mil_path, encoding="utf-8-sig")
    keys = ["animal_id", "label"] + (["inner_fold"] if has_inner else [])
    if b.duplicated(keys).any() or m.duplicated(keys).any():
        raise ValueError(f"Duplicate rows in paired input: {b1_path} / {mil_path}")
    d = b.merge(m, on=keys, suffixes=("_b1", "_mil"), validate="one_to_one")
    if len(d) != len(b) or len(d) != len(m):
        raise ValueError(f"B1/MIL4 animal or label mismatch: {b1_path} / {mil_path}")
    return d.sort_values("animal_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-root", required=True)
    ap.add_argument("--mil-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    b1_root, mil_root, out = map(Path, (args.b1_root, args.mil_root, args.output))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing result: {out}")
    out.mkdir(parents=True, exist_ok=True)

    predictions, fold_rows, candidates_log = [], [], []
    for fold in range(1, 6):
        inner = paired(b1_root / f"outer_{fold}/inner_oof.csv",
                       mil_root / f"outer_{fold}/inner_oof.csv", True)
        y = inner.label.to_numpy(dtype=int)
        pb = inner.probability_b1.to_numpy(dtype=float)
        pm = inner.probability_mil.to_numpy(dtype=float)
        inner_fold = inner.inner_fold.to_numpy(dtype=int)
        b1_threshold = choose_threshold(y, pb)["threshold"]
        base = metrics(y, pb, b1_threshold)
        base["auc"] = float(np.mean([
            roc_auc_score(y[inner_fold == inner_k], pb[inner_fold == inner_k])
            for inner_k in sorted(np.unique(inner_fold))
        ]))
        fold_candidates = []

        for lam in LAMBDAS:
            raw = (1.0 - lam) * pb + lam * pm
            calibrated_cf = crossfit_platt(raw, y, inner_fold)
            selected = choose_threshold(y, calibrated_cf)
            # Compare AUC foldwise so fold-specific calibrators do not create
            # artificial cross-fold ranking changes.
            aucs = []
            for inner_k in sorted(np.unique(inner_fold)):
                mask = inner_fold == inner_k
                aucs.append(roc_auc_score(y[mask], calibrated_cf[mask]))
            selected["auc"] = float(np.mean(aucs))
            record = {"outer_fold": fold, "lambda": float(lam), **selected}
            candidates_log.append({**record, "passes_all_four": bool(
                selected["accuracy"] > base["accuracy"] + 1e-12 and
                selected["balanced_accuracy"] > base["balanced_accuracy"] + 1e-12 and
                selected["auc"] > base["auc"] + 1e-12 and
                selected["brier"] < base["brier"] - 1e-12)})
            if candidates_log[-1]["passes_all_four"]:
                fold_candidates.append(record)

        if fold_candidates:
            choice = max(fold_candidates, key=lambda r: (
                r["balanced_accuracy"], r["accuracy"], r["auc"], -r["brier"],
                -r["lambda"]))
            lam = choice["lambda"]
            raw_i = (1.0 - lam) * pb + lam * pm
            calibrator = fit_platt(raw_i, y)
            threshold = choice["threshold"]
            mode = "fused_platt"
        else:
            # Honest static-anchor fallback: do not claim the requested gain.
            lam, calibrator, threshold, mode = 0.0, None, b1_threshold, "raw_b1_fallback"

        test = paired(b1_root / f"outer_{fold}/test_predictions.csv",
                      mil_root / f"outer_{fold}/test_predictions.csv", False)
        yt = test.label.to_numpy(dtype=int)
        b1_test = test.probability_b1.to_numpy(dtype=float)
        mil_test = test.probability_mil.to_numpy(dtype=float)
        raw_test = (1.0 - lam) * b1_test + lam * mil_test
        final_p = raw_test if calibrator is None else apply_platt(calibrator, raw_test)
        test_metrics = metrics(yt, final_p, threshold)
        result = pd.DataFrame({
            "animal_id": test.animal_id, "label": yt, "outer_fold": fold,
            "b1_probability": b1_test, "mil4_probability": mil_test,
            "fused_probability": final_p,
            "prediction": (final_p >= threshold).astype(int),
            "threshold": threshold, "selected_lambda": lam,
            "selection_mode": mode,
        })
        result.to_csv(out / f"outer_{fold}_test_predictions.csv", index=False,
                      encoding="utf-8-sig")
        predictions.append(result)
        fold_rows.append({"outer_fold": fold, "baseline_inner": base,
                          "selected_lambda": lam, "selection_mode": mode,
                          **{f"test_{k}": v for k, v in test_metrics.items()},
                          "n_feasible_inner_candidates": len(fold_candidates)})

    allp = pd.concat(predictions, ignore_index=True)
    if len(allp) != 128 or not allp.animal_id.is_unique:
        raise ValueError("Expected exactly 128 unique outer-test animals")
    allp.to_csv(out / "all_outer_test_predictions.csv", index=False,
                encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out / "outer_fold_summary.csv", index=False)
    pd.DataFrame(candidates_log).to_csv(out / "inner_candidate_scores.csv", index=False)
    y = allp.label.to_numpy(dtype=int)
    p = allp.fused_probability.to_numpy(dtype=float)
    q = allp.prediction.to_numpy(dtype=int)
    summary = {
        "method": "B1_MIL4_all_four_metrics_inner_guarded",
        "outer_folds": 5, "inner_folds": 4, "n_repeats": 1,
        "animal_denominator": int(allp.animal_id.nunique()),
        "selection": "cross-fitted inner Platt; strict ACC, BACC, AUC improvement and Brier reduction vs raw B1; otherwise raw B1 fallback",
        "outer_test_used_for_selection": False,
        "per_animal_outer_predictions": True,
        "selected_modes": [r["selection_mode"] for r in fold_rows],
        "selected_lambdas": [r["selected_lambda"] for r in fold_rows],
        "pooled_metrics": {
            "accuracy": float(accuracy_score(y, q)),
            "balanced_accuracy": float(balanced_accuracy_score(y, q)),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
        },
        "n_inner_feasible_folds": int(sum(r["n_feasible_inner_candidates"] > 0
                                           for r in fold_rows)),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
