"""Nested constrained B1 + MIL-4 fusion with inner-only Platt calibration.

For each outer fold, the matching inner OOF predictions are used to fit the
optional Platt calibrators and select lambda/threshold.  The outer test set
is used once for final reporting only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)


LAMBDAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
CALIBRATIONS = ["none", "platt"]
THRESHOLDS = np.arange(0.05, 0.951, 0.01)


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def fit_platt(p, y):
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(logit(p).reshape(-1, 1), np.asarray(y, dtype=int))
    return model


def apply_platt(model, p):
    return model.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def pred_at(p, threshold):
    return (np.asarray(p) >= float(threshold)).astype(int)


def metric_dict(y, p, threshold):
    pred = pred_at(p, threshold)
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "ap": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
    }


def best_threshold(y, p, prefer_acc=True):
    candidates = []
    for t in THRESHOLDS:
        m = metric_dict(y, p, t)
        candidates.append((m["balanced_accuracy"], m["accuracy"],
                           -abs(float(t) - 0.5), float(t)))
    if prefer_acc:
        return max(candidates)[-1]
    return max(candidates, key=lambda x: (x[0], x[2]))[-1]


def fusion(pb, pm, lam, mode="probability"):
    if mode == "logit":
        return 1.0 / (1.0 + np.exp(-((1.0 - lam) * logit(pb) + lam * logit(pm))))
    return (1.0 - lam) * np.asarray(pb, dtype=float) + lam * np.asarray(pm, dtype=float)


def validate_pair(b1, mil, where):
    b1 = b1.sort_values("animal_id").reset_index(drop=True)
    mil = mil.sort_values("animal_id").reset_index(drop=True)
    if not np.array_equal(b1.animal_id.to_numpy(), mil.animal_id.to_numpy()):
        raise ValueError(f"{where}: animal order differs")
    if not np.array_equal(b1.label.to_numpy(), mil.label.to_numpy()):
        raise ValueError(f"{where}: labels differ")
    return b1, mil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-root", default="runs/B1_B5/b1")
    ap.add_argument("--mil-root", default="runs/B1_B5/quality_mil/mil_4")
    ap.add_argument("--output", default="runs/B1_B5/b1_mil4_constrained_calibrated")
    ap.add_argument("--fusion", choices=["probability", "logit"], default="logit")
    ap.add_argument("--brier-abs-tol", type=float, default=0.005)
    ap.add_argument("--acc-abs-tol", type=float, default=0.0)
    args = ap.parse_args()

    b1_root, mil_root, output = map(Path, (args.b1_root, args.mil_root, args.output))
    output.mkdir(parents=True, exist_ok=True)
    summaries, all_predictions = [], []

    for fold in range(1, 6):
        b1_i = pd.read_csv(b1_root / f"outer_{fold}" / "inner_oof.csv", encoding="utf-8-sig")
        mil_i = pd.read_csv(mil_root / f"outer_{fold}" / "inner_oof.csv", encoding="utf-8-sig")
        b1_i, mil_i = validate_pair(b1_i, mil_i, f"outer_{fold}/inner")
        y_i = b1_i.label.to_numpy(dtype=int)
        pb_i = b1_i.probability.to_numpy(float)
        pm_i = mil_i.probability.to_numpy(float)

        b1_t0 = best_threshold(y_i, pb_i)
        b1_m = metric_dict(y_i, pb_i, b1_t0)
        candidates = []

        for calibration in CALIBRATIONS:
            cb = cm = None
            if calibration == "platt":
                cb = fit_platt(pb_i, y_i)
                cm = fit_platt(pm_i, y_i)
                pb_c = apply_platt(cb, pb_i)
                pm_c = apply_platt(cm, pm_i)
            else:
                pb_c, pm_c = pb_i, pm_i

            for lam in LAMBDAS:
                p_i = fusion(pb_c, pm_c, lam, args.fusion)
                for threshold in THRESHOLDS:
                    m = metric_dict(y_i, p_i, threshold)
                    acc_ok = m["accuracy"] >= b1_m["accuracy"] - args.acc_abs_tol
                    brier_ok = m["brier"] <= b1_m["brier"] + args.brier_abs_tol
                    if acc_ok and brier_ok:
                        candidates.append({
                            "calibration": calibration, "lambda": float(lam),
                            "threshold": float(threshold), **m,
                        })

        fallback = False
        if candidates:
            # Guardrails first; optimize BACC, then AUC, then Brier, then the
            # smallest deviation from B1 (calibration/lam) for stability.
            selected = max(candidates, key=lambda x: (
                x["balanced_accuracy"], x["auc"], -x["brier"],
                -float(x["lambda"]), -abs(x["threshold"] - b1_t0),
                1 if x["calibration"] == "none" else 0))
        else:
            fallback = True
            selected = {
                "calibration": "none", "lambda": 0.0,
                "threshold": float(b1_t0), **b1_m,
            }

        b1_t = pd.read_csv(b1_root / f"outer_{fold}" / "test_predictions.csv", encoding="utf-8-sig")
        mil_t = pd.read_csv(mil_root / f"outer_{fold}" / "test_predictions.csv", encoding="utf-8-sig")
        b1_t, mil_t = validate_pair(b1_t, mil_t, f"outer_{fold}/test")
        y_t = b1_t.label.to_numpy(dtype=int)
        pb_t = b1_t.probability.to_numpy(float)
        pm_t = mil_t.probability.to_numpy(float)

        if selected["calibration"] == "platt":
            pb_t = apply_platt(cb, pb_t)
            pm_t = apply_platt(cm, pm_t)
        p_t = fusion(pb_t, pm_t, selected["lambda"], args.fusion)
        m_t = metric_dict(y_t, p_t, selected["threshold"])
        result = pd.DataFrame({
            "animal_id": b1_t.animal_id,
            "label": y_t,
            "b1_probability": b1_t.probability,
            "mil4_probability": mil_t.probability,
            "calibrated_b1_probability": pb_t,
            "calibrated_mil4_probability": pm_t,
            "fused_probability": p_t,
            "prediction": pred_at(p_t, selected["threshold"]),
            "outer_fold": fold,
            "calibration": selected["calibration"],
            "selected_lambda": selected["lambda"],
            "threshold": selected["threshold"],
        })
        result.to_csv(output / f"outer_{fold}_test_predictions.csv", index=False, encoding="utf-8-sig")
        summary = {
            "outer_fold": fold,
            "calibration": selected["calibration"],
            "selected_lambda": selected["lambda"],
            "threshold": selected["threshold"],
            "fallback_to_b1": fallback,
            "inner_b1_accuracy": b1_m["accuracy"],
            "inner_b1_balanced_accuracy": b1_m["balanced_accuracy"],
            "inner_b1_brier": b1_m["brier"],
            "inner_selected_accuracy": selected["accuracy"],
            "inner_selected_balanced_accuracy": selected["balanced_accuracy"],
            "inner_selected_brier": selected["brier"],
        }
        summary.update({f"test_{k}": v for k, v in m_t.items()})
        summaries.append(summary)
        all_predictions.append(result)

    summary = pd.DataFrame(summaries)
    allp = pd.concat(all_predictions, ignore_index=True)
    summary.to_csv(output / "outer_fold_summary.csv", index=False, encoding="utf-8-sig")
    allp.to_csv(output / "all_outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "method": "b1_plus_mil4_constrained_inner_calibrated",
        "outer_folds": [1, 2, 3, 4, 5], "n_outer_folds": 5,
        "inner_folds": 4, "n_repeats": 1,
        "animal_denominator": int(allp.animal_id.nunique()),
        "source_b1": str(b1_root.resolve()), "source_mil4": str(mil_root.resolve()),
        "calibration_candidates": CALIBRATIONS,
        "fusion": args.fusion,
        "lambda_grid": LAMBDAS,
        "selection": "inner OOF: ACC >= B1-acc_tol and Brier <= B1+brier_tol; maximize BACC, then AUC, lower Brier, smaller lambda",
        "acc_abs_tol": args.acc_abs_tol,
        "brier_abs_tol": args.brier_abs_tol,
        "threshold_selection": "inner-only; threshold grid 0.05..0.95",
        "no_outer_test_used_for_selection": True,
        "per_animal_outer_prediction": True,
        "mean_outer_auc": float(summary.test_auc.mean()),
        "mean_outer_accuracy": float(summary.test_accuracy.mean()),
        "mean_outer_balanced_accuracy": float(summary.test_balanced_accuracy.mean()),
        "mean_outer_brier": float(summary.test_brier.mean()),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
