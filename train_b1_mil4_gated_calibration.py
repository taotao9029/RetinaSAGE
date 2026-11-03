"""Exploratory B1-anchored MIL-4 disagreement gate with inner-only selection.

The fixed candidate grid is evaluated using cross-fitted Platt probabilities
within each outer-training set. Outer-test labels are read only after the
gate, calibration, and decision threshold have been selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, recall_score, roc_auc_score)


MARGINS = (0.10, 0.20, 0.35)
STRENGTHS = (0.05, 0.10, 0.20)
THRESHOLDS = np.arange(0.05, 0.951, 0.01)


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def platt_fit(p, y):
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(logit(p).reshape(-1, 1), y)
    return model


def platt_predict(model, p):
    return model.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def crossfit_platt(p, y, inner_fold):
    result = np.empty(len(y), dtype=float)
    for fold in sorted(np.unique(inner_fold)):
        train, valid = inner_fold != fold, inner_fold == fold
        if len(np.unique(y[train])) != 2:
            raise ValueError(f"Calibration training fold {fold} lacks a class")
        result[valid] = platt_predict(platt_fit(p[train], y[train]), p[valid])
    return result


def stats(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, pos_label=1)),
        "specificity": float(recall_score(y, pred, pos_label=0)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
    }


def best_threshold(y, p, acc_floor=-np.inf, sensitivity_floor=-np.inf):
    options = []
    for threshold in THRESHOLDS:
        result = stats(y, p, threshold)
        if result["accuracy"] + 1e-12 < acc_floor:
            continue
        if result["sensitivity"] + 1e-12 < sensitivity_floor:
            continue
        options.append(result)
    if not options:
        return None
    return max(options, key=lambda v: (
        v["balanced_accuracy"], v["accuracy"], -v["brier"],
        -abs(v["threshold"] - 0.5)))


def paired(b1_path, mil_path, inner):
    a = pd.read_csv(b1_path, encoding="utf-8-sig")
    b = pd.read_csv(mil_path, encoding="utf-8-sig")
    keys = ["animal_id", "label"] + (["inner_fold"] if inner else [])
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError(f"Duplicate paired rows: {b1_path}, {mil_path}")
    frame = a.merge(b, on=keys, suffixes=("_b1", "_mil"), validate="one_to_one")
    if len(frame) != len(a) or len(frame) != len(b):
        raise ValueError(f"B1/MIL-4 rows differ: {b1_path}, {mil_path}")
    return frame.sort_values("animal_id").reset_index(drop=True)


def gated_score(pb, pm, b1_threshold, mil_threshold, margin, strength):
    pb, pm = clip(pb), clip(pm)
    if strength == 0.0:
        return pb, np.zeros(len(pb), dtype=bool)
    b1_positive = pb >= b1_threshold
    mil_positive = pm >= mil_threshold
    distance = np.abs(pb - b1_threshold)
    gate = (b1_positive != mil_positive) & (distance <= margin)
    correction = strength * np.maximum(0.0, 1.0 - distance / margin)
    score = pb + gate * correction * np.where(mil_positive, 1.0, -1.0)
    return clip(score), gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-root", required=True)
    parser.add_argument("--mil-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--brier-tol", type=float, default=0.005)
    args = parser.parse_args()
    b1_root, mil_root, out = map(Path, (args.b1_root, args.mil_root, args.output))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing result: {out}")
    out.mkdir(parents=True, exist_ok=True)
    all_predictions, all_summaries, all_candidates = [], [], []

    for outer_fold in range(1, 6):
        inner = paired(b1_root / f"outer_{outer_fold}/inner_oof.csv",
                       mil_root / f"outer_{outer_fold}/inner_oof.csv", True)
        y = inner.label.to_numpy(dtype=int)
        pb = inner.probability_b1.to_numpy(dtype=float)
        pm = inner.probability_mil.to_numpy(dtype=float)
        inner_fold = inner.inner_fold.to_numpy(dtype=int)
        b1_threshold = best_threshold(y, pb)["threshold"]
        mil_threshold = best_threshold(y, pm)["threshold"]
        baseline_cal = crossfit_platt(pb, y, inner_fold)
        baseline = best_threshold(y, baseline_cal)

        candidates = []
        for margin, strength in [(0.0, 0.0)] + [
            (m, s) for m in MARGINS for s in STRENGTHS
        ]:
            score, gate = gated_score(pb, pm, b1_threshold, mil_threshold,
                                      margin, strength)
            calibrated = crossfit_platt(score, y, inner_fold)
            selected = best_threshold(
                y, calibrated, acc_floor=baseline["accuracy"],
                sensitivity_floor=baseline["sensitivity"])
            if selected is None:
                continue
            if selected["brier"] > baseline["brier"] + args.brier_tol + 1e-12:
                continue
            record = {"outer_fold": outer_fold, "margin": margin,
                      "strength": strength, "n_inner_gated": int(gate.sum()),
                      **selected}
            candidates.append(record)
            all_candidates.append(record)
        if not candidates:
            raise RuntimeError(f"No feasible candidate for outer fold {outer_fold}")
        choice = max(candidates, key=lambda v: (
            v["balanced_accuracy"], v["accuracy"], v["auc"],
            -v["brier"], -v["strength"], -v["margin"]))

        test = paired(b1_root / f"outer_{outer_fold}/test_predictions.csv",
                      mil_root / f"outer_{outer_fold}/test_predictions.csv", False)
        score_i, _ = gated_score(pb, pm, b1_threshold, mil_threshold,
                                 choice["margin"], choice["strength"])
        calibrator = platt_fit(score_i, y)
        score_t, gate_t = gated_score(
            test.probability_b1.to_numpy(dtype=float),
            test.probability_mil.to_numpy(dtype=float),
            b1_threshold, mil_threshold, choice["margin"], choice["strength"])
        final_p = platt_predict(calibrator, score_t)
        result = pd.DataFrame({
            "animal_id": test.animal_id, "label": test.label.astype(int),
            "outer_fold": outer_fold,
            "b1_probability": test.probability_b1,
            "mil4_probability": test.probability_mil,
            "raw_gated_score": score_t, "fused_probability": final_p,
            "prediction": (final_p >= choice["threshold"]).astype(int),
            "threshold": choice["threshold"], "gated": gate_t,
            "selected_margin": choice["margin"],
            "selected_strength": choice["strength"],
        })
        result.to_csv(out / f"outer_{outer_fold}_test_predictions.csv",
                      index=False, encoding="utf-8-sig")
        all_predictions.append(result)
        all_summaries.append({
            "outer_fold": outer_fold, "n_test": len(test),
            "b1_inner_threshold": b1_threshold,
            "mil_inner_threshold": mil_threshold,
            "baseline_crossfit_accuracy": baseline["accuracy"],
            "baseline_crossfit_sensitivity": baseline["sensitivity"],
            "baseline_crossfit_brier": baseline["brier"],
            "selected_margin": choice["margin"],
            "selected_strength": choice["strength"],
            "selected_threshold": choice["threshold"],
            "n_inner_gated": choice["n_inner_gated"],
            "n_test_gated": int(gate_t.sum()),
            "calibrator_coef": float(calibrator.coef_[0, 0]),
            "calibrator_intercept": float(calibrator.intercept_[0]),
            **{f"test_{k}": v for k, v in stats(test.label.to_numpy(int),
                                                   final_p, choice["threshold"]).items()},
        })

    predictions = pd.concat(all_predictions, ignore_index=True)
    if len(predictions) != 128 or not predictions.animal_id.is_unique:
        raise ValueError("Outer predictions must contain 128 unique animals")
    predictions.to_csv(out / "all_outer_test_predictions.csv", index=False,
                       encoding="utf-8-sig")
    pd.DataFrame(all_summaries).to_csv(out / "outer_fold_summary.csv", index=False,
                                        encoding="utf-8-sig")
    pd.DataFrame(all_candidates).to_csv(out / "inner_candidate_scores.csv", index=False,
                                          encoding="utf-8-sig")
    manifest = {
        "method": "symmetric_disagreement_gate_B1_anchor_MIL4_final_Platt",
        "source_b1": str(b1_root.resolve()), "source_mil4": str(mil_root.resolve()),
        "outer_folds": [1, 2, 3, 4, 5], "inner_folds": 4,
        "animal_denominator": 128, "n_repeats": 1,
        "gate": "only when B1 and MIL4 inner-threshold decisions disagree and B1 is within margin of its inner threshold; symmetric shrinkage toward MIL4 decision",
        "margin_grid": list(MARGINS), "strength_grid": list(STRENGTHS),
        "brier_tolerance": args.brier_tol,
        "selection": "cross-fitted inner Platt; inner ACC and sensitivity no lower than calibrated B1, Brier <= B1 + tolerance; maximize BACC, ACC, AUC, lower Brier, smaller correction",
        "final_calibration": "Platt fitted on all outer-training inner OOF scores only",
        "outer_test_used_for_selection": False,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                   indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()),
                      "accuracy": float(accuracy_score(predictions.label,
                                                       predictions.prediction)),
                      "balanced_accuracy": float(balanced_accuracy_score(
                          predictions.label, predictions.prediction)),
                      "auc": float(roc_auc_score(predictions.label,
                                                 predictions.fused_probability)),
                      "brier": float(brier_score_loss(predictions.label,
                                                      predictions.fused_probability)),
                      "selected_strengths": [r["selected_strength"] for r in all_summaries],
                      "gated_test_counts": [r["n_test_gated"] for r in all_summaries]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
