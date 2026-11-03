"""B1-anchored MIL-4 fusion followed by one final inner-only calibration.

Unlike separate calibration of B1 and MIL-4, this script first forms a raw
constrained fusion and then calibrates that final score.  Platt calibration is
monotonic, so it preserves the selected fusion's ranking/AUC.
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
THRESHOLDS = np.arange(0.05, 0.951, 0.01)


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def metric(y, p, t):
    pred = (p >= t).astype(int)
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(t),
    }


def fit_platt(p, y):
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    m.fit(logit(p).reshape(-1, 1), np.asarray(y, dtype=int))
    return m


def apply_platt(m, p):
    return m.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def choose_threshold(y, p, acc_floor=None, brier_floor=None):
    choices = []
    for t in THRESHOLDS:
        m = metric(y, p, t)
        if acc_floor is not None and m["accuracy"] < acc_floor:
            continue
        if brier_floor is not None and m["brier"] > brier_floor:
            continue
        choices.append(m)
    if not choices:
        return None
    return max(choices, key=lambda x: (x["balanced_accuracy"], x["accuracy"],
                                       x["auc"], -x["brier"],
                                       -abs(x["threshold"] - 0.5)))


def pair(b1, mil, where):
    b1 = b1.sort_values("animal_id").reset_index(drop=True)
    mil = mil.sort_values("animal_id").reset_index(drop=True)
    if not np.array_equal(b1.animal_id, mil.animal_id):
        raise ValueError(f"{where}: animal IDs differ")
    if not np.array_equal(b1.label, mil.label):
        raise ValueError(f"{where}: labels differ")
    return b1, mil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-root", default="runs/B1_B5/b1")
    ap.add_argument("--s3-root", default="runs/B1_B5/s1_s3/s3_16")
    ap.add_argument("--output", default="runs/B1_B5/b1_s3_16_final_calibration")
    ap.add_argument("--brier-tol", type=float, default=0.005)
    args = ap.parse_args()
    b1_root, s3_root, out = map(Path, (args.b1_root, args.s3_root, args.output))
    out.mkdir(parents=True, exist_ok=True)
    summaries, predictions = [], []

    for fold in range(1, 6):
        bi = pd.read_csv(b1_root / f"outer_{fold}/inner_oof.csv", encoding="utf-8-sig")
        mi = pd.read_csv(s3_root / f"outer_{fold}/inner_oof.csv", encoding="utf-8-sig")
        bi, mi = pair(bi, mi, f"outer_{fold}/inner")
        y = bi.label.to_numpy(int)
        pb, pm = bi.probability.to_numpy(float), mi.probability.to_numpy(float)

        # B1 baseline threshold and guardrails are fixed from inner OOF only.
        b1_candidates = [metric(y, pb, t) for t in THRESHOLDS]
        b1_base = max(b1_candidates, key=lambda x: (x["balanced_accuracy"],
                                                     x["accuracy"],
                                                     -abs(x["threshold"] - 0.5)))
        selected = []
        for lam in LAMBDAS:
            raw_i = (1 - lam) * pb + lam * pm
            calibrator = fit_platt(raw_i, y)
            cal_i = apply_platt(calibrator, raw_i)
            chosen = choose_threshold(
                y, cal_i,
                acc_floor=b1_base["accuracy"],
                brier_floor=b1_base["brier"] + args.brier_tol,
            )
            if chosen is not None:
                selected.append({"lambda": float(lam), **chosen, "calibrator": calibrator})

        fallback = False
        if selected:
            best = max(selected, key=lambda x: (x["balanced_accuracy"], x["auc"],
                                                 -x["brier"], -x["lambda"],
                                                 -abs(x["threshold"] - 0.5)))
        else:
            fallback = True
            raw_i = pb
            best = {"lambda": 0.0, **b1_base, "calibrator": fit_platt(raw_i, y)}

        bt = pd.read_csv(b1_root / f"outer_{fold}/test_predictions.csv", encoding="utf-8-sig")
        mt = pd.read_csv(s3_root / f"outer_{fold}/test_predictions.csv", encoding="utf-8-sig")
        bt, mt = pair(bt, mt, f"outer_{fold}/test")
        yt = bt.label.to_numpy(int)
        raw_t = (1 - best["lambda"]) * bt.probability.to_numpy(float) + best["lambda"] * mt.probability.to_numpy(float)
        cal_t = apply_platt(best["calibrator"], raw_t)
        test_m = metric(yt, cal_t, best["threshold"])
        test_raw_auc = float(roc_auc_score(yt, raw_t))
        result = pd.DataFrame({
            "animal_id": bt.animal_id, "label": yt,
            "b1_probability": bt.probability, "s3_16_probability": mt.probability,
            "raw_fused_probability": raw_t, "fused_probability": cal_t,
            "prediction": (cal_t >= best["threshold"]).astype(int),
            "outer_fold": fold, "selected_lambda": best["lambda"],
            "threshold": best["threshold"], "calibration": "final_platt",
        })
        result.to_csv(out / f"outer_{fold}_test_predictions.csv", index=False, encoding="utf-8-sig")
        summaries.append({
            "outer_fold": fold, "selected_lambda": best["lambda"],
            "threshold": best["threshold"], "fallback_to_b1": fallback,
            "inner_b1_accuracy": b1_base["accuracy"],
            "inner_b1_balanced_accuracy": b1_base["balanced_accuracy"],
            "inner_b1_brier": b1_base["brier"],
            "inner_selected_accuracy": best["accuracy"],
            "inner_selected_balanced_accuracy": best["balanced_accuracy"],
            "inner_selected_brier": best["brier"],
            "test_raw_auc": test_raw_auc,
            **{f"test_{k}": v for k, v in test_m.items()},
        })
        predictions.append(result)

    summary = pd.DataFrame(summaries)
    allp = pd.concat(predictions, ignore_index=True)
    summary.to_csv(out / "outer_fold_summary.csv", index=False, encoding="utf-8-sig")
    allp.to_csv(out / "all_outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "method": "b1_plus_s3_16_raw_constrained_fusion_then_final_platt",
        "outer_folds": [1, 2, 3, 4, 5], "n_outer_folds": 5,
        "inner_folds": 4, "n_repeats": 1,
        "animal_denominator": int(allp.animal_id.nunique()),
        "lambda_grid": LAMBDAS, "brier_abs_tol": args.brier_tol,
        "selection": "inner OOF: final calibrated fusion ACC >= B1 and Brier <= B1+tol; maximize BACC, then AUC, lower Brier, smaller lambda",
        "calibration": "one Platt calibrator fitted to each candidate final raw fusion using inner OOF only",
        "threshold_selection": "inner-only threshold grid 0.05..0.95",
        "no_outer_test_used_for_selection": True,
        "per_animal_outer_prediction": True,
        "mean_outer_auc": float(summary.test_auc.mean()),
        "mean_outer_raw_auc": float(summary.test_raw_auc.mean()),
        "mean_outer_accuracy": float(summary.test_accuracy.mean()),
        "mean_outer_balanced_accuracy": float(summary.test_balanced_accuracy.mean()),
        "mean_outer_brier": float(summary.test_brier.mean()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
