"""Strict nested B1 + quality-MIL-4 probability blend evaluation.

The B1 and MIL-4 models are already trained on the same outer/inner animal
splits.  Lambda is selected only from the matching inner OOF predictions of
each outer fold, then applied once to that fold's outer-test predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)


LAMBDAS = np.arange(0.0, 1.0001, 0.05)


def choose_threshold(y, p):
    best = None
    for t in np.arange(0.0, 1.0001, 0.01):
        pred = (p >= t).astype(int)
        key = (balanced_accuracy_score(y, pred), f1_score(y, pred, zero_division=0),
               -abs(float(t) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
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


def validate_pair(b1, mil, where):
    b1 = b1.sort_values("animal_id").reset_index(drop=True)
    mil = mil.sort_values("animal_id").reset_index(drop=True)
    if not np.array_equal(b1.animal_id.to_numpy(), mil.animal_id.to_numpy()):
        raise ValueError(f"{where}: B1 and MIL-4 animal order differs")
    if not np.array_equal(b1.label.to_numpy(), mil.label.to_numpy()):
        raise ValueError(f"{where}: B1 and MIL-4 labels differ")
    return b1, mil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-root", default="runs/B1_B5/b1")
    ap.add_argument("--mil-root", default="runs/B1_B5/quality_mil/mil_4")
    ap.add_argument("--output", default="runs/B1_B5/b1_plus_mil4_blend")
    args = ap.parse_args()
    b1_root, mil_root, output = map(Path, (args.b1_root, args.mil_root, args.output))
    output.mkdir(parents=True, exist_ok=True)
    fold_summaries, all_predictions = [], []

    for fold in range(1, 6):
        b1_i = pd.read_csv(b1_root / f"outer_{fold}" / "inner_oof.csv", encoding="utf-8-sig")
        mil_i = pd.read_csv(mil_root / f"outer_{fold}" / "inner_oof.csv", encoding="utf-8-sig")
        b1_i, mil_i = validate_pair(b1_i, mil_i, f"outer_{fold}/inner")
        y = b1_i.label.to_numpy(dtype=int)
        pb, pm = b1_i.probability.to_numpy(float), mil_i.probability.to_numpy(float)
        candidates = []
        for lam in LAMBDAS:
            pi = (1.0 - lam) * pb + lam * pm
            ti = choose_threshold(y, pi)
            mi = metrics(y, pi, ti)
            candidates.append((mi["auc"], mi["balanced_accuracy"],
                               mi["f1"], -float(lam), float(lam), ti))
        _, _, _, _, lam, threshold = max(candidates)

        b1_t = pd.read_csv(b1_root / f"outer_{fold}" / "test_predictions.csv", encoding="utf-8-sig")
        mil_t = pd.read_csv(mil_root / f"outer_{fold}" / "test_predictions.csv", encoding="utf-8-sig")
        b1_t, mil_t = validate_pair(b1_t, mil_t, f"outer_{fold}/test")
        y_t = b1_t.label.to_numpy(dtype=int)
        p_t = ((1.0 - lam) * b1_t.probability.to_numpy(float)
               + lam * mil_t.probability.to_numpy(float))
        result = pd.DataFrame({
            "animal_id": b1_t.animal_id,
            "label": y_t,
            "b1_probability": b1_t.probability,
            "mil4_probability": mil_t.probability,
            "blend_probability": p_t,
            "prediction": (p_t >= threshold).astype(int),
            "outer_fold": fold,
            "selected_lambda": lam,
            "threshold": threshold,
        })
        result.to_csv(output / f"outer_{fold}_test_predictions.csv", index=False, encoding="utf-8-sig")
        summary = {"outer_fold": fold, "selected_lambda": lam,
                   "inner_threshold": threshold}
        summary.update({f"test_{k}": v for k, v in metrics(y_t, p_t, threshold).items()})
        fold_summaries.append(summary)
        all_predictions.append(result)

    summary = pd.DataFrame(fold_summaries)
    summary.to_csv(output / "outer_fold_summary.csv", index=False, encoding="utf-8-sig")
    allp = pd.concat(all_predictions, ignore_index=True)
    allp.to_csv(output / "all_outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "method": "b1_plus_mil4_probability_blend",
        "outer_folds": [1, 2, 3, 4, 5], "n_outer_folds": 5,
        "inner_folds": 4, "n_repeats": 1, "animal_denominator": int(allp.animal_id.nunique()),
        "source_b1": str(b1_root.resolve()), "source_mil4": str(mil_root.resolve()),
        "selection": "lambda selected by matching inner OOF AUC; threshold selected by inner OOF balanced accuracy",
        "lambda_grid": LAMBDAS.tolist(), "no_outer_test_used_for_selection": True,
        "formula": "p_blend=(1-lambda)*p_B1+lambda*p_MIL4",
        "per_animal_outer_prediction": True,
        "mean_outer_auc": float(summary.test_auc.mean()),
        "mean_outer_accuracy": float(summary.test_accuracy.mean()),
        "mean_outer_balanced_accuracy": float(summary.test_balanced_accuracy.mean()),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
