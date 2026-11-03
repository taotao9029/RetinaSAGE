"""Paired animal-level comparison of B1 and B1+MIL-4 with equal calibration.

The confidence intervals resample the same animals for both models within
each label class. They describe sampling variability on this 128-animal
cohort, not independent-cohort generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)


def scores(y, prediction, raw_probability, calibrated_probability):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "brier": float(brier_score_loss(y, calibrated_probability)),
        "raw_auc": float(roc_auc_score(y, raw_probability)),
        "calibrated_auc": float(roc_auc_score(y, calibrated_probability)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-root", default="runs/B1_B5/b1_final_calibration_matched")
    ap.add_argument("--fusion-root", default="runs/B1_B5/b1_mil4_final_calibration")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    b1_root, fusion_root = Path(args.b1_root), Path(args.fusion_root)
    a = pd.read_csv(b1_root / "all_outer_test_predictions.csv", encoding="utf-8-sig")
    b = pd.read_csv(fusion_root / "all_outer_test_predictions.csv", encoding="utf-8-sig")
    keys = ["animal_id", "label", "outer_fold"]
    for name, frame in (("B1", a), ("fusion", b)):
        if frame.duplicated(keys).any():
            raise ValueError(f"{name}: repeated animal/fold prediction")
        if frame.animal_id.nunique() != len(frame):
            raise ValueError(f"{name}: animal appears in multiple outer folds")
    aligned = a.merge(b, on=keys, suffixes=("_b1", "_fusion"),
                      validate="one_to_one", how="inner")
    if len(aligned) != len(a) or len(aligned) != len(b):
        raise ValueError("B1 and fusion have different outer animal/label/fold assignments")
    aligned = aligned.sort_values("animal_id").reset_index(drop=True)
    y = aligned.label.to_numpy(dtype=int)

    def result(frame, side):
        return scores(frame.label.to_numpy(dtype=int),
                      frame[f"prediction_{side}"].to_numpy(dtype=int),
                      frame[f"raw_fused_probability_{side}"].to_numpy(dtype=float),
                      frame[f"fused_probability_{side}"].to_numpy(dtype=float))

    b1_scores, fusion_scores = result(aligned, "b1"), result(aligned, "fusion")
    deltas = {key: fusion_scores[key] - b1_scores[key] for key in b1_scores}

    rng = np.random.default_rng(args.seed)
    strata = [np.flatnonzero(y == label) for label in np.unique(y)]
    draws = {key: [] for key in deltas}
    for _ in range(args.bootstrap):
        idx = np.concatenate([rng.choice(stratum, size=len(stratum), replace=True)
                              for stratum in strata])
        sample = aligned.iloc[idx]
        x, z = result(sample, "b1"), result(sample, "fusion")
        for key in draws:
            draws[key].append(z[key] - x[key])
    intervals = {key: [float(v) for v in np.percentile(draws[key], [2.5, 97.5])]
                 for key in draws}

    aligned["b1_correct"] = (aligned.prediction_b1 == aligned.label)
    aligned["fusion_correct"] = (aligned.prediction_fusion == aligned.label)
    changes = aligned.loc[aligned.prediction_b1 != aligned.prediction_fusion,
                          ["animal_id", "label", "outer_fold", "prediction_b1",
                           "prediction_fusion", "b1_correct", "fusion_correct",
                           "fused_probability_b1", "fused_probability_fusion"]]
    changes.to_csv(b1_root / "paired_changed_predictions.csv", index=False,
                   encoding="utf-8-sig")

    fold_comparison = []
    for fold, subset in aligned.groupby("outer_fold", sort=True):
        bs, fs = result(subset, "b1"), result(subset, "fusion")
        fold_comparison.append({"outer_fold": int(fold), "n": int(len(subset)),
                                **{f"b1_{k}": v for k, v in bs.items()},
                                **{f"fusion_{k}": v for k, v in fs.items()},
                                **{f"delta_{k}": fs[k] - bs[k] for k in bs}})
    pd.DataFrame(fold_comparison).to_csv(b1_root / "paired_fold_comparison.csv",
                                         index=False, encoding="utf-8-sig")
    report = {
        "comparison": "B1+same final Platt versus B1+MIL4+same final Platt",
        "n_animals": int(len(aligned)),
        "n_outer_folds": int(aligned.outer_fold.nunique()),
        "b1": b1_scores, "fusion": fusion_scores,
        "delta_fusion_minus_b1": deltas,
        "paired_stratified_animal_bootstrap_95pct": intervals,
        "bootstrap_iterations": args.bootstrap, "bootstrap_seed": args.seed,
        "improved_animals": int((~aligned.b1_correct & aligned.fusion_correct).sum()),
        "worsened_animals": int((aligned.b1_correct & ~aligned.fusion_correct).sum()),
        "changed_animals": changes.animal_id.tolist(),
        "auc_definition": "raw_auc uses pre-calibration scores; calibrated_auc pools fold-specific calibrated scores",
        "limitation": "same cohort/splits; calibration fit and threshold selected on the same inner OOF predictions",
    }
    (b1_root / "paired_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
