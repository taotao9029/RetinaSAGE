"""Nested exploratory B1 anchor + simple four-frame complementary features.

All candidate feature types, regularization, fusion weights and thresholds are
selected from the outer-training animals' existing four-fold OOF predictions.
Outer-test outcomes are used only for the final report.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, recall_score, roc_auc_score)

from train_b1_V2 import cached_frame_bundle, quality_select_indices


KINDS = ("uniform4_mean", "quality4_mean", "quality4_delta")
CS = (0.01, 0.1, 1.0, 10.0)
ALPHAS = (0.10, 0.25, 0.50, 0.75)
THRESHOLDS = np.arange(0.01, 1.001, 0.01)


@lru_cache(maxsize=512)
def features(cache_path: str, kind: str) -> np.ndarray:
    vectors, quality = cached_frame_bundle(cache_path)
    global_features = vectors[:, -384:]
    n = len(global_features)
    if kind == "uniform4_mean":
        indices = np.unique(np.linspace(0, n - 1, min(4, n)).round().astype(int))
    else:
        indices = quality_select_indices(quality, 4)
    mean = global_features[indices].mean(axis=0)
    return (mean - global_features[0] if kind == "quality4_delta" else mean).astype(np.float32)


def fit_predict(train, valid, kind, c):
    x = np.stack([features(str(path), kind) for path in train.cache_path])
    z = np.stack([features(str(path), kind) for path in valid.cache_path])
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.0
    model = LogisticRegression(C=c, max_iter=2000, solver="liblinear")
    model.fit((x - mean) / std, train.label.to_numpy(int))
    return model.predict_proba((z - mean) / std)[:, 1]


def score(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "sensitivity": float(recall_score(y, pred, pos_label=1)),
            "specificity": float(recall_score(y, pred, pos_label=0)),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "threshold": float(threshold)}


def choose_threshold(y, p, acc_floor, sensitivity_floor):
    choices = []
    for threshold in THRESHOLDS:
        result = score(y, p, threshold)
        if (result["accuracy"] + 1e-12 >= acc_floor and
                result["sensitivity"] + 1e-12 >= sensitivity_floor):
            choices.append(result)
    if not choices:
        return None
    return max(choices, key=lambda x: (
        x["balanced_accuracy"], x["accuracy"], x["auc"],
        -x["brier"], -abs(x["threshold"] - 0.5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-root", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--brier-tol", type=float, default=0.005)
    args = parser.parse_args()
    root, out = Path(args.b1_root), Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {out}")
    manifest = pd.read_csv(args.feature_manifest, encoding="utf-8-sig")
    if len(manifest) != 128 or not manifest.animal_id.is_unique:
        raise ValueError("Expected exactly 128 unique manifest animals")
    if not manifest.cache_path.map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError("Feature manifest has missing cache paths")
    manifest = manifest.set_index("animal_id", drop=False)
    out.mkdir(parents=True, exist_ok=True)
    folds, summaries, candidates = [], [], []

    for outer in range(1, 6):
        inner = pd.read_csv(root / f"outer_{outer}/inner_oof.csv",
                            encoding="utf-8-sig")
        test = pd.read_csv(root / f"outer_{outer}/test_predictions.csv",
                           encoding="utf-8-sig")
        if (inner.animal_id.duplicated().any() or test.animal_id.duplicated().any()
                or set(inner.animal_id) & set(test.animal_id)):
            raise ValueError(f"Invalid outer split {outer}")
        if not set(inner.animal_id).union(test.animal_id) == set(manifest.index):
            raise ValueError(f"Outer split {outer} does not cover the manifest")
        inner = inner.join(manifest[["cache_path"]], on="animal_id", validate="one_to_one")
        test = test.join(manifest[["cache_path"]], on="animal_id", validate="one_to_one")
        if not np.array_equal(inner.label.to_numpy(int),
                              manifest.loc[inner.animal_id, "label"].to_numpy(int)):
            raise ValueError(f"Label mismatch in outer fold {outer}")
        y, pb = inner.label.to_numpy(int), inner.probability.to_numpy(float)
        base_threshold = float(test.threshold.iloc[0])
        base = score(y, pb, base_threshold)
        feasible = [{"kind": "b1_only", "c": 0.0, "alpha": 0.0,
                     "probability": pb, **base}]

        for kind in KINDS:
            for c in CS:
                pm = np.empty(len(inner), dtype=float)
                for fold in sorted(inner.inner_fold.unique()):
                    fit = inner[inner.inner_fold != fold]
                    valid = inner[inner.inner_fold == fold]
                    pm[valid.index.to_numpy()] = fit_predict(fit, valid, kind, c)
                for alpha in ALPHAS:
                    p = (1 - alpha) * pb + alpha * pm
                    chosen = choose_threshold(y, p, base["accuracy"],
                                              base["sensitivity"])
                    if chosen is None or chosen["brier"] > base["brier"] + args.brier_tol:
                        continue
                    record = {"kind": kind, "c": c, "alpha": alpha,
                              "probability": p, **chosen}
                    feasible.append(record)
                    candidates.append({"outer_fold": outer, **{
                        k: v for k, v in record.items() if k != "probability"}})
        best = max(feasible, key=lambda z: (
            z["balanced_accuracy"], z["accuracy"], z["auc"],
            -z["brier"], -z["alpha"]))
        if best["kind"] == "b1_only":
            test_p = test.probability.to_numpy(float)
        else:
            pm = fit_predict(inner, test, best["kind"], best["c"])
            test_p = (1 - best["alpha"]) * test.probability.to_numpy(float) + best["alpha"] * pm
        pred = (test_p >= best["threshold"]).astype(int)
        result = pd.DataFrame({"animal_id": test.animal_id, "label": test.label,
                               "outer_fold": outer,
                               "b1_probability": test.probability,
                               "probability": test_p, "prediction": pred,
                               "threshold": best["threshold"],
                               "selected_kind": best["kind"],
                               "selected_c": best["c"],
                               "selected_alpha": best["alpha"]})
        result.to_csv(out / f"outer_{outer}_test_predictions.csv", index=False,
                      encoding="utf-8-sig")
        folds.append(result)
        summaries.append({"outer_fold": outer, "selected_kind": best["kind"],
                          "selected_c": best["c"],
                          "selected_alpha": best["alpha"],
                          "inner_b1_accuracy": base["accuracy"],
                          "inner_b1_balanced_accuracy": base["balanced_accuracy"],
                          "inner_selected_accuracy": best["accuracy"],
                          "inner_selected_balanced_accuracy": best["balanced_accuracy"],
                          **{f"test_{k}": v for k, v in score(
                              test.label.to_numpy(int), test_p, best["threshold"]).items()}})
        print(f"outer={outer} selected={best['kind']} C={best['c']} alpha={best['alpha']}",
              flush=True)

    allp = pd.concat(folds, ignore_index=True)
    if len(allp) != 128 or not allp.animal_id.is_unique:
        raise ValueError("Outer-test predictions must contain 128 unique animals")
    allp.to_csv(out / "all_outer_test_predictions.csv", index=False,
                encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out / "outer_fold_summary.csv", index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame(candidates).to_csv(out / "inner_feasible_candidates.csv",
                                    index=False, encoding="utf-8-sig")
    report = {"method": "B1_anchor_plus_simple_four_frame_logistic_probe",
              "feature_types": KINDS, "c_grid": CS, "alpha_grid": ALPHAS,
              "outer_folds": 5, "inner_folds": 4, "n_repeats": 1,
              "animal_denominator": 128,
              "selection": "inner OOF ACC and sensitivity >= B1; Brier <= B1 + tolerance; maximize balanced accuracy, ACC, AUC, lower Brier and alpha",
              "brier_tolerance": args.brier_tol,
              "no_outer_test_used_for_selection": True,
              "accuracy": float(accuracy_score(allp.label, allp.prediction)),
              "balanced_accuracy": float(balanced_accuracy_score(allp.label, allp.prediction)),
              "auc": float(roc_auc_score(allp.label, allp.probability)),
              "brier": float(brier_score_loss(allp.label, allp.probability))}
    (out / "run_manifest.json").write_text(json.dumps(report, ensure_ascii=False,
                                                   indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
