"""Exploratory nested B1 anchor plus nonlinear frozen-feature complement.

The complement is an RBF-SVM on global, temporal and/or spatial DINO features.
Only inner OOF predictions select feature type, SVM regularization, blending,
and threshold. Outer-test labels are used only for the final report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


VARIANTS = ("global", "global_motion", "global_spatial", "global_motion_spatial")
COMPONENTS = (16, 32)
CS = (0.1, 1.0, 10.0, 100.0)
ALPHAS = (0.25, 0.50, 0.75, 1.0)
THRESHOLDS = np.arange(0.01, 1.001, 0.01)


def load_features(manifest):
    rows = {}
    for row in manifest.itertuples(index=False):
        payload = torch.load(row.cache_path, map_location="cpu", weights_only=False)
        if int(payload["video_id"]) != int(row.video_id) or int(payload["label"]) != int(row.label):
            raise ValueError(f"Cache video_id or label disagrees with manifest: {row.cache_path}")
        global_ = payload["global_features"].float().numpy()
        region = payload["region_features"].float().numpy()
        valid = payload["region_valid_mask"].bool().numpy()[0]
        if not valid.any():
            valid = np.ones(len(valid), dtype=bool)
        subset = region[0, valid]
        frame_ids = np.unique(np.linspace(0, len(global_) - 1,
                                           min(4, len(global_))).round().astype(int))
        motion = global_[frame_ids].mean(0) - global_[0]
        spatial = np.concatenate((subset.mean(0) - global_[0], subset.std(0)))
        first = global_[0]
        rows[str(row.animal_id)] = {
            "global": first,
            "global_motion": np.concatenate((first, motion)),
            "global_spatial": np.concatenate((first, spatial)),
            "global_motion_spatial": np.concatenate((first, motion, spatial)),
        }
    return rows


def make_model(n_components, c, seed):
    return make_pipeline(StandardScaler(),
                         PCA(n_components=n_components, svd_solver="randomized",
                             random_state=seed),
                         SVC(C=c, kernel="rbf", gamma="scale", probability=True,
                             random_state=seed))


def metrics(y, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "sensitivity": float(recall_score(y, prediction, pos_label=1)),
            "specificity": float(recall_score(y, prediction, pos_label=0)),
            "auc": float(roc_auc_score(y, probability)),
            "brier": float(brier_score_loss(y, probability)),
            "threshold": float(threshold)}


def choose_threshold(y, p, min_acc, min_sensitivity):
    pred = p[:, None] >= THRESHOLDS[None, :]
    positive = y == 1
    negative = ~positive
    acc = (pred == positive[:, None]).mean(0)
    sensitivity = pred[positive].mean(0)
    specificity = (~pred[negative]).mean(0)
    balanced = (sensitivity + specificity) / 2
    eligible = np.flatnonzero((acc + 1e-12 >= min_acc) &
                              (sensitivity + 1e-12 >= min_sensitivity))
    if not len(eligible):
        return None
    auc = float(roc_auc_score(y, p))
    brier = float(brier_score_loss(y, p))
    i = max(eligible, key=lambda j: (balanced[j], acc[j],
                                     -abs(THRESHOLDS[j] - 0.5)))
    return {"accuracy": float(acc[i]), "balanced_accuracy": float(balanced[i]),
            "sensitivity": float(sensitivity[i]),
            "specificity": float(specificity[i]), "auc": auc,
            "brier": brier, "threshold": float(THRESHOLDS[i])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-root", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--brier-tol", type=float, default=0.01)
    args = parser.parse_args()
    root, out = Path(args.b1_root), Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {out}")
    manifest = pd.read_csv(args.feature_manifest, encoding="utf-8-sig")
    if len(manifest) != 128 or not manifest.animal_id.is_unique:
        raise ValueError("Expected 128 unique animals")
    feature_bank = load_features(manifest)
    out.mkdir(parents=True, exist_ok=True)
    predictions, summaries, candidates = [], [], []

    for outer in range(1, 6):
        inner = pd.read_csv(root / f"outer_{outer}/inner_oof.csv", encoding="utf-8-sig")
        test = pd.read_csv(root / f"outer_{outer}/test_predictions.csv", encoding="utf-8-sig")
        if inner.animal_id.duplicated().any() or test.animal_id.duplicated().any():
            raise ValueError(f"Repeated animal in outer fold {outer}")
        if set(inner.animal_id) & set(test.animal_id):
            raise ValueError(f"Train/test overlap in outer fold {outer}")
        if set(inner.animal_id) | set(test.animal_id) != set(feature_bank):
            raise ValueError(f"Outer fold {outer} does not cover all animals")
        y, pb = inner.label.to_numpy(int), inner.probability.to_numpy(float)
        baseline = metrics(y, pb, float(test.threshold.iloc[0]))
        feasible = [{"variant": "b1_only", "components": 0, "c": 0.0,
                     "alpha": 0.0, **baseline}]

        for variant in VARIANTS:
            x = np.stack([feature_bank[str(a)][variant] for a in inner.animal_id])
            for components in COMPONENTS:
                oof = {c: np.empty(len(inner), dtype=float) for c in CS}
                for fold in sorted(inner.inner_fold.unique()):
                    train_idx = np.flatnonzero(inner.inner_fold.to_numpy() != fold)
                    val_idx = np.flatnonzero(inner.inner_fold.to_numpy() == fold)
                    for c in CS:
                        model = make_model(components, c, seed=42 + outer * 1000 + int(fold))
                        model.fit(x[train_idx], y[train_idx])
                        oof[c][val_idx] = model.predict_proba(x[val_idx])[:, 1]
                for c in CS:
                    for alpha in ALPHAS:
                        p = (1 - alpha) * pb + alpha * oof[c]
                        chosen = choose_threshold(y, p, baseline["accuracy"],
                                                  baseline["sensitivity"])
                        if chosen is None or chosen["brier"] > baseline["brier"] + args.brier_tol:
                            continue
                        row = {"variant": variant, "components": components,
                               "c": c, "alpha": alpha, **chosen}
                        feasible.append(row)
                        candidates.append({"outer_fold": outer, **row})

        best = max(feasible, key=lambda v: (
            v["balanced_accuracy"], v["accuracy"], v["auc"],
            -v["brier"], -v["alpha"], -v["components"]))
        if best["variant"] == "b1_only":
            test_p = test.probability.to_numpy(float)
        else:
            x_train = np.stack([feature_bank[str(a)][best["variant"]]
                                for a in inner.animal_id])
            x_test = np.stack([feature_bank[str(a)][best["variant"]]
                               for a in test.animal_id])
            model = make_model(best["components"], best["c"],
                               seed=42 + outer * 1000 + 99)
            model.fit(x_train, y)
            supplement = model.predict_proba(x_test)[:, 1]
            test_p = (1 - best["alpha"]) * test.probability.to_numpy(float) + best["alpha"] * supplement

        result = pd.DataFrame({"animal_id": test.animal_id, "label": test.label,
                               "outer_fold": outer, "b1_probability": test.probability,
                               "probability": test_p,
                               "prediction": (test_p >= best["threshold"]).astype(int),
                               "threshold": best["threshold"],
                               "selected_variant": best["variant"],
                               "selected_components": best["components"],
                               "selected_c": best["c"],
                               "selected_alpha": best["alpha"]})
        result.to_csv(out / f"outer_{outer}_test_predictions.csv", index=False,
                      encoding="utf-8-sig")
        predictions.append(result)
        summaries.append({"outer_fold": outer,
                          "selected_variant": best["variant"],
                          "selected_components": best["components"],
                          "selected_c": best["c"], "selected_alpha": best["alpha"],
                          "inner_b1_accuracy": baseline["accuracy"],
                          "inner_b1_balanced_accuracy": baseline["balanced_accuracy"],
                          "inner_selected_accuracy": best["accuracy"],
                          "inner_selected_balanced_accuracy": best["balanced_accuracy"],
                          **{f"test_{k}": v for k, v in metrics(
                              test.label.to_numpy(int), test_p, best["threshold"]).items()}})
        print(f"outer={outer} variant={best['variant']} components={best['components']} "
              f"C={best['c']} alpha={best['alpha']}", flush=True)

    allp = pd.concat(predictions, ignore_index=True)
    if len(allp) != 128 or not allp.animal_id.is_unique:
        raise ValueError("Expected exactly one outer prediction per animal")
    allp.to_csv(out / "all_outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out / "outer_fold_summary.csv", index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame(candidates).to_csv(out / "inner_feasible_candidates.csv",
                                    index=False, encoding="utf-8-sig")
    report = {"method": "B1_anchor_plus_RBF_SVM_frozen_feature_complement",
              "feature_variants": VARIANTS, "pca_components": COMPONENTS,
              "c_grid": CS, "alpha_grid": ALPHAS,
              "outer_folds": 5, "inner_folds": 4, "n_repeats": 1,
              "animal_denominator": 128, "brier_tolerance": args.brier_tol,
              "selection": "inner OOF only: ACC and sensitivity >= B1, Brier <= B1 + tolerance; maximize balanced accuracy, ACC, AUC, then lower Brier and complexity",
              "outer_test_used_for_selection": False,
              "accuracy": float(accuracy_score(allp.label, allp.prediction)),
              "balanced_accuracy": float(balanced_accuracy_score(allp.label, allp.prediction)),
              "auc": float(roc_auc_score(allp.label, allp.probability)),
              "brier": float(brier_score_loss(allp.label, allp.probability))}
    (out / "run_manifest.json").write_text(json.dumps(report, ensure_ascii=False,
                                                   indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
