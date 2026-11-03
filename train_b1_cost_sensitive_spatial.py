"""Nested B1 anchor plus cost-sensitive logistic regional residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, recall_score, roc_auc_score)


WEIGHTS = (1.0, 2.0, 4.0)
CS = (0.1, 1.0, 10.0)
ALPHAS = (0.25, 0.5, 1.0)
FEATURES = ("global", "global_spatial_residual")
THRESHOLDS = np.arange(0.01, 1.001, 0.01)


def feature_for(path, kind):
    p = torch.load(path, map_location="cpu", weights_only=False)
    g = p["global_features"].float()[0].numpy()
    if kind == "global":
        return g
    r = p["region_features"].float()[0].numpy()
    valid = p["region_valid_mask"].bool()[0].numpy()
    r = r[valid] if valid.any() else r
    return np.concatenate((g, r.mean(0) - g, r.std(0))).astype(np.float32)


def fit_fold(train, valid, kind, c, healthy_weight):
    x = np.stack([feature_for(p, kind) for p in train.cache_path])
    z = np.stack([feature_for(p, kind) for p in valid.cache_path])
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.0
    model = LogisticRegression(C=c, max_iter=2000, solver="liblinear",
                               class_weight={0: healthy_weight, 1: 1.0})
    model.fit((x - mean) / std, train.label.to_numpy(int))
    return model.predict_proba((z - mean) / std)[:, 1]


def evaluate(y, p, threshold):
    pred = p >= threshold
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "sensitivity": float(recall_score(y, pred, pos_label=1)),
            "specificity": float(recall_score(y, pred, pos_label=0)),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "threshold": float(threshold)}


def choose_threshold(y, p, base):
    options = []
    for threshold in THRESHOLDS:
        m = evaluate(y, p, threshold)
        if m["accuracy"] + 1e-12 < base["accuracy"]:
            continue
        if m["sensitivity"] + 1e-12 < base["sensitivity"]:
            continue
        if m["brier"] > base["brier"] + 0.01:
            continue
        options.append(m)
    if not options:
        return None
    return max(options, key=lambda m: (m["balanced_accuracy"], m["accuracy"],
                                        m["auc"], -m["brier"],
                                        -abs(m["threshold"] - 0.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-root", required=True)
    ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, output = Path(args.b1_root), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {output}")
    manifest = pd.read_csv(args.feature_manifest, encoding="utf-8-sig")
    if len(manifest) != 128 or not manifest.animal_id.is_unique:
        raise ValueError("Expected 128 unique animals")
    manifest = manifest.set_index("animal_id", drop=False)
    output.mkdir(parents=True, exist_ok=True)
    results, summaries, candidates = [], [], []

    for outer in range(1, 6):
        inner = pd.read_csv(root / f"outer_{outer}/inner_oof.csv", encoding="utf-8-sig")
        test = pd.read_csv(root / f"outer_{outer}/test_predictions.csv", encoding="utf-8-sig")
        inner = inner.join(manifest[["cache_path"]], on="animal_id", validate="one_to_one")
        test = test.join(manifest[["cache_path"]], on="animal_id", validate="one_to_one")
        if set(inner.animal_id) | set(test.animal_id) != set(manifest.index):
            raise ValueError(f"Outer fold {outer} coverage mismatch")
        y, pb = inner.label.to_numpy(int), inner.probability.to_numpy(float)
        base = evaluate(y, pb, float(test.threshold.iloc[0]))
        feasible = [{"feature": "b1_only", "healthy_weight": 0.0,
                     "c": 0.0, "alpha": 0.0, **base}]
        for kind in FEATURES:
            for c in CS:
                for weight in WEIGHTS:
                    complement = np.empty(len(inner), dtype=float)
                    for fold in sorted(inner.inner_fold.unique()):
                        train = inner[inner.inner_fold != fold]
                        valid = inner[inner.inner_fold == fold]
                        complement[valid.index.to_numpy()] = fit_fold(
                            train, valid, kind, c, weight)
                    for alpha in ALPHAS:
                        p = (1 - alpha) * pb + alpha * complement
                        chosen = choose_threshold(y, p, base)
                        if chosen is None:
                            continue
                        record = {"feature": kind, "healthy_weight": weight,
                                  "c": c, "alpha": alpha, **chosen}
                        feasible.append(record)
                        candidates.append({"outer_fold": outer, **record})
        best = max(feasible, key=lambda m: (m["balanced_accuracy"], m["accuracy"],
                                             m["auc"], -m["brier"], -m["alpha"]))
        if best["feature"] == "b1_only":
            probability = test.probability.to_numpy(float)
        else:
            probability = fit_fold(inner, test, best["feature"], best["c"],
                                   best["healthy_weight"])
            probability = ((1 - best["alpha"]) * test.probability.to_numpy(float)
                           + best["alpha"] * probability)
        pred = (probability >= best["threshold"]).astype(int)
        result = pd.DataFrame({"animal_id": test.animal_id, "label": test.label,
                               "outer_fold": outer,
                               "b1_probability": test.probability,
                               "probability": probability, "prediction": pred,
                               "threshold": best["threshold"],
                               "selected_feature": best["feature"],
                               "selected_weight": best["healthy_weight"],
                               "selected_c": best["c"], "selected_alpha": best["alpha"]})
        result.to_csv(output / f"outer_{outer}_test_predictions.csv", index=False,
                      encoding="utf-8-sig")
        results.append(result)
        summaries.append({"outer_fold": outer, "selected_feature": best["feature"],
                          "selected_healthy_weight": best["healthy_weight"],
                          "selected_c": best["c"], "selected_alpha": best["alpha"],
                          "inner_b1_accuracy": base["accuracy"],
                          "inner_selected_accuracy": best["accuracy"],
                          "inner_b1_balanced_accuracy": base["balanced_accuracy"],
                          "inner_selected_balanced_accuracy": best["balanced_accuracy"],
                          **{f"test_{k}": v for k, v in evaluate(
                              test.label.to_numpy(int), probability,
                              best["threshold"]).items()}})
        print(f"outer={outer} feature={best['feature']} weight={best['healthy_weight']} "
              f"C={best['c']} alpha={best['alpha']}", flush=True)

    allp = pd.concat(results, ignore_index=True)
    if len(allp) != 128 or not allp.animal_id.is_unique:
        raise ValueError("Expected one outer prediction per animal")
    allp.to_csv(output / "all_outer_test_predictions.csv", index=False,
                encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output / "outer_fold_summary.csv", index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame(candidates).to_csv(output / "inner_feasible_candidates.csv",
                                    index=False, encoding="utf-8-sig")
    report = {"method": "B1_anchor_cost_sensitive_logistic_spatial_residual",
              "features": FEATURES, "healthy_class_weight_grid": WEIGHTS,
              "c_grid": CS, "alpha_grid": ALPHAS, "outer_folds": 5,
              "inner_folds": 4, "n_repeats": 1, "animal_denominator": 128,
              "outer_test_used_for_selection": False,
              "selection": "inner OOF ACC and sensitivity >= B1, Brier <= B1 + 0.01; maximize balanced accuracy then ACC/AUC",
              "accuracy": float(accuracy_score(allp.label, allp.prediction)),
              "balanced_accuracy": float(balanced_accuracy_score(allp.label, allp.prediction)),
              "auc": float(roc_auc_score(allp.label, allp.probability)),
              "brier": float(brier_score_loss(allp.label, allp.probability))}
    (output / "run_manifest.json").write_text(json.dumps(report, ensure_ascii=False,
                                                        indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
