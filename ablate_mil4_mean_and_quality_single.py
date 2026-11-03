"""Matched feature-mean and quality-single-frame ablations for MIL-4.

Uses the same frozen DINOv3 cache, outer split, inner SGKF seed, label-free
quality score, and LogisticRegression family as B1. Predictions are written in
the format consumed by train_b1_mil4_final_calibration.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import train_b1_V2 as shared
from train_p123 import load_config


CANDIDATES = [0.01, 0.1, 1.0, 10.0]


def standardized_frame_rows(rows):
    all_values = np.concatenate([shared.frame_features(r) for r in rows], axis=0)
    mean, std = all_values.mean(0), all_values.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def selected_matrix(row, mean, std, budget, selection):
    raw = ((shared.frame_features(row) - mean) / std).astype(np.float32)
    if selection == "quality":
        idx = shared.quality_select_indices(shared.frame_quality(row), budget)
    elif selection == "uniform":
        idx = np.unique(np.floor((np.arange(budget) + 0.5) * len(raw) / budget).astype(int))
        idx = np.clip(idx, 0, len(raw) - 1)
    else:
        raise ValueError(selection)
    return raw[idx], idx


def animal_vectors(rows, mean, std, budget, selection):
    vectors, indices = [], []
    for row in rows:
        frames, idx = selected_matrix(row, mean, std, budget, selection)
        vectors.append(frames.mean(axis=0))
        indices.append(idx.tolist())
    return np.stack(vectors), indices


def fit_lr(x, y, c):
    model = LogisticRegression(C=float(c), max_iter=2000, solver="liblinear")
    model.fit(x, y)
    return model


def run_fold(frame, split, fold, config, budget, selection, out):
    trcfg = config["training"]
    seed = int(trcfg.get("seed", 42)) + fold * 1000
    sf = split[split.outer_fold.astype(int) == fold]
    test_ids = set(sf.loc[sf.role == "outer_test", "animal_id"].astype(str))
    train_ids = set(sf.loc[sf.role != "outer_test", "animal_id"].astype(str))
    outer_train = frame[frame.animal_id.isin(train_ids)].reset_index(drop=True)
    outer_test = frame[frame.animal_id.isin(test_ids)].reset_index(drop=True)
    n_inner = 4
    inner = list(StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed).split(
        outer_train, outer_train.label, groups=outer_train.animal_id))

    candidates = []
    for c in CANDIDATES:
        parts = []
        for k, (fi, vi) in enumerate(inner, 1):
            fit_rows = outer_train.iloc[fi].to_dict("records")
            val_rows = outer_train.iloc[vi].to_dict("records")
            mean, std = standardized_frame_rows(fit_rows)
            xfit, _ = animal_vectors(fit_rows, mean, std, budget, selection)
            xval, val_idx = animal_vectors(val_rows, mean, std, budget, selection)
            yfit = outer_train.iloc[fi].label.to_numpy(int)
            yval = outer_train.iloc[vi].label.to_numpy(int)
            model = fit_lr(xfit, yfit, c)
            p = model.predict_proba(xval)[:, 1]
            part = pd.DataFrame({"animal_id": outer_train.iloc[vi].animal_id.to_numpy(),
                                 "label": yval, "probability": p, "inner_fold": k,
                                 "selected_frame_indices": [json.dumps(v) for v in val_idx]})
            parts.append(part)
        oof = pd.concat(parts, ignore_index=True)
        auc = shared.safe_auc(oof.label.to_numpy(), oof.probability.to_numpy())
        candidates.append((auc, c, oof))
    best = max(candidates, key=lambda z: (-np.inf if not np.isfinite(z[0]) else z[0], -z[1]))

    train_rows = outer_train.to_dict("records")
    test_rows = outer_test.to_dict("records")
    mean, std = standardized_frame_rows(train_rows)
    xtrain, train_idx = animal_vectors(train_rows, mean, std, budget, selection)
    xtest, test_idx = animal_vectors(test_rows, mean, std, budget, selection)
    ytrain = outer_train.label.to_numpy(int)
    ytest = outer_test.label.to_numpy(int)
    final = fit_lr(xtrain, ytrain, best[1])
    ptest = final.predict_proba(xtest)[:, 1]

    fold_dir = out / f"outer_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best[2].to_csv(fold_dir / "inner_oof.csv", index=False, encoding="utf-8-sig")
    test = pd.DataFrame({"animal_id": outer_test.animal_id.to_numpy(), "label": ytest,
                         "probability": ptest, "outer_fold": fold,
                         "selected_frame_indices": [json.dumps(v) for v in test_idx]})
    test.to_csv(fold_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    return test, {"outer_fold": fold, "selected_C": best[1], "inner_auc": best[0],
                  "budget": budget, "selection": selection,
                  "mean_selected_frame_count": float(np.mean([len(v) for v in test_idx]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v1_local.yaml")
    ap.add_argument("--feature-manifest", default="runs/P1/feature_extraction_manifest.csv")
    ap.add_argument("--split-manifest", default="runs/B/split_manifest.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--selection", choices=["quality", "uniform"], default="quality")
    args = ap.parse_args()
    config = load_config(args.config)
    config.setdefault("training", {})["inner_folds_actual"] = 4
    frame = shared.read_frame(config, args.feature_manifest)
    frame["animal_id"] = frame.animal_id.astype(str)
    split = pd.read_csv(args.split_manifest, encoding="utf-8-sig")
    split["animal_id"] = split.animal_id.astype(str)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    predictions, summaries = [], []
    for fold in range(1, 6):
        p, s = run_fold(frame, split, fold, config, args.budget, args.selection, out)
        predictions.append(p); summaries.append(s)
    allp = pd.concat(predictions, ignore_index=True)
    allp.to_csv(out / "all_outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out / "outer_fold_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {"method": f"{args.selection}_{args.budget}_frame_mean_logistic",
                "selection": args.selection, "frame_budget": args.budget,
                "aggregation": "arithmetic_mean_of_standardized_frozen_frame_features",
                "classifier": "LogisticRegression(solver=liblinear,max_iter=2000)",
                "C_candidates": CANDIDATES, "outer_folds": [1,2,3,4,5],
                "inner_folds": 4, "n_repeats": 1,
                "animal_denominator": int(allp.animal_id.nunique()),
                "config": str(Path(args.config).resolve()),
                "feature_manifest": str(Path(args.feature_manifest).resolve()),
                "split_manifest": str(Path(args.split_manifest).resolve()),
                "split_sha256": hashlib.sha256(Path(args.split_manifest).read_bytes()).hexdigest(),
                "no_outer_test_used_for_model_selection": True,
                "calibration_applied_later": "same constrained B1+component final Platt procedure"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
