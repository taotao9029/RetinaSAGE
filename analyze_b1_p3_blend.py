#!/usr/bin/env python3
"""Nested OOF selection and outer-test evaluation of a B1/P3 probability blend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def ap(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def best_threshold(y, p):
    grid = np.arange(0.0, 1.001, 0.01)
    scores = [(balanced_accuracy_score(y, p >= t), t) for t in grid]
    return float(max(scores, key=lambda x: (x[0], -abs(x[1] - 0.5)))[1])


def load_outer(run_dir: Path):
    frame = pd.read_csv(run_dir / "all_outer_test_predictions.csv", encoding="utf-8-sig")
    required = {"animal_id", "label", "probability", "outer_fold"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{run_dir}: missing columns {sorted(required - set(frame.columns))}")
    frame = frame[["animal_id", "label", "probability", "outer_fold"]].copy()
    frame["animal_id"] = frame.animal_id.astype(str)
    if frame.animal_id.duplicated().any():
        raise ValueError(f"{run_dir}: duplicate outer-test animal predictions")
    return frame


def validate_pair(a, b, where):
    a = a.copy(); b = b.copy()
    a["animal_id"] = a.animal_id.astype(str); b["animal_id"] = b.animal_id.astype(str)
    if set(a.animal_id) != set(b.animal_id):
        raise ValueError(f"{where}: animal IDs differ (B1={len(a)}, P3={len(b)})")
    a = a.sort_values("animal_id").reset_index(drop=True)
    b = b.sort_values("animal_id").reset_index(drop=True)
    if not np.array_equal(a.label.to_numpy(), b.label.to_numpy()):
        raise ValueError(f"{where}: labels differ")
    return a, b


def metrics(y, p, threshold=None):
    result = {"n": int(len(y)), "auc": auc(y, p), "ap": ap(y, p),
              "brier": float(brier_score_loss(y, p))}
    if threshold is not None:
        result["threshold"] = float(threshold)
        result["balanced_accuracy"] = float(balanced_accuracy_score(y, p >= threshold))
    return result


def paired_bootstrap(y, a, b, seed=20261020, n_boot=10000):
    """Exploratory animal bootstrap; outer-fold model dependence remains."""
    rng = np.random.default_rng(seed)
    n = len(y)
    output = {}
    for name, fn in (("auc", auc), ("ap", ap),
                     ("brier", lambda yy, pp: float(brier_score_loss(yy, pp)))):
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            value = fn(y[idx], b[idx]) - fn(y[idx], a[idx])
            if np.isfinite(value):
                vals.append(value)
        output[name] = {"mean_delta": float(fn(y, b) - fn(y, a)),
                        "ci95_percentile": [float(x) for x in np.quantile(vals, [0.025, 0.975])]}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-dir", type=Path, default=Path("runs/B1/b1"))
    parser.add_argument("--p3-dir", type=Path, default=Path("runs/P3/p3"))
    parser.add_argument("--output", type=Path, default=Path("runs/analysis_b1_p3_blend"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    b1_test = load_outer(args.b1_dir)
    p3_test = load_outer(args.p3_dir)
    b1_test, p3_test = validate_pair(b1_test, p3_test, "outer predictions")
    if not np.array_equal(b1_test.outer_fold.to_numpy(), p3_test.outer_fold.to_numpy()):
        raise ValueError("outer fold assignments differ")

    selections, fold_results, blended_rows = [], [], []
    for fold in sorted(b1_test.outer_fold.unique()):
        b1_inner = pd.read_csv(args.b1_dir / f"outer_{int(fold)}" / "inner_oof.csv", encoding="utf-8-sig")
        p3_inner = pd.read_csv(args.p3_dir / f"outer_{int(fold)}" / "inner_oof.csv", encoding="utf-8-sig")
        b1_inner, p3_inner = validate_pair(b1_inner, p3_inner, f"outer {fold} inner OOF")
        if "inner_fold" in b1_inner and "inner_fold" in p3_inner:
            if not np.array_equal(b1_inner.inner_fold.to_numpy(), p3_inner.inner_fold.to_numpy()):
                raise ValueError(f"outer {fold}: inner fold assignments differ")

        yi = b1_inner.label.to_numpy(dtype=int)
        pb1i = b1_inner.probability.to_numpy(dtype=float)
        pp3i = p3_inner.probability.to_numpy(dtype=float)
        grid = np.linspace(0.0, 1.0, 21)
        candidates = [(auc(yi, (1.0 - w) * pb1i + w * pp3i), float(w)) for w in grid]
        # Prefer B1 on ties, to avoid adding a branch without inner-OOF evidence.
        selected_auc, weight = max(candidates, key=lambda z: (z[0], -z[1]))
        pblend_i = (1.0 - weight) * pb1i + weight * pp3i
        threshold = best_threshold(yi, pblend_i)

        test_fold_b1 = b1_test[b1_test.outer_fold == fold].sort_values("animal_id")
        test_fold_p3 = p3_test[p3_test.outer_fold == fold].sort_values("animal_id")
        test_fold_b1, test_fold_p3 = validate_pair(test_fold_b1, test_fold_p3, f"outer {fold} test")
        y = test_fold_b1.label.to_numpy(dtype=int)
        pb1 = test_fold_b1.probability.to_numpy(dtype=float)
        pp3 = test_fold_p3.probability.to_numpy(dtype=float)
        pblend = (1.0 - weight) * pb1 + weight * pp3
        fold_row = {"outer_fold": int(fold), "selected_p3_weight": weight,
                    "inner_oof_auc_at_selected_weight": float(selected_auc),
                    "inner_oof_threshold": threshold}
        for name, prob in (("b1", pb1), ("p3", pp3), ("blend", pblend)):
            fold_row.update({f"{name}_{k}": v for k, v in metrics(y, prob, threshold if name == "blend" else None).items()})
        fold_results.append(fold_row)
        selections.append({"outer_fold": int(fold), "p3_weight": weight,
                           "inner_oof_auc": float(selected_auc), "threshold": threshold})
        for animal, label, bprob, pprob, blendprob in zip(test_fold_b1.animal_id, y, pb1, pp3, pblend):
            blended_rows.append({"animal_id": animal, "label": int(label), "outer_fold": int(fold),
                                 "b1_probability": float(bprob), "p3_probability": float(pprob),
                                 "blend_probability": float(blendprob), "p3_weight": weight,
                                 "blend_threshold": threshold})

    fold_frame = pd.DataFrame(fold_results)
    pred_frame = pd.DataFrame(blended_rows).sort_values("animal_id")
    fold_frame.to_csv(args.output / "fold_metrics.csv", index=False)
    pred_frame.to_csv(args.output / "outer_predictions.csv", index=False)

    y = pred_frame.label.to_numpy(dtype=int)
    pb1 = pred_frame.b1_probability.to_numpy(dtype=float)
    pp3 = pred_frame.p3_probability.to_numpy(dtype=float)
    pblend = pred_frame.blend_probability.to_numpy(dtype=float)
    overall = {name: metrics(y, prob) for name, prob in (("b1", pb1), ("p3", pp3), ("blend", pblend))}
    thresholds = pred_frame.blend_threshold.to_numpy(dtype=float)
    overall["blend_fold_thresholded_balanced_accuracy"] = float(
        balanced_accuracy_score(y, pblend >= thresholds))
    report = {
        "selection": "per outer fold, select P3 probability weight on inner OOF AUC; fixed grid 0.00..1.00 step 0.05; ties prefer B1",
        "threshold": "inner OOF balanced-accuracy threshold, applied only to that outer fold's predictions",
        "n_animals": int(len(pred_frame)), "n_outer_folds": int(pred_frame.outer_fold.nunique()),
        "selected_weights": selections, "pooled_outer_metrics": overall,
        "fold_mean_metrics": {name: {metric: float(fold_frame[f"{name}_{metric}"].mean())
                                      for metric in ("auc", "ap", "brier")}
                              for name in ("b1", "p3", "blend")},
        "paired_animal_bootstrap_blend_minus_b1": paired_bootstrap(y, pb1, pblend),
        "caveat": "Bootstrap resamples animals paired, but outer-fold models share training data; intervals are exploratory, not independent-sample inference.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
