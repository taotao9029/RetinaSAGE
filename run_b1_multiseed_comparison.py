"""Run matched B1 vs cost-sensitive spatial residual over fixed inner seeds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/B1_multiseed_20261029"
FEATURES = ROOT / "runs/P1/feature_extraction_manifest.csv"
SPLITS = ROOT / "runs/B/split_manifest.csv"
BASELINE_ROOTS = {
    42: ROOT / "runs/B1_fusion/b1",
}
MODEL_ROOTS = {
    42: ROOT / "runs/B1_fusion/b1_cost_sensitive_spatial",
}
SEEDS = (42, 43, 44, 45, 46)


def run(command, env):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def metrics(frame, probability_col, prediction_col):
    y = frame.label.to_numpy(int)
    p = frame[probability_col].to_numpy(float)
    pred = frame[prediction_col].to_numpy(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    for seed in SEEDS:
        if seed == 42:
            continue
        seed_root = OUT / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        config = yaml.safe_load((ROOT / "config_v1_local.yaml").read_text(encoding="utf-8"))
        config["training"]["seed"] = seed
        config["data"]["output_dir"] = str(seed_root.resolve())
        config_path = seed_root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                               encoding="utf-8")
        b1_root = seed_root / "b1"
        run([sys.executable, "-u", "train_b12345.py", "--task", "b1",
             "--config", str(config_path), "--feature-manifest", str(FEATURES),
             "--split-manifest", str(SPLITS), "--device", "cpu"], env)
        cost_root = seed_root / "b1_cost_sensitive_spatial"
        run([sys.executable, "-u", "train_b1_cost_sensitive_spatial.py",
             "--b1-root", str(b1_root), "--feature-manifest", str(FEATURES),
             "--output", str(cost_root)], env)
        BASELINE_ROOTS[seed] = b1_root
        MODEL_ROOTS[seed] = cost_root

    rows = []
    paired_predictions = []
    for seed in SEEDS:
        b1_root, model_root = BASELINE_ROOTS[seed], MODEL_ROOTS[seed]
        b1 = pd.read_csv(b1_root / "all_outer_test_predictions.csv", encoding="utf-8-sig")
        model = pd.read_csv(model_root / "all_outer_test_predictions.csv", encoding="utf-8-sig")
        keys = ["animal_id", "label", "outer_fold"]
        pair = b1.merge(model, on=keys, suffixes=("_b1", "_new"), validate="one_to_one")
        if len(pair) != 128:
            raise ValueError(f"Seed {seed}: expected 128 paired outer predictions")
        baseline_metrics = metrics(pair, "probability_b1", "prediction_b1")
        model_metrics = metrics(pair, "probability_new", "prediction_new")
        rows.append({"seed": seed, **{f"b1_{k}": v for k, v in baseline_metrics.items()},
                     **{f"new_{k}": v for k, v in model_metrics.items()},
                     **{f"delta_{k}": model_metrics[k] - baseline_metrics[k]
                        for k in baseline_metrics},
                     "changed_predictions": int((pair.prediction_b1 != pair.prediction_new).sum()),
                     "improved_animals": int(((pair.prediction_b1 != pair.label) &
                                               (pair.prediction_new == pair.label)).sum()),
                     "worsened_animals": int(((pair.prediction_b1 == pair.label) &
                                               (pair.prediction_new != pair.label)).sum())})
        pair.insert(0, "seed", seed)
        paired_predictions.append(pair)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "paired_multiseed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(paired_predictions, ignore_index=True).to_csv(
        OUT / "paired_multiseed_predictions.csv", index=False, encoding="utf-8-sig")
    metric_names = ("accuracy", "balanced_accuracy", "auc", "brier")
    summary = {"seeds": list(SEEDS), "outer_split": str(SPLITS),
               "feature_manifest": str(FEATURES), "n_animals_per_seed": 128,
               "comparison": "matched B1 versus cost-sensitive logistic plus spatial residual",
               "note": "Same fixed outer folds; seed changes inner CV only. Exploratory because this outer cohort has been inspected in prior experiments.",
               "mean_delta": {k: float(table[f"delta_{k}"].mean()) for k in metric_names},
               "sd_delta": {k: float(table[f"delta_{k}"].std(ddof=1)) for k in metric_names},
               "per_seed": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                 indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
