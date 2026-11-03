"""B6 strict ablation: P1 pipeline with the spatial GATv2 disabled.
python train_b6.py \
    --config config_v1.yaml \
    --feature-manifest /path/to/feature_extraction_manifest.csv \
    --split-manifest /path/to/split_manifest.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

import train_p123 as p123


class NoGATModel(p123.StrictP123Model):
    def __init__(self, feature_dim, hidden_dim=128, heads=4,
                 temporal_layers=2, dropout=0.1, no_gat=False,
                 horizons=(0.5, 1.0, 2.0)):
        super().__init__(feature_dim, hidden_dim, heads, temporal_layers,
                         dropout, no_gat=True, horizons=horizons)


# train_p123.build_model resolves StrictP123Model at call time. Replacing the
# class here keeps every split, loss and evaluation rule identical to P1.
p123.StrictP123Model = NoGATModel


THRESHOLD_RULE = {
    "source": "inner_oof",
    "grid_start": 0.0,
    "grid_stop": 1.0,
    "grid_step": 0.01,
    "primary_metric": "balanced_accuracy",
    "tie_break_1": "f1",
    "tie_break_2": "closest_to_0.5",
}


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {"path": str(path), "exists": True,
                "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return {
        "path": str(path),
        "exists": True,
        "files": [file_fingerprint(path / name) for name in
                   ("config.json", "preprocessor_config.json", "model.safetensors")
                   if (path / name).exists()],
    }


def resource_snapshot(device):
    result = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "python": os.sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        result.update({
            "gpu_index": int(index),
            "gpu_name": torch.cuda.get_device_name(index),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        })
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        result["git_commit"] = "unavailable"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v1.yaml")
    parser.add_argument("--feature-manifest")
    parser.add_argument("--split-manifest")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = p123.load_config(args.config)
    config.setdefault("training", {})["inner_folds_actual"] = 4
    seed = int(config["training"].get("seed", 42))
    p123.seed_all(seed)
    frame, default_split = p123.read_frame(config, "p1", args.feature_manifest)
    feature_manifest_path = (Path(args.feature_manifest).resolve()
                             if args.feature_manifest else None)
    split_manifest_path = Path(args.split_manifest or default_split).resolve()
    split = pd.read_csv(split_manifest_path, encoding="utf-8-sig")
    split["animal_id"] = split["animal_id"].astype(str)
    output = Path(config["data"]["output_dir"]) / "b6"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.monotonic()
    resources = resource_snapshot(device)

    predictions, summaries = [], []
    folds = [int(fold) for fold in
             sorted(split["outer_fold"].dropna().astype(int).unique())]
    for fold in folds:
        prediction, summary, _, _, _ = p123.run_outer(
            frame, split, fold, config, "p1", device, output
        )
        predictions.append(prediction)
        summaries.append(summary)

    all_predictions = pd.concat(predictions, ignore_index=True)
    if all_predictions["animal_id"].duplicated().any():
        raise RuntimeError("B6 outer-test output contains duplicate animals")
    summary = pd.DataFrame(summaries)
    all_predictions.to_csv(output / "all_outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "outer_fold_summary.csv", index=False, encoding="utf-8-sig")
    config["experiment"] = {
        "task": "b6", "base_task": "p1", "ablation": "no_gat",
        "outer_folds": [int(fold) for fold in folds],
        "n_outer_folds": int(len(folds)), "inner_folds": 4,
        "n_repeats": 1, "repeat_id": 0, "seed": seed,
        "animal_denominator": "unique_animals_in_frame_and_outer_test_once",
        "threshold_selection": THRESHOLD_RULE,
        "probability_calibration": "not_applied",
        "config_path": str(Path(args.config).resolve()),
        "config_version": "sha256_recorded_in_run_manifest",
        "feature_manifest": str(feature_manifest_path) if feature_manifest_path else None,
        "split_manifest": str(split_manifest_path),
    }
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    split_seed = split["split_seed"].dropna().iloc[0] if "split_seed" in split.columns and split["split_seed"].notna().any() else None
    split_version = split["split_version"].dropna().iloc[0] if "split_version" in split.columns and split["split_version"].notna().any() else None
    split_seed = int(split_seed) if split_seed is not None else None
    split_version = str(split_version) if split_version is not None else None
    required_metrics = ["test_accuracy", "test_balanced_accuracy", "test_auc", "test_ap", "test_f1", "test_sensitivity", "test_specificity", "test_brier", "test_threshold"]
    manifest = {
        "task": "b6",
        "base_task": "p1",
        "ablation": "no_gat",
        "outer_folds": [int(fold) for fold in folds],
        "n_outer_folds": int(len(folds)), "inner_folds": 4,
        "n_repeats": 1, "repeat_id": 0, "seed": seed,
        "split_seed": split_seed, "split_version": split_version,
        "n_unique_animals": int(frame["animal_id"].nunique()),
        "animal_denominator": int(frame["animal_id"].nunique()),
        "label_counts": {str(k): int(v) for k, v in frame.groupby("animal_id")["label"].first().value_counts().sort_index().items()},
        "n_outer_test_predictions": int(len(all_predictions)),
        "no_duplicate_outer_predictions": True,
        "prediction_file": "all_outer_test_predictions.csv",
        "outer_summary_file": "outer_fold_summary.csv",
        "per_animal_outer_prediction": True,
        "reported_metrics": required_metrics,
        "threshold_selection": THRESHOLD_RULE,
        "probability_calibration": "not_applied",
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "feature_manifest": str(feature_manifest_path) if feature_manifest_path else None,
        "feature_manifest_sha256": sha256_file(feature_manifest_path) if feature_manifest_path else None,
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "backbone": file_fingerprint(config.get("backbone", {}).get("model_name", "")),
        "weights_version": file_fingerprint(config.get("backbone", {}).get("model_name", "")),
        "mean_outer_auc": float(summary["test_auc"].mean()),
        "run_seconds": float(time.monotonic() - started),
        "resource": resources,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
