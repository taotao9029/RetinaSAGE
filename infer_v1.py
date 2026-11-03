"""Inference from V1 final bundles on an existing feature cache."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from dataset_v1 import discover_videos, load_config
from model_v1 import V1VideoModel
from train_v1 import FeatureDataset, collate_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v1.yaml")
    parser.add_argument("--bundle-dir", default="./runs/v1/full")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="./runs/v1/inference_predictions.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    frame = discover_videos(config)
    cache_dir = Path(config["data"]["cache_dir"])
    rows = []
    for row in frame.to_dict("records"):
        row["cache_path"] = str(cache_dir / f"{int(row['video_id']):05d}.pt")
        if not Path(row["cache_path"]).exists():
            raise FileNotFoundError(f"缺少特征缓存: {row['cache_path']}")
        rows.append(row)

    device = torch.device(args.device)
    loader = torch.utils.data.DataLoader(
        FeatureDataset(rows),
        batch_size=int(config["training"].get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 2)),
        collate_fn=collate_features,
    )
    bundle_paths = sorted(Path(args.bundle_dir).glob("outer_*/final_bundle.pt"))
    if not bundle_paths:
        raise FileNotFoundError(f"找不到模型 bundle: {args.bundle_dir}")
    models, thresholds = [], []
    feature_dim = int(torch.load(rows[0]["cache_path"], map_location="cpu")["region_features"].shape[-1])
    for path in bundle_paths:
        bundle = torch.load(path, map_location="cpu")
        model_cfg = config["model"]
        network = V1VideoModel(
            feature_dim=feature_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            heads=int(model_cfg.get("attention_heads", 4)),
            temporal_layers=int(model_cfg.get("temporal_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            graph=str(model_cfg.get("graph", "grid4")),
            seed=int(config["training"].get("seed", 42)),
            no_gat=bundle.get("ablation") == "no_gat",
        ).to(device)
        network.load_state_dict(bundle["state_dict"], strict=True)
        network.eval()
        models.append(network)
        thresholds.append(float(bundle.get("threshold", 0.5)))

    records = []
    with torch.no_grad():
        for batch in loader:
            regions = batch["regions"].to(device)
            global_features = batch["global_features"].to(device)
            valid = batch["valid"].to(device)
            outputs = [network(regions, global_features, valid, return_details=True) for network in models]
            probability = torch.stack([item["probability"] for item in outputs]).mean(0).cpu()
            state = torch.stack([item["state"] for item in outputs]).mean(0).cpu()
            for index, animal_id in enumerate(batch["animal_ids"]):
                row = {"animal_id": animal_id, "probability": float(probability[index])}
                row["prediction"] = int(row["probability"] >= sum(thresholds) / len(thresholds))
                row.update({f"state_{j:03d}": float(value) for j, value in enumerate(state[index])})
                records.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"saved {len(records)} predictions to {output.resolve()}")


if __name__ == "__main__":
    main()

