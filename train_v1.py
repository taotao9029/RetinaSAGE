from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset_v1 import discover_videos, load_config, save_manifest
from model_Transformer import V1VideoModel

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FeatureDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        payload = torch.load(row["cache_path"], map_location="cpu")
        return {
            "regions": payload["region_features"].float(),
            "global_features": payload["global_features"].float(),
            "timestamps": payload["timestamps"].float(),
            "label": torch.tensor(int(row["label"]), dtype=torch.float32),
            "animal_id": str(row["animal_id"]),
        }

def collate_features(batch):
    max_steps = max(item["regions"].shape[0] for item in batch)
    feature_dim = batch[0]["regions"].shape[-1]
    n_patches = batch[0]["regions"].shape[1]
    regions = torch.zeros(len(batch), max_steps, n_patches, feature_dim)
    global_features = torch.zeros(len(batch), max_steps, feature_dim)
    timestamps = torch.zeros(len(batch), max_steps)
    valid = torch.zeros(len(batch), max_steps, dtype=torch.bool)
    labels = torch.stack([item["label"] for item in batch])
    animal_ids = []
    for index, item in enumerate(batch):
        steps = item["regions"].shape[0]
        regions[index, :steps] = item["regions"]
        global_features[index, :steps] = item["global_features"]
        timestamps[index, :steps] = item["timestamps"]
        valid[index, :steps] = True
        animal_ids.append(item["animal_id"])
    return {
        "regions": regions,
        "global_features": global_features,
        "timestamps": timestamps,
        "valid": valid,
        "labels": labels,
        "animal_ids": animal_ids,
    }

def make_loader(rows, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FeatureDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_features,
        generator=generator,
    )

def evaluate(model, loader, device, loss_cls_fn=None, loss_pred_fn=None, lambda_pred=0.2, collect_state=False):
    model.eval()
    losses, cls_losses, pred_losses = [], [], []
    labels, probabilities, ids, states = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            regions = batch["regions"].to(device)
            global_features = batch["global_features"].to(device)
            valid = batch["valid"].to(device)
            target = batch["labels"].to(device)
            # model forward: logits, future_pred, gt_future_feat, state
            logits, future_pred, gt_future_feat, state = model(
                regions, global_features, valid, return_details=collect_state
            )
            if loss_cls_fn is not None and loss_pred_fn is not None:
                loss_cls = loss_cls_fn(logits, target)
                mask_time = valid[:, 1:]
                if torch.any(mask_time):
                    future_pred_valid = future_pred[mask_time]
                    gt_future_valid = gt_future_feat[mask_time]
                    loss_pred = loss_pred_fn(future_pred_valid, gt_future_valid)
                else:
                    loss_pred = 0.0 * loss_cls
                total_loss = loss_cls + lambda_pred * loss_pred
                losses.append(float(total_loss.item()))
                cls_losses.append(float(loss_cls.item()))
                pred_losses.append(float(loss_pred.item()))
            labels.extend(target.cpu().numpy().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            ids.extend(batch["animal_ids"])
            if collect_state:
                states.append(state.cpu())
    result = {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "cls_loss": float(np.mean(cls_losses)) if cls_losses else np.nan,
        "pred_loss": float(np.mean(pred_losses)) if pred_losses else np.nan,
        "label": np.asarray(labels, dtype=np.int64),
        "probability": np.asarray(probabilities, dtype=np.float64),
        "animal_id": ids,
    }
    if collect_state:
        result["state"] = torch.cat(states).numpy() if states else np.empty((0, 128))
    return result

def safe_auc(labels, probabilities):
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else np.nan

def metrics(labels, probabilities, threshold):
    prediction = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "auc": safe_auc(labels, probabilities),
        "ap": float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) > 1 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "threshold": float(threshold),
        "n": int(len(labels)),
    }

def choose_threshold(labels, probabilities):
    best = None
    for threshold in np.arange(0.0, 1.001, 0.01):
        current = metrics(labels, probabilities, float(threshold))
        score = (
            current["balanced_accuracy"],
            current["f1"],
            current["accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if best is None or score > best[0]:
            best = (score, float(threshold))
    return best[1]

def build_model(feature_dim, config, device, ablation):
    model_cfg = config["model"]
    return V1VideoModel(
        feature_dim=feature_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        heads=int(model_cfg.get("attention_heads", 4)),
        temporal_layers=int(model_cfg.get("temporal_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        graph=str(model_cfg.get("graph", "gatv2")),
        no_gat=ablation == "no_gat",
    ).to(device)

def train_epochs(model, train_loader, val_loader, config, device, epochs, save_dir, seed):
    training = config["training"]
    lambda_pred = float(training.get("lambda_pred", 0.2))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loss_cls_fn = nn.BCEWithLogitsLoss()
    loss_pred_fn = nn.MSELoss()
    patience = int(training.get("early_stopping_patience", 15))
    best_auc = -np.inf
    best_epoch = 1
    stale = 0
    history = []
    best_path = Path(save_dir) / "best_model.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, train_cls_losses, train_pred_losses = [], [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            regions = batch["regions"].to(device)
            global_features = batch["global_features"].to(device)
            valid = batch["valid"].to(device)
            target = batch["labels"].to(device)
            logits, future_pred, gt_future_feat, _ = model(regions, global_features, valid, return_details=False)
            loss_cls = loss_cls_fn(logits, target)
            # =========修改点：时间掩码，增加空值防护=========
            mask_time = valid[:, 1:]
            if torch.any(mask_time):
                future_pred_valid = future_pred[mask_time]
                gt_future_valid = gt_future_feat[mask_time]
                loss_pred = loss_pred_fn(future_pred_valid, gt_future_valid)
            else:
                loss_pred = 0.0 * loss_cls
            total_loss = loss_cls + lambda_pred * loss_pred
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(total_loss.item()))
            train_cls_losses.append(float(loss_cls.item()))
            train_pred_losses.append(float(loss_pred.item()))
        scheduler.step()
        validation = evaluate(model, val_loader, device, loss_cls_fn, loss_pred_fn, lambda_pred, collect_state=False)
        val_auc_metric = safe_auc(validation["label"], validation["probability"])
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
            "train_cls_loss": float(np.mean(train_cls_losses)) if train_cls_losses else np.nan,
            "train_pred_loss": float(np.mean(train_pred_losses)) if train_pred_losses else np.nan,
            "val_loss": validation["loss"],
            "val_cls_loss": validation["cls_loss"],
            "val_pred_loss": validation["pred_loss"],
            "val_auc": val_auc_metric,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        monitor = val_auc_metric if np.isfinite(val_auc_metric) else -np.inf
        if monitor > best_auc:
            best_auc = monitor
            best_epoch = epoch
            stale = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "seed": seed}, best_path)
        else:
            stale += 1
        if stale >= patience:
            break
    if not best_path.exists():
        torch.save({"state_dict": model.state_dict(), "epoch": best_epoch, "seed": seed}, best_path)
    return best_epoch, history, best_path

def rows_for_ids(frame, ids):
    return frame[frame["animal_id"].isin(set(ids))].to_dict("records")

def run_outer_fold(frame, outer_fold, train_ids, test_ids, config, output_dir, device, ablation):
    training = config["training"]
    seed = int(training.get("seed", 42)) + outer_fold * 1000
    outer_train = frame[frame["animal_id"].isin(set(train_ids))].reset_index(drop=True)
    outer_test = frame[frame["animal_id"].isin(set(test_ids))].reset_index(drop=True)
    # =========防御校验=========
    if len(outer_train) == 0:
        raise RuntimeError(f"Outer Fold {outer_fold}: outer_train 为空！train_ids长度={len(train_ids)}，请检查split_manifest.csv的role字段！")
    label_counts = outer_train["label"].value_counts()
    if len(label_counts) < 2:
        raise RuntimeError(f"Outer Fold {outer_fold}: outer_train 缺少一类标签，label分布：{label_counts.to_dict()}")
    min_per_class = label_counts.min()
    # ==============================
    from sklearn.model_selection import StratifiedGroupKFold
    inner_folds = min(int(training.get("inner_folds", 5)), int(min_per_class))
    if inner_folds < 2:
        raise ValueError(f"outer fold {outer_fold} 内层每类样本不足")
    inner = StratifiedGroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed,
    )
    oof_parts, best_epochs = [], []
    feature_dim = int(torch.load(outer_train.iloc[0]["cache_path"], map_location="cpu")["region_features"].shape[-1])
    for inner_fold, (fit_idx, val_idx) in enumerate(
        inner.split(outer_train, outer_train["label"], groups=outer_train["animal_id"]), 1
    ):
        fit_rows = outer_train.iloc[fit_idx].to_dict("records")
        val_rows = outer_train.iloc[val_idx].to_dict("records")
        fold_dir = output_dir / f"outer_{outer_fold}" / f"inner_{inner_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(seed + inner_fold)
        model = build_model(feature_dim, config, device, ablation)
        train_loader = make_loader(fit_rows, int(training.get("batch_size", 4)), int(training.get("num_workers", 2)), True, seed + inner_fold)
        val_loader = make_loader(val_rows, int(training.get("batch_size", 4)), int(training.get("num_workers", 2)), False, seed + inner_fold)
        best_epoch, history, best_path = train_epochs(
            model, train_loader, val_loader, config, device,
            int(training.get("epochs", 100)), fold_dir, seed + inner_fold,
        )
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        prediction = evaluate(model, val_loader, device, collect_state=False)
        oof_parts.append(pd.DataFrame({
            "animal_id": prediction["animal_id"],
            "label": prediction["label"],
            "probability": prediction["probability"],
            "inner_fold": inner_fold,
        }))
        best_epochs.append(best_epoch)
    oof = pd.concat(oof_parts, ignore_index=True)
    threshold = choose_threshold(oof["label"].to_numpy(), oof["probability"].to_numpy())
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    final_rows = outer_train.to_dict("records")
    test_rows = outer_test.to_dict("records")
    final_dir = output_dir / f"outer_{outer_fold}" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed + 99)
    final_model = build_model(feature_dim, config, device, ablation)
    final_loader = make_loader(final_rows, int(training.get("batch_size", 4)), int(training.get("num_workers", 2)), True, seed + 99)
    empty_val = make_loader(final_rows[:1], 1, 0, False, seed + 100)
    _, history, final_path = train_epochs(
        final_model, final_loader, empty_val, config, device,
        final_epochs, final_dir, seed + 99,
    )
    # Final training has no selection loader; retain its last state as the deployable state.
    final_model.eval()
    test_loader = make_loader(test_rows, int(training.get("batch_size", 4)), int(training.get("num_workers", 2)), False, seed + 500)
    prediction = evaluate(final_model, test_loader, device, collect_state=True)
    test_metrics = metrics(prediction["label"], prediction["probability"], threshold)
    prediction_frame = pd.DataFrame({
        "animal_id": prediction["animal_id"],
        "label": prediction["label"],
        "probability": prediction["probability"],
        "prediction": (prediction["probability"] >= threshold).astype(int),
        "outer_fold": outer_fold,
        "threshold": threshold,
    })
    state_frame = pd.DataFrame(
        prediction["state"],
        columns=[f"state_{index:03d}" for index in range(prediction["state"].shape[1])],
    )
    prediction_frame = pd.concat([prediction_frame.reset_index(drop=True), state_frame], axis=1)
    prediction_frame.to_csv(output_dir / f"outer_{outer_fold}" / "test_predictions.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(output_dir / f"outer_{outer_fold}" / "inner_oof.csv", index=False, encoding="utf-8-sig")
    bundle = {
        "feature_dim": feature_dim,
        "config": config["model"],
        "ablation": ablation,
        "state_dict": final_model.state_dict(),
        "threshold": threshold,
        "final_epochs": final_epochs,
        "outer_fold": outer_fold,
        "test_animal_ids": list(test_ids),
    }
    torch.save(bundle, output_dir / f"outer_{outer_fold}" / "final_bundle.pt")
    return prediction_frame, {
        "outer_fold": outer_fold,
        "final_epochs": final_epochs,
        "threshold": threshold,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v1.yaml")
    parser.add_argument("--ablation", choices=["full", "no_gat"], default="full")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    training = config["training"]
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    audit_csv = Path(config["data"]["output_dir"]) / "feature_extraction_manifest.csv"
    with open(audit_csv, "r", encoding="gbk", errors="replace") as f:
        frame = pd.read_csv(f)
    print(f"✅ 读取清单，样本数量={len(frame)}")
    frame["video_id"] = frame["video_id"].astype(int)
    frame["label"] = frame["label"].astype(int)
    cache_dir = Path(config["data"]["cache_dir"])
    def build_cache_path(vid):
        return str((cache_dir / f"{vid:05d}.pt").resolve())
    frame["cache_path"] = frame["video_id"].apply(build_cache_path)
    missing = []
    for p in frame["cache_path"]:
        if not Path(p).exists():
            missing.append(p)
    if len(missing) > 0:
        raise FileNotFoundError(f"缺失特征文件，共{len(missing)}个，例如：{missing[0]}")
    print("✅ 全部pt特征文件校验通过")
    split_csv = Path(config["data"]["output_dir"]) / "split_manifest.csv"
    with open(split_csv, "r", encoding="gbk", errors="replace") as f:
        split_df = pd.read_csv(f)
    print(f"✅ 读取预定义折划分清单，animal数量={len(split_df)}")
    split_df = split_df[["animal_id", "outer_fold", "role"]]
    frame = pd.merge(frame, split_df, on="animal_id", how="left")
    if frame["outer_fold"].isna().sum() > 0:
        raise ValueError(f"有{frame['outer_fold'].isna().sum()}个animal_id在split_manifest找不到匹配，请检查ID命名！")
    output_dir = Path(config["data"]["output_dir"]) / args.ablation
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    all_predictions, summaries = [], []
    for outer_fold in [1,2,3,4,5]:
        fold_df = frame[frame["outer_fold"] == outer_fold]
        train_ids = fold_df[fold_df["role"].isin(["inner_train", "inner_val"])]["animal_id"].unique().tolist()
        test_ids = fold_df[fold_df["role"] == "outer_test"]["animal_id"].unique().tolist()
        print(f"\n==== Outer Fold {outer_fold} ====")
        print(f"train animal count: {len(train_ids)}, test animal count: {len(test_ids)}")
        predictions, summary = run_outer_fold(
            frame, outer_fold, train_ids, test_ids, config, output_dir, device, args.ablation
        )
        all_predictions.append(predictions)
        summaries.append(summary)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = pd.DataFrame(summaries)
    predictions.to_csv(output_dir / "all_outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "outer_fold_summary.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    manifest = {
        "ablation": args.ablation,
        "device": str(device),
        "n_animals": int(frame["animal_id"].nunique()),
        "n_outer_test_predictions": int(len(predictions)),
        "mean_outer_auc": float(summary["test_auc"].mean()),
        "mean_outer_balanced_accuracy": float(summary["test_balanced_accuracy"].mean()),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
