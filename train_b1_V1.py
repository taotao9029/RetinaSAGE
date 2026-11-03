"""Matched S1/S2/S3 experiments on the same DINOv3 P1 cache.

S1: first valid frame, Frangi-weighted local regions + global feature, LR.
S2: the same per-frame local+global feature, Deep Sets aggregation.
S3: the same per-frame local+global feature, small GRU aggregation.

All three methods use the existing animal-level outer split manifest, four
inner folds for selection/early stopping, one repeat, and no GAT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from train_p123 import load_config


METHODS = ("s1", "s2", "s3", "static_residual")
THRESHOLD_RULE = {
    "source": "inner_oof",
    "grid_start": 0.0,
    "grid_stop": 1.0,
    "grid_step": 0.01,
    "primary_metric": "balanced_accuracy",
    "tie_break_1": "f1",
    "tie_break_2": "closest_to_0.5",
}


def seed_all(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path):
    path = Path(path)
    if not path.exists(): return {"path": str(path), "exists": False}
    if path.is_file():
        return {"path": str(path), "exists": True, "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size}
    return {"path": str(path), "exists": True, "files": [
        file_fingerprint(path / name) for name in
        ("config.json", "preprocessor_config.json", "model.safetensors")
        if (path / name).exists()]}


def resource_snapshot(device):
    result = {"timestamp_utc": pd.Timestamp.utcnow().isoformat(),
              "python": os.sys.executable, "platform": platform.platform(),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "cuda_available": bool(torch.cuda.is_available()), "device": str(device),
              "torch_num_threads": torch.get_num_threads()}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        result["git_commit"] = "unavailable"
    return result


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def metric_dict(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "auc": safe_auc(y, p),
        "ap": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(brier_score_loss(y, p)), "threshold": float(threshold),
    }


def choose_threshold(y, p):
    candidates = [(metric_dict(y, p, float(t)), float(t))
                  for t in np.arange(0.0, 1.001, 0.01)]
    return max(candidates, key=lambda z: (
        z[0]["balanced_accuracy"], z[0]["f1"], -abs(z[1] - 0.5)))[1]


def read_frame(config, manifest_path):
    frame = pd.read_csv(manifest_path, encoding="utf-8-sig")
    frame["animal_id"] = frame["animal_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    if "cache_path" not in frame:
        cache = Path(config["data"]["cache_dir"])
        frame["cache_path"] = frame["video_id"].map(
            lambda x: str((cache / f"{int(x):05d}.pt").resolve()))
    if frame["animal_id"].duplicated().any():
        labels = frame.groupby("animal_id")["label"].nunique()
        if (labels > 1).any(): raise ValueError("animal has conflicting labels")
        frame = frame.sort_values(["animal_id", "video_id"]).drop_duplicates(
            "animal_id", keep="first")
    missing = [p for p in frame["cache_path"] if not Path(p).exists()]
    if missing: raise FileNotFoundError(f"missing cache file: {missing[0]}")
    return frame.reset_index(drop=True)


def frame_features(row):
    """Build the shared S1/S2/S3 frame vector from the frozen P1 cache."""
    payload = torch.load(row["cache_path"], map_location="cpu")
    local = payload["region_features"].float()       # reference-Frangi weighted
    global_ = payload["global_features"].float()     # DINOv3 pooler/CLS feature
    valid = payload.get("region_valid_mask",
                        torch.ones(local.shape[:2], dtype=torch.bool)).bool()
    local = local * valid[..., None].float()
    # Fixed region order is retained. This is the same vector for S1, S2 and S3;
    # only the aggregation over frames changes.
    return torch.cat((local.flatten(1), global_), dim=1).numpy().astype(np.float32)


def fit_standardizer(rows):
    values = np.concatenate([frame_features(row) for row in rows], axis=0)
    mean = values.mean(0); std = values.std(0); std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def make_features(rows, mean, std):
    return [((frame_features(row) - mean) / std).astype(np.float32) for row in rows]


class SetOrGRU(nn.Module):
    def __init__(self, dim, kind, hidden=128, dropout=0.1):
        super().__init__(); self.kind = kind
        if kind == "s2":
            self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        elif kind == "s3":
            self.gru = nn.GRU(dim, hidden, batch_first=True)
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, mask):
        if self.kind == "s2":
            z = self.phi(x); z = (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        else:
            z, _ = self.gru(x); last = mask.sum(1).long().clamp_min(1) - 1
            z = z[torch.arange(len(z), device=z.device), last]
        return self.head(z).squeeze(-1)


class StaticResidualEventModel(nn.Module):
    """Bounded dynamic correction to a frozen static B1 anchor."""
    def __init__(self, dim, hidden=128, dropout=0.1, alpha=0.5):
        super().__init__()
        self.event = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                   nn.GELU())
        self.head = nn.Linear(hidden * 2, 1)
        # At initialization the new model is exactly the static anchor.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.alpha = float(alpha)

    def forward(self, events, mask, anchor_logit):
        z = self.event(events)
        m = mask[..., None]
        mean = (z * m).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        neg_inf = torch.finfo(z.dtype).min
        vmax = z.masked_fill(~m, neg_inf).amax(1)
        vmax = torch.where(mask.any(1, keepdim=True), vmax, torch.zeros_like(vmax))
        raw = self.head(torch.cat((mean, vmax), dim=1)).squeeze(-1)
        dynamic_correction = torch.tanh(raw)
        residual = self.alpha * dynamic_correction
        return anchor_logit + residual, dynamic_correction


def pad_sequences(xs, ys, device):
    t = max(x.shape[0] for x in xs); d = xs[0].shape[1]
    x = torch.zeros(len(xs), t, d, device=device); mask = torch.zeros(len(xs), t, dtype=torch.bool, device=device)
    for i, value in enumerate(xs):
        x[i, :len(value)] = torch.from_numpy(value).to(device); mask[i, :len(value)] = True
    return x, mask, torch.tensor(ys, dtype=torch.float32, device=device)


def event_sequences(xs):
    """Continuous consecutive differences; no thresholding or event deletion."""
    out = []
    for x in xs:
        if len(x) >= 2:
            out.append(np.diff(x, axis=0).astype(np.float32))
        else:
            out.append(np.zeros((1, x.shape[1]), dtype=np.float32))
    return out


def pad_event_sequences(xs, device):
    t = max(x.shape[0] for x in xs); d = xs[0].shape[1]
    x = torch.zeros(len(xs), t, d, device=device)
    mask = torch.zeros(len(xs), t, dtype=torch.bool, device=device)
    for i, value in enumerate(xs):
        x[i, :len(value)] = torch.from_numpy(value).to(device)
        mask[i, :len(value)] = True
    return x, mask


def seq_predict(model, xs, device):
    model.eval()
    with torch.no_grad():
        x, mask, _ = pad_sequences(xs, [0] * len(xs), device)
        return torch.sigmoid(model(x, mask)).cpu().numpy()


def fit_seq(train_x, train_y, val_x, val_y, kind, cfg, seed, device, lr, epochs):
    seed_all(seed)
    model = SetOrGRU(train_x[0].shape[1], kind, 128,
                     float(cfg.get("dropout", 0.1))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(cfg.get("weight_decay", 0.01)))
    patience = int(cfg.get("early_stopping_patience", 15)); best = -np.inf
    best_state = None; best_epoch = 1; stale = 0
    for epoch in range(1, epochs + 1):
        model.train(); x, mask, y = pad_sequences(train_x, train_y, device)
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x, mask), y)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip_norm", 1.0)))
        opt.step()
        p = seq_predict(model, val_x, device); score = safe_auc(val_y, p)
        score = score if np.isfinite(score) else -float(loss.detach().cpu())
        if score > best:
            best = score; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, best_epoch


def fit_seq_fixed(train_x, train_y, kind, cfg, seed, device, lr, epochs):
    seed_all(seed)
    model = SetOrGRU(train_x[0].shape[1], kind, 128,
                     float(cfg.get("dropout", 0.1))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(cfg.get("weight_decay", 0.01)))
    x, mask, y = pad_sequences(train_x, train_y, device)
    for _ in range(max(1, int(epochs))):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x, mask), y)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip_norm", 1.0)))
        opt.step()
    return model


def fit_static_anchor(x_rows, y, c):
    """Fit the B1-equivalent first-frame logistic-regression anchor."""
    clf = LogisticRegression(C=float(c), max_iter=2000, solver="liblinear")
    clf.fit(np.stack([x[0] for x in x_rows]), y)
    return clf


def anchor_logits(clf, xs):
    # sklearn expects a 2-D array even when predicting one sample.
    return np.asarray(
        [clf.decision_function(x[0][None, :])[0] for x in xs],
        dtype=np.float32,
    )


def static_residual_predict(model, anchor, xs, device):
    model.eval()
    events = event_sequences(xs)
    x, mask = pad_event_sequences(events, device)
    a = torch.from_numpy(anchor_logits(anchor, xs)).to(device)
    with torch.no_grad():
        logits, dynamic_correction = model(x, mask, a)
        residual = model.alpha * dynamic_correction
    return torch.sigmoid(logits).cpu().numpy(), residual.cpu().numpy()


def fit_static_residual_cv(train_x, train_y, val_x, val_y, anchor,
                           cfg, seed, device, lr, epochs):
    """Train only the dynamic branch; the logistic anchor is never updated."""
    seed_all(seed)
    model = StaticResidualEventModel(
        train_x[0].shape[1], 128, float(cfg.get("dropout", 0.1)),
        float(cfg.get("alpha", 0.5)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=float(cfg.get("weight_decay", 0.01)))
    tx, tm = pad_event_sequences(event_sequences(train_x), device)
    ta = torch.from_numpy(anchor_logits(anchor, train_x)).to(device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    patience = int(cfg.get("early_stopping_patience", 15)); best = -np.inf
    best_state = None; best_epoch = 1; stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        logits, dynamic_correction = model(tx, tm, ta)
        residual = model.alpha * dynamic_correction
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + float(cfg.get("residual_shrinkage", 0.01)) * residual.square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip_norm", 1.0)))
        opt.step()
        p, _ = static_residual_predict(model, anchor, val_x, device)
        score = safe_auc(val_y, p)
        score = score if np.isfinite(score) else -float(loss.detach().cpu())
        if score > best:
            best = score; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    p, residual = static_residual_predict(model, anchor, val_x, device)
    return model, best_epoch, p, residual


def fit_static_residual_fixed(train_x, train_y, anchor, cfg, seed, device, lr, epochs):
    """Refit the residual branch with a frozen anchor after inner selection."""
    seed_all(seed)
    model = StaticResidualEventModel(
        train_x[0].shape[1], 128, float(cfg.get("dropout", 0.1)),
        float(cfg.get("alpha", 0.5)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=float(cfg.get("weight_decay", 0.01)))
    x, mask = pad_event_sequences(event_sequences(train_x), device)
    a = torch.from_numpy(anchor_logits(anchor, train_x)).to(device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    for _ in range(max(1, int(epochs))):
        model.train(); opt.zero_grad(set_to_none=True)
        logits, dynamic_correction = model(x, mask, a)
        residual = model.alpha * dynamic_correction
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + float(cfg.get("residual_shrinkage", 0.01)) * residual.square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip_norm", 1.0)))
        opt.step()
    return model


def run_outer(frame, split, fold, config, method, device, output):
    tr = config["training"]; base_seed = int(tr.get("seed", 42)) + fold * 1000
    sf = split[split["outer_fold"].astype(int) == fold]
    test_ids = set(sf.loc[sf["role"] == "outer_test", "animal_id"].astype(str))
    train_ids = set(sf.loc[sf["role"] != "outer_test", "animal_id"].astype(str))
    outer_train = frame[frame.animal_id.isin(train_ids)].reset_index(drop=True)
    outer_test = frame[frame.animal_id.isin(test_ids)].reset_index(drop=True)
    if train_ids & test_ids or outer_train.empty or outer_test.empty:
        raise ValueError(f"invalid outer split {fold}")
    if int(outer_train.label.value_counts().min()) < 4:
        raise ValueError(f"outer fold {fold} cannot provide exactly 4 inner folds")
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=base_seed).split(
        outer_train, outer_train.label, groups=outer_train.animal_id))
    fold_root = output / f"outer_{fold}"; fold_root.mkdir(parents=True, exist_ok=True)
    static_anchor_c = float(tr.get("static_anchor_c", tr.get("logistic_c_candidates", [1.0])[0]))
    if method == "s1":
        candidates = [("c", float(c)) for c in tr.get("logistic_c_candidates", [0.01, 0.1, 1.0, 10.0])]
    elif method == "static_residual":
        candidates = [("lr", float(lr)) for lr in tr.get(
            "residual_learning_rate_candidates",
            tr.get("learning_rate_candidates", [1e-4, 3e-4]))]
    else:
        candidates = [("lr", float(lr)) for lr in tr.get("learning_rate_candidates", [1e-4, 3e-4])]
    records = []
    for name, value in candidates:
        parts = []; epochs_used = []
        for k, (fi, vi) in enumerate(inner, 1):
            fit_rows = outer_train.iloc[fi].to_dict("records"); val_rows = outer_train.iloc[vi].to_dict("records")
            mean, std = fit_standardizer(fit_rows)
            fit_x = make_features(fit_rows, mean, std); val_x = make_features(val_rows, mean, std)
            y_fit = outer_train.iloc[fi].label.to_numpy(); y_val = outer_train.iloc[vi].label.to_numpy()
            if method == "s1":
                clf = LogisticRegression(C=value, max_iter=2000, solver="liblinear")
                clf.fit(np.stack([x[0] for x in fit_x]), y_fit)
                p = clf.predict_proba(np.stack([x[0] for x in val_x]))[:, 1]; ep = 1
            elif method == "static_residual":
                anchor = fit_static_anchor(fit_x, y_fit, static_anchor_c)
                _, ep, p, _ = fit_static_residual_cv(
                    fit_x, y_fit, val_x, y_val, anchor, tr,
                    base_seed + k, device, value, int(tr.get("epochs", 100)))
            else:
                model, ep = fit_seq(fit_x, y_fit, val_x, y_val, method, tr,
                                    base_seed + k, device, value, int(tr.get("epochs", 100)))
                p = seq_predict(model, val_x, device)
            parts.append(pd.DataFrame({"animal_id": outer_train.iloc[vi].animal_id.to_numpy(),
                                       "label": y_val, "probability": p, "inner_fold": k}))
            epochs_used.append(ep)
        oof = pd.concat(parts, ignore_index=True); auc = safe_auc(oof.label, oof.probability)
        records.append({"value": value, "auc": auc, "oof": oof, "epochs": epochs_used})
    best = max(records, key=lambda r: -np.inf if not np.isfinite(r["auc"]) else r["auc"])
    best["oof"].to_csv(fold_root / "inner_oof.csv", index=False, encoding="utf-8-sig")
    threshold = choose_threshold(best["oof"].label.to_numpy(), best["oof"].probability.to_numpy())
    all_rows = outer_train.to_dict("records"); test_rows = outer_test.to_dict("records")
    mean, std = fit_standardizer(all_rows); train_x = make_features(all_rows, mean, std); test_x = make_features(test_rows, mean, std)
    y_train = outer_train.label.to_numpy(); y_test = outer_test.label.to_numpy()
    dynamic_residual = np.zeros(len(test_x), dtype=np.float32)
    if method == "s1":
        final = LogisticRegression(C=best["value"], max_iter=2000, solver="liblinear")
        final.fit(np.stack([x[0] for x in train_x]), y_train)
        p = final.predict_proba(np.stack([x[0] for x in test_x]))[:, 1]; final_epochs = 1
        torch.save({"method": method, "classifier": final, "mean": mean, "std": std,
                    "threshold": threshold, "outer_fold": fold}, fold_root / "final_bundle.pt")
    elif method == "static_residual":
        anchor = fit_static_anchor(train_x, y_train, static_anchor_c)
        final_epochs = max(1, int(round(np.median(best["epochs"]))))
        final = fit_static_residual_fixed(
            train_x, y_train, anchor, tr, base_seed + 99, device,
            best["value"], final_epochs)
        p, dynamic_residual = static_residual_predict(final, anchor, test_x, device)
        torch.save({"method": method, "anchor_classifier": anchor,
                    "residual_state_dict": final.state_dict(), "mean": mean, "std": std,
                    "static_anchor_c": static_anchor_c,
                    "alpha": float(tr.get("alpha", 0.5)),
                    "residual_shrinkage": float(tr.get("residual_shrinkage", 0.01)),
                    "learning_rate": best["value"], "final_epochs": final_epochs,
                    "threshold": threshold, "outer_fold": fold}, fold_root / "final_bundle.pt")
    else:
        final_epochs = max(1, int(round(np.median(best["epochs"]))))
        final = fit_seq_fixed(train_x, y_train, method, tr, base_seed + 99, device, best["value"], final_epochs)
        p = seq_predict(final, test_x, device)
        torch.save({"method": method, "state_dict": final.state_dict(), "mean": mean, "std": std,
                    "learning_rate": best["value"], "final_epochs": final_epochs,
                    "threshold": threshold, "outer_fold": fold}, fold_root / "final_bundle.pt")
    result = pd.DataFrame({"animal_id": outer_test.animal_id.to_numpy(), "label": y_test,
                           "probability": p, "prediction": (p >= threshold).astype(int),
                           "dynamic_residual_logit": dynamic_residual,
                           "outer_fold": fold, "threshold": threshold})
    result.to_csv(fold_root / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {"method": method, "outer_fold": fold, "inner_folds": 4,
               "selected_value": best["value"], "inner_auc": best["auc"],
               "final_epochs": final_epochs,
               "static_anchor_c": static_anchor_c if method == "static_residual" else np.nan,
               "residual_mean_abs": float(np.mean(np.abs(dynamic_residual))) if method == "static_residual" else np.nan,
               "residual_max_abs": float(np.max(np.abs(dynamic_residual))) if method == "static_residual" else np.nan}
    summary.update({f"test_{k}": v for k, v in metric_dict(y_test, p, threshold).items()})
    return result, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v1.yaml")
    ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--method", choices=["all", *METHODS], default="all")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    config = load_config(args.config); config.setdefault("training", {})["inner_folds_actual"] = 4
    seed = int(config["training"].get("seed", 42)); seed_all(seed)
    device = torch.device(args.device); started = time.monotonic(); resources = resource_snapshot(device)
    frame = read_frame(config, args.feature_manifest)
    split_path = Path(args.split_manifest).resolve(); split = pd.read_csv(split_path, encoding="utf-8-sig")
    split["animal_id"] = split["animal_id"].astype(str)
    split_seed = (split["split_seed"].dropna().iloc[0]
                  if "split_seed" in split.columns and split["split_seed"].notna().any() else None)
    split_version = (split["split_version"].dropna().iloc[0]
                     if "split_version" in split.columns and split["split_version"].notna().any() else None)
    split_seed = int(split_seed) if split_seed is not None else None
    split_version = str(split_version) if split_version is not None else None
    folds = [int(x) for x in sorted(split["outer_fold"].dropna().astype(int).unique())]
    if len(folds) != 5: raise ValueError(f"unified protocol requires 5 outer folds, got {folds}")
    methods = list(METHODS) if args.method == "all" else [args.method]
    root = Path(config["data"]["output_dir"]) / "s1_s3"
    for method in methods:
        output = root / method; output.mkdir(parents=True, exist_ok=True)
        predictions = []; summaries = []
        for fold in folds:
            p, s = run_outer(frame, split, fold, config, method, device, output)
            predictions.append(p); summaries.append(s)
        allp = pd.concat(predictions, ignore_index=True)
        if allp.animal_id.duplicated().any(): raise RuntimeError("duplicate outer animal prediction")
        allp.to_csv(output / "all_outer_test_predictions.csv", index=False, encoding="utf-8-sig")
        summary = pd.DataFrame(summaries); summary.to_csv(output / "outer_fold_summary.csv", index=False, encoding="utf-8-sig")
        resolved = dict(config)
        resolved["experiment"] = {"task": method, "method": method, "outer_folds": folds,
            "n_outer_folds": 5, "inner_folds": 4, "n_repeats": 1, "repeat_id": 0,
            "seed": seed, "animal_denominator": int(frame.animal_id.nunique()),
            "split_seed": split_seed, "split_version": split_version,
            "local_feature_source": "region_features_reference_frame_frangi_weighted",
            "global_feature_source": "global_features_dinov3_pooler_or_cls",
            "aggregation": {"s1": "first_valid_frame_logistic_regression",
                             "s2": "deep_sets_unordered_frames",
                             "s3": "gru_ordered_frames",
                             "static_residual": "frozen_first_frame_lr_plus_continuous_difference_residual"}[method],
            "static_anchor": "frozen_first_valid_frame_logistic_regression" if method == "static_residual" else None,
            "dynamic_branch": "continuous_consecutive_standardized_frame_differences" if method == "static_residual" else None,
            "residual_fusion": "single_frame_logit_plus_alpha_times_tanh_dynamic_correction" if method == "static_residual" else None,
            "alpha": float(config["training"].get("alpha", 0.5)) if method == "static_residual" else None,
            "residual_shrinkage": float(config["training"].get("residual_shrinkage", 0.01)) if method == "static_residual" else None,
            "gat": "not_used", "threshold_selection": THRESHOLD_RULE,
            "probability_calibration": "not_applied"}
        with (output / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved, f, allow_unicode=True, sort_keys=False)
        manifest = {"task": method, "method": method, "outer_folds": folds,
            "n_outer_folds": 5, "inner_folds": 4, "n_repeats": 1, "repeat_id": 0,
            "seed": seed, "n_unique_animals": int(frame.animal_id.nunique()),
            "animal_denominator": int(frame.animal_id.nunique()),
            "split_seed": split_seed, "split_version": split_version,
            "label_counts": {str(k): int(v) for k, v in frame.groupby("animal_id")["label"].first().value_counts().sort_index().items()},
            "n_outer_test_predictions": int(len(allp)), "no_duplicate_outer_predictions": True,
            "per_animal_outer_prediction": True, "prediction_file": "all_outer_test_predictions.csv",
            "outer_summary_file": "outer_fold_summary.csv",
            "reported_metrics": ["test_accuracy", "test_balanced_accuracy", "test_auc", "test_ap", "test_f1", "test_sensitivity", "test_specificity", "test_brier", "test_threshold"],
            "local_feature_source": "region_features_reference_frame_frangi_weighted",
            "global_feature_source": "global_features_dinov3_pooler_or_cls", "gat": "not_used",
            "static_anchor": "frozen_first_valid_frame_logistic_regression" if method == "static_residual" else None,
            "dynamic_branch": "continuous_consecutive_standardized_frame_differences" if method == "static_residual" else None,
            "residual_fusion": "single_frame_logit_plus_alpha_times_tanh_dynamic_correction" if method == "static_residual" else None,
            "alpha": float(config["training"].get("alpha", 0.5)) if method == "static_residual" else None,
            "residual_shrinkage": float(config["training"].get("residual_shrinkage", 0.01)) if method == "static_residual" else None,
            "threshold_selection": THRESHOLD_RULE, "probability_calibration": "not_applied",
            "config_path": str(Path(args.config).resolve()), "config_sha256": sha256_file(args.config),
            "config_version": "sha256_recorded_in_run_manifest",
            "feature_manifest": str(Path(args.feature_manifest).resolve()), "feature_manifest_sha256": sha256_file(args.feature_manifest),
            "split_manifest": str(split_path), "split_manifest_sha256": sha256_file(split_path),
            "weights_version": file_fingerprint(config.get("backbone", {}).get("model_name", "")),
            "mean_outer_auc": float(summary.test_auc.mean()), "run_seconds": float(time.monotonic() - started),
            "resource": resources}
        (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
