"""Future-prediction baselines and causal-prefix checks for P1/P2/P3."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge


HORIZONS = (0.5, 1.0, 2.0)


def _pairs_for_row(row, horizon, tolerance):
    payload = torch.load(row["cache_path"], map_location="cpu")
    features = payload.get("teacher_region_features", payload["region_features"]).float().numpy()
    timestamps = payload["timestamps"].float().numpy()
    valid = payload.get("region_valid_mask")
    if valid is None:
        valid = np.ones(features.shape[:2], dtype=bool)
    else:
        valid = valid.bool().numpy()

    pairs = []
    for origin in range(len(timestamps)):
        candidates = np.arange(origin + 1, len(timestamps))
        if len(candidates) == 0:
            continue
        errors = np.abs(timestamps[candidates] - (timestamps[origin] + horizon))
        target = int(candidates[int(np.argmin(errors))])
        if float(np.min(errors)) > tolerance:
            continue
        regions = np.where(valid[origin] & valid[target])[0]
        for region in regions.tolist():
            pairs.append(
                {
                    "animal_id": str(row["animal_id"]),
                    "origin_index": int(origin),
                    "target_index": target,
                    "region_index": int(region),
                    "origin_time": float(timestamps[origin]),
                    "target_time": float(timestamps[target]),
                    "x": features[origin, region].astype(np.float32),
                    "y": (features[target, region] - features[origin, region]).astype(np.float32),
                }
            )
    return pairs


def fit_future_baselines(rows, tolerance, ridge_alpha=1.0):
    fitted = {}
    for horizon in HORIZONS:
        pairs = [p for row in rows for p in _pairs_for_row(row, horizon, tolerance)]
        if not pairs:
            fitted[horizon] = {"mean": None, "ridge": None, "n_pairs": 0}
            continue
        x = np.stack([p["x"] for p in pairs])
        y = np.stack([p["y"] for p in pairs])
        counts = {}
        for pair in pairs:
            counts[pair["animal_id"]] = counts.get(pair["animal_id"], 0) + 1
        sample_weight = np.asarray([1.0 / counts[p["animal_id"]] for p in pairs], dtype=np.float64)
        animal_means = []
        for animal_id in sorted(counts):
            animal_means.append(np.stack([p["y"] for p in pairs if p["animal_id"] == animal_id]).mean(axis=0))
        ridge = Ridge(alpha=float(ridge_alpha))
        ridge.fit(x, y, sample_weight=sample_weight)
        linear = LinearRegression()
        linear.fit(x, y, sample_weight=sample_weight)
        fitted[horizon] = {
            "mean": np.stack(animal_means).mean(axis=0),
            "ridge": ridge,
            "linear": linear,
            "n_pairs": int(len(pairs)),
            "n_animals": int(len(animal_means)),
        }
    return fitted


def _rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def evaluate_future(model, train_rows, test_rows, scaler, tolerance, device):
    baselines = fit_future_baselines(train_rows, tolerance)
    details = []
    model.eval()
    with torch.no_grad():
        for row in test_rows:
            payload = torch.load(row["cache_path"], map_location="cpu")
            regions = payload["region_features"].float()
            global_features = payload["global_features"].float()
            timestamps = payload["timestamps"].float()
            region_valid = payload.get("region_valid_mask")
            if region_valid is None:
                region_valid = torch.ones(regions.shape[:2], dtype=torch.bool)
            else:
                region_valid = region_valid.bool()
            t = regions.shape[0]
            valid = torch.ones(1, t, dtype=torch.bool)
            out = model(
                regions.unsqueeze(0).to(device),
                global_features.unsqueeze(0).to(device),
                timestamps.unsqueeze(0).to(device),
                valid.to(device),
                region_valid.unsqueeze(0).to(device),
                True,
            )
            state = out["state"][0]
            for hi, horizon in enumerate(HORIZONS):
                fitted = baselines[horizon]
                if fitted["mean"] is None:
                    continue
                pairs = _pairs_for_row(row, horizon, tolerance)
                by_origin = {}
                for pair in pairs:
                    by_origin.setdefault(pair["origin_index"], []).append(pair)
                for origin, origin_pairs in by_origin.items():
                    region_ids = torch.tensor(
                        [p["region_index"] for p in origin_pairs], dtype=torch.long, device=device
                    )
                    pred_norm = model.predict_delta(state[origin, region_ids], horizon)
                    pred_model = (
                        pred_norm * scaler.std[hi].to(device) + scaler.mean[hi].to(device)
                    ).cpu().numpy()
                    target = np.stack([p["y"] for p in origin_pairs])
                    x = np.stack([p["x"] for p in origin_pairs])
                    mean_pred = np.repeat(fitted["mean"][None, :], len(origin_pairs), axis=0)
                    ridge_pred = fitted["ridge"].predict(x)
                    linear_pred = fitted["linear"].predict(x)
                    details.append(
                        {
                            "animal_id": str(row["animal_id"]),
                            "horizon_seconds": float(horizon),
                            "origin_index": int(origin),
                            "target_index": int(origin_pairs[0]["target_index"]),
                            "origin_time": float(origin_pairs[0]["origin_time"]),
                            "target_time": float(origin_pairs[0]["target_time"]),
                            "n_valid_regions": int(len(origin_pairs)),
                            "model_rmse": float(np.mean([_rmse(a, b) for a, b in zip(pred_model, target)])),
                            "hold_zero_rmse": float(np.mean([_rmse(np.zeros_like(b), b) for b in target])),
                            "train_mean_rmse": float(np.mean([_rmse(a, b) for a, b in zip(mean_pred, target)])),
                            "ridge_rmse": float(np.mean([_rmse(a, b) for a, b in zip(ridge_pred, target)])),
                        }
                    )
    detail = pd.DataFrame(details)
    if detail.empty:
        summary = pd.DataFrame(
            columns=["horizon_seconds", "method", "mean_rmse", "n_origins", "n_animals"]
        )
    else:
        rows = []
        for horizon, group in detail.groupby("horizon_seconds"):
            for method in ("model", "hold_zero", "train_mean", "ridge"):
                column = f"{method}_rmse"
                rows.append(
                    {
                        "horizon_seconds": float(horizon),
                        "method": method,
                        "mean_rmse": float(group[column].mean()),
                        "n_origins": int(len(group)),
                        "n_animals": int(group["animal_id"].nunique()),
                        "relative_to_hold_zero": float(
                            1.0 - group[column].mean() / max(group["hold_zero_rmse"].mean(), 1e-12)
                        ),
                    }
                )
        summary = pd.DataFrame(rows)
    return detail, summary


def _normal_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(values.mean())
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return mean, mean - half if np.isfinite(half) else np.nan, mean + half if np.isfinite(half) else np.nan, int(len(values))


def _variance_diagnostics(rows, tolerance, split_name):
    records = []
    for horizon in HORIZONS:
        pairs = [p for row in rows for p in _pairs_for_row(row, horizon, tolerance)]
        if not pairs:
            continue
        x = np.stack([p["x"] for p in pairs])
        y = np.stack([p["y"] for p in pairs])
        records.append({
            "split": split_name,
            "horizon_seconds": float(horizon),
            "n_pairs": int(len(pairs)),
            "n_animals": int(len({p["animal_id"] for p in pairs})),
            "input_variance_mean": float(np.var(x, axis=0).mean()),
            "input_variance_median": float(np.median(np.var(x, axis=0))),
            "target_variance_mean": float(np.var(y, axis=0).mean()),
            "target_variance_median": float(np.median(np.var(y, axis=0))),
            "target_zero_rmse": float(np.sqrt(np.mean(y ** 2))),
        })
    return records


def evaluate_future_acceptance(model, train_rows, test_rows, scaler, tolerance, device, seed=42):
    """Animal-level future-feature acceptance with causal and negative controls.

    Targets are always frozen teacher-region deltas selected by the original
    timestamps. No timestamp is renumbered or rewritten. The current model is
    evaluated one prefix at a time, so its input at an origin contains only
    frames observed at or before that origin.
    """
    baselines = fit_future_baselines(train_rows, tolerance)
    details = []
    raw_for_shuffle = []
    model.eval()
    with torch.no_grad():
        for row in test_rows:
            payload = torch.load(row["cache_path"], map_location="cpu")
            regions = payload["region_features"].float()
            global_features = payload["global_features"].float()
            timestamps = payload["timestamps"].float()
            region_valid = payload.get("region_valid_mask")
            if region_valid is None:
                region_valid = torch.ones(regions.shape[:2], dtype=torch.bool)
            else:
                region_valid = region_valid.bool()
            for horizon in HORIZONS:
                fitted = baselines[horizon]
                if fitted["mean"] is None:
                    continue
                pairs = _pairs_for_row(row, horizon, tolerance)
                by_origin = {}
                for pair in pairs:
                    by_origin.setdefault(pair["origin_index"], []).append(pair)
                for origin, origin_pairs in by_origin.items():
                    # Explicit prefix construction: no future tensor is passed
                    # to the predictor for this origin.
                    end = origin + 1
                    prefix_regions = regions[:end].unsqueeze(0).to(device)
                    prefix_global = global_features[:end].unsqueeze(0).to(device)
                    prefix_times = timestamps[:end].unsqueeze(0).to(device)
                    prefix_valid = torch.ones(1, end, dtype=torch.bool, device=device)
                    prefix_region_valid = region_valid[:end].unsqueeze(0).to(device)
                    out = model(prefix_regions, prefix_global, prefix_times,
                                prefix_valid, prefix_region_valid, True)
                    region_ids = torch.tensor([p["region_index"] for p in origin_pairs], dtype=torch.long, device=device)
                    pred_norm = model.predict_delta(out["state"][0, -1, region_ids], horizon)
                    pred_model = (pred_norm * scaler.std.to(device)[HORIZONS.index(horizon)] +
                                  scaler.mean.to(device)[HORIZONS.index(horizon)]).cpu().numpy()

                    # True temporal perturbation: reverse the observed prefix,
                    # while retaining the same target and real target time.
                    rev_regions = torch.flip(prefix_regions, dims=[1])
                    rev_global = torch.flip(prefix_global, dims=[1])
                    rev_times = torch.flip(prefix_times, dims=[1])
                    rev_valid = torch.flip(prefix_region_valid, dims=[1])
                    rev_out = model(rev_regions, rev_global, rev_times, prefix_valid, rev_valid, True)
                    rev_norm = model.predict_delta(rev_out["state"][0, -1, region_ids], horizon)
                    pred_reverse = (rev_norm * scaler.std.to(device)[HORIZONS.index(horizon)] +
                                    scaler.mean.to(device)[HORIZONS.index(horizon)]).cpu().numpy()

                    target = np.stack([p["y"] for p in origin_pairs])
                    x = np.stack([p["x"] for p in origin_pairs])
                    mean_pred = np.repeat(fitted["mean"][None, :], len(origin_pairs), axis=0)
                    linear_pred = fitted["linear"].predict(x)
                    ridge_pred = fitted["ridge"].predict(x)
                    item = {
                        "animal_id": str(row["animal_id"]),
                        "horizon_seconds": float(horizon),
                        "origin_index": int(origin),
                        "target_index": int(origin_pairs[0]["target_index"]),
                        "origin_time": float(origin_pairs[0]["origin_time"]),
                        "target_time": float(origin_pairs[0]["target_time"]),
                        "actual_span_seconds": float(origin_pairs[0]["target_time"] - origin_pairs[0]["origin_time"]),
                        "n_valid_regions": int(len(origin_pairs)),
                        "model_rmse": float(np.mean([_rmse(a, b) for a, b in zip(pred_model, target)])),
                        "hold_zero_rmse": float(np.mean([_rmse(np.zeros_like(b), b) for b in target])),
                        "train_mean_rmse": float(np.mean([_rmse(a, b) for a, b in zip(mean_pred, target)])),
                        "ridge_rmse": float(np.mean([_rmse(a, b) for a, b in zip(ridge_pred, target)])),
                        "linear_rmse": float(np.mean([_rmse(a, b) for a, b in zip(linear_pred, target)])),
                        "temporal_reverse_rmse": float(np.mean([_rmse(a, b) for a, b in zip(pred_reverse, target)])),
                    }
                    details.append(item)
                    raw_for_shuffle.append((item, target, pred_model))

    # Cross-animal target shuffle: keep model predictions and real inputs fixed,
    # but replace each target with a target vector from another animal.
    by_horizon_animal = {}
    for item, target, _ in raw_for_shuffle:
        by_horizon_animal.setdefault(item["horizon_seconds"], {}).setdefault(item["animal_id"], []).append(target)
    rng = np.random.default_rng(int(seed))
    for horizon, animals in by_horizon_animal.items():
        animal_ids = sorted(animals)
        if len(animal_ids) < 2:
            continue
        source_for = {animal_ids[i]: animal_ids[(i + 1) % len(animal_ids)] for i in range(len(animal_ids))}
        used = {a: 0 for a in animal_ids}
        for item, pred_target, pred_model in raw_for_shuffle:
            if item["horizon_seconds"] != horizon:
                continue
            source = source_for[item["animal_id"]]
            pool = animals[source]
            shuffled_target = pool[used[item["animal_id"]] % len(pool)]
            used[item["animal_id"]] += 1
            # Store the negative-control error in the corresponding detail row.
            item["target_shuffle_rmse"] = float(np.mean([_rmse(v, shuffled_target) for v in pred_model]))

    detail = pd.DataFrame(details)
    methods = ["model", "hold_zero", "train_mean", "linear", "ridge", "temporal_reverse", "target_shuffle"]
    summary_rows = []
    if not detail.empty:
        for horizon, group in detail.groupby("horizon_seconds"):
            for method in methods:
                column = f"{method}_rmse"
                if column not in group:
                    continue
                # First average origins within each animal, then form the
                # across-animal estimate and interval.
                animal_values = group.groupby("animal_id")[column].mean().to_numpy()
                mean, low, high, n = _normal_ci(animal_values)
                summary_rows.append({
                    "horizon_seconds": float(horizon), "method": method,
                    "animal_mean_rmse": mean, "ci95_low": low, "ci95_high": high,
                    "n_animals": n, "n_origins": int(len(group)),
                })
    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.DataFrame(_variance_diagnostics(train_rows, tolerance, "outer_train") +
                               _variance_diagnostics(test_rows, tolerance, "outer_test"))
    controls = pd.DataFrame([
        {"control": "target_shuffle_cross_animal",
         "interpretation": "跨动物替换冻结教师目标，输入和模型预测不变；用于检验任务是否退化为目标分布/均值匹配。"},
        {"control": "temporal_reverse",
         "interpretation": "保留真实目标，仅反转已观察前缀的时间顺序；这是输入时序扰动，不是跨动物负对照。"},
    ])
    return detail, summary, diagnostics, controls


def summarize_future_acceptance_details(detail):
    """Pool outer-fold details, then form intervals over unique animals."""
    if detail is None or detail.empty:
        return pd.DataFrame()
    methods = ["model", "hold_zero", "train_mean", "linear", "ridge", "temporal_reverse", "target_shuffle"]
    rows = []
    for horizon, group in detail.groupby("horizon_seconds"):
        for method in methods:
            column = f"{method}_rmse"
            if column not in group:
                continue
            animal_values = group.groupby("animal_id")[column].mean().to_numpy()
            mean, low, high, n = _normal_ci(animal_values)
            rows.append({"horizon_seconds": float(horizon), "method": method,
                         "animal_mean_rmse": mean, "ci95_low": low,
                         "ci95_high": high, "n_animals": n,
                         "n_origins": int(len(group))})
    return pd.DataFrame(rows)


def causal_prefix_check(model, row, device):
    payload = torch.load(row["cache_path"], map_location="cpu")
    regions = payload["region_features"].float().unsqueeze(0).to(device)
    global_features = payload["global_features"].float().unsqueeze(0).to(device)
    timestamps = payload["timestamps"].float().unsqueeze(0).to(device)
    region_valid = payload.get("region_valid_mask")
    if region_valid is None:
        region_valid = torch.ones(regions.shape[1], regions.shape[2], dtype=torch.bool)
    region_valid = region_valid.bool().unsqueeze(0).to(device)
    valid = torch.ones(1, regions.shape[1], dtype=torch.bool, device=device)
    split = max(1, regions.shape[1] // 2)
    original = model(regions, global_features, timestamps, valid, region_valid, True)
    changed_regions = regions.clone()
    changed_global = global_features.clone()
    changed_regions[:, split:] = changed_regions[:, split:] + 0.123
    changed_global[:, split:] = changed_global[:, split:] - 0.217
    changed = model(changed_regions, changed_global, timestamps, valid, region_valid, True)
    state_diff = (original["state"][:, :split] - changed["state"][:, :split]).abs().max().item()
    future_input_diff = (
        model.predict_delta(original["state"][:, :split, 0], HORIZONS[0])
        - model.predict_delta(changed["state"][:, :split, 0], HORIZONS[0])
    ).abs().max().item()
    return {
        "animal_id": str(row["animal_id"]),
        "prefix_length": int(split),
        "max_prefix_state_difference": float(state_diff),
        "max_prefix_future_input_difference": float(future_input_diff),
        "passed": bool(state_diff <= 1e-6 and future_input_diff <= 1e-6),
    }
