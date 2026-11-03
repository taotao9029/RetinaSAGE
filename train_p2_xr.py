"""Strict nested-CV runner for the P1, P2 and P3 tasks in 模型算法任务书.md.
运行完整 P2 消融：

  python train_p23_xr.py \
    --task p2 \
    --config config_v1.yaml \
    --feature-manifest /path/to/P2/feature_extraction_manifest.csv \
    --split-manifest /path/to/split_manifest.csv \
    --ablation all

  运行完整 P3 消融：

  python train_p23_xr.py \
    --task p3 \
    --config config_v1.yaml \
    --feature-manifest /path/to/P3/feature_extraction_manifest_p3.csv \
    --split-manifest /path/to/split_manifest.csv \
    --ablation all

  如果同时运行配准和亮度敏感性分析，再加：

  --preprocess-sensitivity
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset_v1 import load_config
from model_p123 import StrictP123Model, _sinusoidal
from future_evaluation import (HORIZONS as FUTURE_HORIZONS, evaluate_future,
                               evaluate_future_acceptance, summarize_future_acceptance_details,
                               causal_prefix_check)


HORIZONS = (0.5, 1.0, 2.0)


ABLATION_NAMES = ("baseline", "no_global_cls", "no_pred_loss", "mean_region", "no_frangi_weight",
                  "no_reliable_mask", "random_graph",
                  "raw_unregistered", "brightness_corrected")


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
        "path": str(path), "exists": True,
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
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        result["git_commit"] = "unavailable"
    return result


def materialize_preprocess_variant(config, task, manifest_path, mode, device, batch_size=8):
    """Re-encode raw sidecar frames for preprocessing sensitivity analyses.

    The frozen DINO encoder and the downstream CV split stay unchanged.  The
    generated manifest is deliberately separate from the normal cache so a
    sensitivity run cannot silently overwrite the primary experiment.
    """
    import cv2
    from extract_features_v2 import DINOv3Encoder, make_reference_frangi_weight, resize_with_padding

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    out = Path(config["data"]["output_dir"]) / task / "preprocess_cache" / mode
    out.mkdir(parents=True, exist_ok=True)
    encoder = DINOv3Encoder(config["backbone"]["model_name"], int(config["data"].get("image_size", 256)), device)
    rows = []
    target_size = int(config["data"].get("image_size", 256))
    for row in manifest.to_dict("records"):
        target = out / f"{int(row['video_id']):05d}.pt"
        if target.exists():
            rows.append({**row, "cache_path": str(target.resolve())})
            continue
        sidecar = row.get("raw_frames_path")
        if not sidecar or not Path(str(sidecar)).exists():
            payload = torch.load(row["cache_path"], map_location="cpu")
            sidecar = payload.get("raw_frames_path")
        if not sidecar or not Path(str(sidecar)).exists():
            raise FileNotFoundError(f"raw frame sidecar missing for video {row['video_id']}")
        raw = np.load(str(sidecar), allow_pickle=False)
        frames = raw["frames"].astype(np.uint8)
        masks = raw["registration_masks"] if "registration_masks" in raw else np.ones(frames.shape[:3], np.uint8) * 255
        if mode == "brightness_corrected":
            gray = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in frames])
            mean, std = float(gray.mean()), max(float(gray.std()), 1.0)
            corrected = []
            for frame in frames:
                x = frame.astype(np.float32)
                local = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                scale = 1.0 + (mean - local) / std
                corrected.append(np.clip((x - 127.5) * scale + 127.5, 0, 255).astype(np.uint8))
            frames = np.stack(corrected)
        padded, view_masks = [], []
        for frame, mask in zip(frames, masks):
            canvas, view = resize_with_padding(frame, target_size)
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            padded.append(canvas); view_masks.append(cv2.bitwise_and(view, mask))
        reference_weight = None
        if task == "p3":
            reference_weight = make_reference_frangi_weight(padded[0], target_size, view_masks[0])
        globals_, teacher_regions_, regions_, valid_, deg_ = [], [], [], [], []
        for start in range(0, len(padded), int(batch_size)):
            batch_frames = padded[start:start + batch_size]
            batch_masks = view_masks[start:start + batch_size]
            g, teacher_r, v, _ = encoder.encode(batch_frames, batch_masks, None)
            _, r, _, d = encoder.encode(batch_frames, batch_masks, reference_weight)
            globals_.append(g); teacher_regions_.append(teacher_r); regions_.append(r); valid_.append(v); deg_.append(d)
        original_payload = torch.load(row["cache_path"], map_location="cpu")
        payload = {
            "global_features": torch.cat(globals_).half(),
            "teacher_region_features": torch.cat(teacher_regions_).half(),
            "region_features": torch.cat(regions_).half(),
            "region_valid_mask": torch.cat(valid_), "frangi_degenerate_mask": torch.cat(deg_),
            "timestamps": original_payload.get("timestamps", torch.arange(len(frames), dtype=torch.float32)),
            "video_id": int(row["video_id"]), "animal_id": str(row["animal_id"]), "label": int(row["label"]),
            "structure_weight_type": "reference_frame_multiscale_frangi" if task == "p3" else None,
            "structure_weight_fixed_across_frames": True if task == "p3" else False,
        }
        torch.save(payload, target)
        rows.append({**row, "cache_path": str(target.resolve())})
    result = out / "feature_manifest.csv"
    pd.DataFrame(rows).to_csv(result, index=False, encoding="utf-8-sig")
    return result


def variant_label(variant):
    return str(variant or "baseline").replace("-", "_")


def degree_preserving_random_graph(n, seed, swaps=None):
    """Randomize the grid graph while preserving every node degree."""
    base = StrictP123Model.adjacency(n, torch.device("cpu")).numpy().astype(bool)
    rng = np.random.default_rng(int(seed))
    edges = {(min(i, j), max(i, j)) for i in range(n) for j in range(i + 1, n) if base[i, j]}
    if swaps is None:
        swaps = max(20, 10 * len(edges))
    for _ in range(int(swaps)):
        if len(edges) < 2:
            break
        e1, e2 = rng.choice(len(edges), size=2, replace=False)
        edge_list = list(edges)
        a, b = edge_list[int(e1)]
        c, d = edge_list[int(e2)]
        if len({a, b, c, d}) < 4:
            continue
        candidates = [((a, c), (b, d)), ((a, d), (b, c))]
        rng.shuffle(candidates)
        for x, y in candidates:
            x = tuple(sorted(x)); y = tuple(sorted(y))
            if x[0] == x[1] or y[0] == y[1] or x in edges or y in edges:
                continue
            edges.remove((a, b)); edges.remove((c, d))
            edges.add(x); edges.add(y)
            break
    out = np.eye(n, dtype=bool)
    for a, b in edges:
        out[a, b] = out[b, a] = True
    return torch.from_numpy(out)


class AblationP123Model(StrictP123Model):
    """P123 model with an optional fixed adjacency for graph ablations."""

    def __init__(self, *args, adjacency_override=None, no_global_cls=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_global_cls = bool(no_global_cls)
        if self.no_global_cls:
            # Matched retrained head: same hidden width, dropout and output
            # shape, but it receives only the local temporal representation.
            self.classifier_no_global = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(self.classifier[2].p)),
                nn.Linear(self.hidden_dim, 1),
            )
        self.register_buffer("adjacency_override", adjacency_override if adjacency_override is not None else torch.empty(0), persistent=False)

    def forward(self, regions, global_features, timestamps, valid, region_valid=None, return_state=False):
        state, region_valid = self.encode(regions, global_features, timestamps, valid, region_valid)
        w = region_valid.to(state.dtype) * valid[:, :, None].to(state.dtype)
        frame = (state * w[..., None]).sum(2) / w.sum(2, keepdim=True).clamp_min(1)
        tw = valid.to(state.dtype)
        pooled = (frame * tw[..., None]).sum(1) / tw.sum(1, keepdim=True).clamp_min(1)
        if self.no_global_cls:
            logits = self.classifier_no_global(pooled).squeeze(-1)
        else:
            gf = (global_features * tw[..., None]).sum(1) / tw.sum(1, keepdim=True).clamp_min(1)
            logits = self.classifier(torch.cat((pooled, self.global_proj(gf)), -1)).squeeze(-1)
        return {"logits": logits, "state": state, "region_valid": region_valid}

    def encode(self, regions, global_features, timestamps, valid, region_valid=None):
        if self.adjacency_override.numel() == 0:
            return super().encode(regions, global_features, timestamps, valid, region_valid)
        b, t, n, _ = regions.shape
        if self.adjacency_override.shape != (n, n):
            raise ValueError(f"adjacency shape {tuple(self.adjacency_override.shape)} != {(n, n)}")
        if region_valid is None:
            region_valid = valid[:, :, None].expand(-1, -1, n)
        x = self.region_in(regions)
        adjacency = self.adjacency_override.to(regions.device)
        spatial = []
        for i in range(t):
            z = x[:, i]
            if not self.no_gat:
                for layer in self.gat:
                    z = layer(z, adjacency)
            spatial.append(z)
        spatial = torch.stack(spatial, 1)
        clean = timestamps.masked_fill(~valid, 0.0)
        first = clean.gather(1, valid.float().argmax(1).view(-1, 1)).squeeze(1)
        rel = (clean - first[:, None]).clamp_min(0)
        dt = torch.zeros_like(rel)
        dt[:, 1:] = (rel[:, 1:] - rel[:, :-1]).clamp_min(0)
        time = torch.cat((_sinusoidal(rel, self.hidden_dim // 2),
                          _sinusoidal(dt, self.hidden_dim - self.hidden_dim // 2)), -1)
        z = spatial + self.time_proj(time).unsqueeze(2)
        flat = z.permute(0, 2, 1, 3).reshape(b * n, t, self.hidden_dim)
        pad = (~valid).unsqueeze(1).expand(-1, n, -1).reshape(b * n, t)
        causal = torch.triu(torch.ones(t, t, device=regions.device, dtype=torch.bool), 1)
        encoded = self.temporal(flat, mask=causal, src_key_padding_mask=pad)
        return encoded.reshape(b, n, t, self.hidden_dim).permute(0, 2, 1, 3), region_valid


def seed_all(seed):
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested):
    """Resolve a usable device and reject unsupported CUDA architectures early."""
    requested = str(requested).lower()
    if requested in ("auto", "cuda"):
        if not torch.cuda.is_available():
            if requested == "cuda":
                raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
            print("[device] CUDA unavailable; using CPU")
            return torch.device("cpu")
        capability = torch.cuda.get_device_capability(0)
        supported = set(torch.cuda.get_arch_list())
        arch = f"sm_{capability[0]}{capability[1]}"
        if arch not in supported:
            message = (
                f"GPU capability {arch} is not supported by this PyTorch build "
                f"({torch.__version__}, CUDA {torch.version.cuda}; supported: {sorted(supported)})."
            )
            if requested == "cuda":
                raise RuntimeError(message + " Install a PyTorch build supporting this GPU or use --device cpu.")
            print("[device] " + message + " Using CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(requested)


class CachedAnimals(Dataset):
    def __init__(self, rows, task, variant="baseline", perturb_seed=42):
        self.rows, self.task = rows, task
        self.variant, self.perturb_seed = variant_label(variant), int(perturb_seed)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        p = torch.load(row["cache_path"], map_location="cpu")
        regions = p["region_features"].float()
        teacher = p.get("teacher_region_features", regions).float()
        global_features = p["global_features"].float()
        valid = p.get("region_valid_mask", torch.ones(regions.shape[:2], dtype=torch.bool)).bool()
        degenerate = p.get("frangi_degenerate_mask", torch.zeros(regions.shape[:2], dtype=torch.bool)).bool()
        if self.task == "p3":
            if p.get("structure_weight_type") != "reference_frame_multiscale_frangi":
                raise ValueError(f"P3 cache is not fixed-reference Frangi: {row['cache_path']}")
            if p.get("structure_weight_fixed_across_frames") is not True:
                raise ValueError(f"P3 structure weights are not fixed: {row['cache_path']}")
        if self.variant in ("mean_region", "no_frangi_weight"):
            if self.task != "p3":
                raise ValueError("Frangi ablations are only valid for a P3 run")
            regions = teacher.clone()
        elif self.variant == "no_reliable_mask":
            if self.task != "p3":
                raise ValueError("no_reliable_mask is only a P3 ablation")
            # Remove only the reliable-region mask. The Frangi-weighted region
            # features remain unchanged; padded/invalid regions are therefore
            # included in the region pooling for this isolated ablation.
            valid = torch.ones_like(valid)
        elif self.variant in ("shuffle", "reverse"):
            q = len(regions)
            if self.variant == "reverse":
                order = torch.arange(q - 1, -1, -1)
            else:
                order = torch.from_numpy(np.random.default_rng(self.perturb_seed + int(row["video_id"])).permutation(q))
            regions, teacher = regions[order], teacher[order]
            global_features = global_features[order]
            region_valid, degenerate = valid[order], degenerate[order]
        return dict(regions=regions, teacher=teacher,
                    global_features=global_features,
                    timestamps=p["timestamps"].float(), region_valid=valid,
                    degenerate=degenerate, label=float(row["label"]),
                    animal_id=str(row["animal_id"]))


def collate(batch):
    b, t = len(batch), max(x["regions"].shape[0] for x in batch)
    n, d = batch[0]["regions"].shape[1:]
    out = dict(regions=torch.zeros(b,t,n,d), teacher=torch.zeros(b,t,n,d),
               global_features=torch.zeros(b,t,d), timestamps=torch.zeros(b,t),
               valid=torch.zeros(b,t,dtype=torch.bool),
               region_valid=torch.zeros(b,t,n,dtype=torch.bool),
               degenerate=torch.zeros(b,t,n,dtype=torch.bool),
               labels=torch.tensor([x["label"] for x in batch]),
               animal_ids=[x["animal_id"] for x in batch])
    for i,x in enumerate(batch):
        q=x["regions"].shape[0]
        for key in ("regions","teacher","global_features","timestamps","region_valid","degenerate"):
            out[key][i,:q]=x[key]
        out["valid"][i,:q]=True
    return out


def loader(rows, task, cfg, shuffle, seed, variant="baseline"):
    g=torch.Generator().manual_seed(int(seed))
    return DataLoader(CachedAnimals(rows, task, variant, seed), batch_size=int(cfg.get("batch_size",4)),
                      shuffle=shuffle, num_workers=int(cfg.get("num_workers",2)),
                      collate_fn=collate, generator=g,
                      pin_memory=torch.cuda.is_available())


def nearest_future_indices(ts, valid, horizon, tolerance):
    """Return target index for every origin; -1 means no timestamp match."""
    result=torch.full_like(ts, -1, dtype=torch.long)
    for b in range(ts.shape[0]):
        ids=torch.where(valid[b])[0]
        for i in ids.tolist():
            candidates=ids[ids>i]
            if not len(candidates): continue
            err=(ts[b,candidates]-(ts[b,i]+horizon)).abs()
            k=int(torch.argmin(err))
            if float(err[k]) <= tolerance: result[b,i]=candidates[k]
    return result


class DeltaScaler:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def to(self, device):
        return DeltaScaler(self.mean.to(device), self.std.to(device))
    def state(self): return {"mean":self.mean.cpu(),"std":self.std.cpu(),"horizons":HORIZONS}


def fit_scaler(rows, tolerance, reliable_mask=True):
    sums=sumsq=count=None
    for row in rows:
        p=torch.load(row["cache_path"],map_location="cpu")
        x=p.get("teacher_region_features",p["region_features"]).float()
        ts=p["timestamps"].float()
        region_valid=p.get("region_valid_mask",torch.ones(x.shape[:2],dtype=torch.bool)).bool()
        if not reliable_mask:
            region_valid=torch.ones_like(region_valid)
        valid=torch.ones(1,len(ts),dtype=torch.bool)
        if sums is None:
            d=x.shape[-1]; sums=torch.zeros(len(HORIZONS),d); sumsq=sums.clone(); count=torch.zeros(len(HORIZONS),1)
        for hi,h in enumerate(HORIZONS):
            j=nearest_future_indices(ts[None],valid,h,tolerance)[0]
            origins=torch.where(j>=0)[0]
            if len(origins):
                delta=x[j[origins]]-x[origins]
                region_mask=region_valid[origins] & region_valid[j[origins]]
                values=delta[region_mask]
                if len(values):
                    values=values.reshape(-1,d)
                    sums[hi]+=values.sum(0); sumsq[hi]+=(values*values).sum(0); count[hi]+=len(values)
    if sums is None: raise ValueError("cannot fit delta scaler on empty training rows")
    mean=sums/count.clamp_min(1); var=sumsq/count.clamp_min(1)-mean.square()
    return DeltaScaler(mean, var.clamp_min(1e-8).sqrt())


def future_loss(model, state, batch, scaler, tolerance):
    per_animal=[]
    ts=batch["timestamps"]; valid=batch["valid"]; teacher=batch["teacher"]
    rv=batch["region_valid"]
    for b in range(state.shape[0]):
        terms=[]
        for hi,h in enumerate(HORIZONS):
            target_index=nearest_future_indices(ts[b:b+1],valid[b:b+1],h,tolerance)[0]
            origins=torch.where(target_index>=0)[0]
            if not len(origins): continue
            targets=(teacher[b,target_index[origins]]-teacher[b,origins]-scaler.mean[hi])/scaler.std[hi]
            pred=model.predict_delta(state[b,origins],h)
            mask=rv[b,origins] & rv[b,target_index[origins]]
            if mask.any():
                err=torch.nn.functional.smooth_l1_loss(pred,targets,reduction="none").mean(-1)
                terms.append(err[mask].mean())
        if terms: per_animal.append(torch.stack(terms).mean())
    return torch.stack(per_animal).mean() if per_animal else state.sum()*0.0


def build_model(dim, cfg, device, variant="baseline", seed=42):
    m=cfg["model"]
    adjacency = None
    if variant_label(variant) == "random_graph":
        adjacency = degree_preserving_random_graph(16, seed)
        base = StrictP123Model.adjacency(16, torch.device("cpu"))
        if not torch.equal(adjacency.sum(1), base.sum(1)):
            raise RuntimeError("degree-preserving graph construction changed the degree sequence")
    return AblationP123Model(dim, int(m.get("hidden_dim",128)), int(m.get("attention_heads",4)),
                             int(m.get("temporal_layers",2)), float(m.get("dropout",.1)),
                             False, HORIZONS, adjacency_override=adjacency,
                             no_global_cls=variant_label(variant) == "no_global_cls").to(device)


def forward_batch(model,batch,device,task,scaler,tolerance,lambda_pred):
    keys=("regions","teacher","global_features","timestamps","valid","region_valid","labels")
    x={k:batch[k].to(device) for k in keys}
    out=model(x["regions"],x["global_features"],x["timestamps"],x["valid"],x["region_valid"],True)
    cls=nn.functional.binary_cross_entropy_with_logits(out["logits"],x["labels"])
    pred=future_loss(model,out["state"],x,scaler,tolerance) if task in ("p2","p3") else cls*0
    return out,cls+lambda_pred*pred,cls,pred,x


def safe_auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan


def evaluate(model,dl,device,task,scaler,tolerance,lambda_pred,collect_state=False):
    model.eval(); ys=[]; ps=[]; ids=[]; states=[]; losses=[]; degenerates=[]
    with torch.no_grad():
        for batch in dl:
            out,total,cls,pred,x=forward_batch(model,batch,device,task,scaler,tolerance,lambda_pred)
            ys.extend(x["labels"].cpu().numpy()); ps.extend(torch.sigmoid(out["logits"]).cpu().numpy())
            ids.extend(batch["animal_ids"]); losses.append((float(total),float(cls),float(pred)))
            degenerates.append(batch["degenerate"].float().mean().item())
            if collect_state:
                w=x["region_valid"].float()*x["valid"][:,:,None].float()
                frame=(out["state"]*w[...,None]).sum(2)/w.sum(2,keepdim=True).clamp_min(1)
                states.append(((frame*x["valid"][:,:,None]).sum(1)/x["valid"].sum(1,keepdim=True).clamp_min(1)).cpu())
    a=np.asarray(losses) if losses else np.empty((0,3))
    return dict(label=np.asarray(ys,int),probability=np.asarray(ps,float),animal_id=ids,
                loss=float(a[:,0].mean()) if len(a) else np.nan,
                cls_loss=float(a[:,1].mean()) if len(a) else np.nan,
                pred_loss=float(a[:,2].mean()) if len(a) else np.nan,
                state=torch.cat(states).numpy() if states else None,
                frangi_degenerate_rate=float(np.mean(degenerates)) if degenerates else 0.0)


def train(model,train_dl,val_dl,cfg,device,task,scaler,tolerance,lr,lambda_pred,epochs,save_path):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=float(cfg.get("weight_decay",.01)))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(epochs,1)); patience=int(cfg.get("early_stopping_patience",15))
    best=-np.inf; best_epoch=1; stale=0; history=[]
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            _,total,cls,pred,_=forward_batch(model,batch,device,task,scaler,tolerance,lambda_pred)
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get("gradient_clip_norm",1.0))); opt.step()
            losses.append((float(total),float(cls),float(pred)))
        sched.step(); val=evaluate(model,val_dl,device,task,scaler,tolerance,lambda_pred)
        auc=safe_auc(val["label"],val["probability"]); score=auc if np.isfinite(auc) else -val["loss"]
        a=np.asarray(losses)
        history.append(dict(epoch=epoch,train_loss=float(a[:,0].mean()),train_cls_loss=float(a[:,1].mean()),
                            train_pred_loss=float(a[:,2].mean()),val_loss=val["loss"],val_auc=auc,
                            val_pred_loss=val["pred_loss"],learning_rate=opt.param_groups[0]["lr"]))
        if score>best:
            best=score; best_epoch=epoch; stale=0
            torch.save({"state_dict":model.state_dict(),"epoch":epoch},save_path)
        else: stale+=1
        if stale>=patience: break
    model.load_state_dict(torch.load(save_path,map_location=device)["state_dict"])
    return best_epoch,history


def train_fixed(model,dl,cfg,device,task,scaler,tolerance,lr,lambda_pred,epochs):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=float(cfg.get("weight_decay",.01)))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(epochs,1))
    for _ in range(epochs):
        model.train()
        for batch in dl:
            opt.zero_grad(set_to_none=True); _,loss,_,_,_=forward_batch(model,batch,device,task,scaler,tolerance,lambda_pred)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get("gradient_clip_norm",1.0))); opt.step()
        sched.step()


def metric_dict(y,p,threshold):
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),auc=safe_auc(y,p),ap=float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan,
                accuracy=float((pred==y).mean()),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                f1=float(f1_score(y,pred,zero_division=0)),sensitivity=float(tp/max(tp+fn,1)),
                specificity=float(tn/max(tn+fp,1)),brier=float(brier_score_loss(y,p)),threshold=threshold)


def choose_threshold(y,p):
    candidates=[(metric_dict(y,p,float(t)),float(t)) for t in np.arange(0,1.001,.01)]
    return max(candidates,key=lambda z:(z[0]["balanced_accuracy"],z[0]["f1"],-abs(z[1]-.5)))[1]


def unique_animals(frame):
    if frame["animal_id"].duplicated().any():
        bad=frame.groupby("animal_id")["label"].nunique()
        if (bad>1).any(): raise ValueError("same animal_id has conflicting labels")
        frame=frame.sort_values(["animal_id","video_id"]).drop_duplicates("animal_id",keep="first")
    return frame.reset_index(drop=True)


def read_frame(config,task,manifest_override=None):
    out=Path(config["data"]["output_dir"])
    candidates=[manifest_override,
                out/("feature_extraction_manifest_p3.csv" if task=="p3" else "feature_extraction_manifest.csv")]
    path=next((Path(x) for x in candidates if x and Path(x).exists()),None)
    if path is None: raise FileNotFoundError(f"feature manifest not found; tried: {candidates}")
    frame=pd.read_csv(path,encoding="utf-8-sig")
    frame["animal_id"]=frame["animal_id"].astype(str); frame["label"]=frame["label"].astype(int)
    if "cache_path" not in frame:
        cache=Path(config["data"]["cache_dir"]); frame["cache_path"]=frame["video_id"].map(lambda x:str((cache/f"{int(x):05d}.pt").resolve()))
    missing=[p for p in frame["cache_path"] if not Path(p).exists()]
    if missing: raise FileNotFoundError(f"missing {len(missing)} cache files, e.g. {missing[0]}")
    return unique_animals(frame),out/"split_manifest.csv"


def run_outer(frame, split, fold, config, task, device, output, variant="baseline"):
    variant = variant_label(variant)
    tr=config["training"]; seed=int(tr.get("seed",42))+fold*1000
    sf=split[split["outer_fold"].astype(int)==fold]
    test_ids=set(sf.loc[sf["role"]=="outer_test","animal_id"].astype(str))
    train_ids=set(sf.loc[sf["role"]!="outer_test","animal_id"].astype(str))
    if train_ids & test_ids: raise ValueError(f"outer fold {fold}: animal leakage in split manifest")
    outer_train=frame[frame.animal_id.isin(train_ids)].reset_index(drop=True)
    outer_test=frame[frame.animal_id.isin(test_ids)].reset_index(drop=True)
    if outer_train.empty or outer_test.empty: raise ValueError(f"outer fold {fold} is empty")
    available_inner_folds=int(outer_train.label.value_counts().min())
    if available_inner_folds<4:
        raise ValueError(
            f"outer fold {fold} has only {available_inner_folds} samples in the minority class; "
            "the unified protocol requires exactly 4 inner folds"
        )
    inner_n=4
    inner=list(StratifiedGroupKFold(inner_n,shuffle=True,random_state=seed).split(
        outer_train,outer_train.label,groups=outer_train.animal_id))
    lrs=[float(x) for x in tr.get("learning_rate_candidates",[1e-4,3e-4])]
    if task == "p1" or variant == "no_pred_loss":
        lambdas = [0.0]
    else:
        lambdas = [float(x) for x in tr.get("lambda_pred_candidates", [0, 0.1, 1.0])]
    tolerance=float(tr.get("horizon_tolerance_seconds",0.26))
    dim=int(torch.load(outer_train.iloc[0].cache_path,map_location="cpu")["region_features"].shape[-1])
    candidate_results=[]
    fold_root=output/f"outer_{fold}"; fold_root.mkdir(parents=True,exist_ok=True)
    for lr in lrs:
        for lam in lambdas:
            parts=[]; epochs=[]
            tag=f"lr_{lr:g}_lambda_{lam:g}"; candidate_dir=fold_root/tag; candidate_dir.mkdir(exist_ok=True)
            for k,(fi,vi) in enumerate(inner,1):
                fit=outer_train.iloc[fi].to_dict("records"); val=outer_train.iloc[vi].to_dict("records")
                scaler=fit_scaler(fit,tolerance,variant!="no_reliable_mask").to(device) if task!="p1" else DeltaScaler(torch.zeros(3,dim,device=device),torch.ones(3,dim,device=device))
                seed_all(seed+k); model=build_model(dim, config, device, variant, seed + k)
                td=loader(fit,task,tr,True,seed+k,variant); vd=loader(val,task,tr,False,seed+k,variant)
                ckpt=candidate_dir/f"inner_{k}.pt"
                ep,hist=train(model,td,vd,tr,device,task,scaler,tolerance,lr,lam,int(tr.get("epochs",100)),ckpt)
                pd.DataFrame(hist).to_csv(candidate_dir/f"inner_{k}_history.csv",index=False)
                pred=evaluate(model,vd,device,task,scaler,tolerance,lam)
                parts.append(pd.DataFrame(dict(animal_id=pred["animal_id"],label=pred["label"],probability=pred["probability"],inner_fold=k)))
                epochs.append(ep)
            oof=pd.concat(parts,ignore_index=True); auc=safe_auc(oof.label,oof.probability)
            candidate_results.append(dict(lr=lr,lambda_pred=lam,auc=auc,oof=oof,epochs=epochs,tag=tag))
    best=max(candidate_results,key=lambda x:(-np.inf if not np.isfinite(x["auc"]) else x["auc"],-x["lambda_pred"]))
    best["oof"].to_csv(fold_root/"inner_oof.csv",index=False,encoding="utf-8-sig")
    threshold=choose_threshold(best["oof"].label.to_numpy(),best["oof"].probability.to_numpy())
    final_epochs=max(1,int(round(np.median(best["epochs"])))); rows=outer_train.to_dict("records")
    scaler=fit_scaler(rows,tolerance,variant!="no_reliable_mask").to(device) if task!="p1" else DeltaScaler(torch.zeros(3,dim,device=device),torch.ones(3,dim,device=device))
    seed_all(seed+99); model=build_model(dim, config, device, variant, seed + 99)
    train_fixed(model,loader(rows,task,tr,True,seed+99,variant),tr,device,task,scaler,tolerance,best["lr"],best["lambda_pred"],final_epochs)
    pred=evaluate(model,loader(outer_test.to_dict("records"),task,tr,False,seed+500,variant),device,task,scaler,tolerance,best["lambda_pred"],True)
    result=pd.DataFrame(dict(animal_id=pred["animal_id"],label=pred["label"],probability=pred["probability"],outer_fold=fold,threshold=threshold))
    result["prediction"]=(result.probability>=threshold).astype(int)
    for j in range(pred["state"].shape[1]): result[f"state_{j:03d}"]=pred["state"][:,j]
    result.to_csv(fold_root/"test_predictions.csv",index=False,encoding="utf-8-sig")
    torch.save(dict(task=task,state_dict=model.state_dict(),feature_dim=dim,threshold=threshold,
                    scaler=scaler.state(),learning_rate=best["lr"],lambda_pred=best["lambda_pred"],
                    final_epochs=final_epochs,outer_fold=fold,test_animal_ids=sorted(test_ids)),fold_root/"final_bundle.pt")

    summary={}

    # These are matched test-time perturbations of the same trained model.
    # They deliberately do not replace B4, which is a separately retrained
    # permutation-invariant model.
    for perturbation in ("shuffle", "reverse"):
        pert = evaluate(
            model,
            loader(outer_test.to_dict("records"), task, tr, False, seed + 700, perturbation),
            device, "p1", scaler, tolerance, 0.0, False,
        )
        perturb_result = pd.DataFrame({
            "animal_id": pert["animal_id"], "label": pert["label"],
            "probability": pert["probability"], "outer_fold": fold,
            "threshold": threshold,
        })
        perturb_result["prediction"] = (perturb_result.probability >= threshold).astype(int)
        perturb_result.to_csv(fold_root / f"order_{perturbation}_test_predictions.csv", index=False, encoding="utf-8-sig")
        summary.update({f"{perturbation}_{k}": v for k, v in metric_dict(pert["label"], pert["probability"], threshold).items()})

    future_detail = pd.DataFrame()
    future_summary = pd.DataFrame()
    if task in ("p2", "p3"):
        if task == "p2":
            future_detail, future_summary, future_diagnostics, future_controls = evaluate_future_acceptance(
                model, outer_train.to_dict("records"), outer_test.to_dict("records"),
                scaler, tolerance, device, seed,
            )
            future_diagnostics.to_csv(fold_root / "future_variance_diagnostics.csv", index=False, encoding="utf-8-sig")
            future_controls.to_csv(fold_root / "future_control_definitions.csv", index=False, encoding="utf-8-sig")
            future_summary.to_csv(fold_root / "future_acceptance_summary.csv", index=False, encoding="utf-8-sig")
        else:
            future_detail, future_summary = evaluate_future(
                model, outer_train.to_dict("records"), outer_test.to_dict("records"),
                scaler, tolerance, device,
            )
        future_detail.to_csv(fold_root / "future_prediction_details.csv", index=False, encoding="utf-8-sig")
        future_summary.to_csv(fold_root / "future_baseline_summary.csv", index=False, encoding="utf-8-sig")

    causal_check = causal_prefix_check(model, outer_test.iloc[0].to_dict(), device)
    (fold_root / "property_checks.json").write_text(
        json.dumps(causal_check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary.update(dict(variant=variant,outer_fold=fold,inner_folds=inner_n,
                        learning_rate=best["lr"],lambda_pred=best["lambda_pred"],inner_auc=best["auc"],
                        final_epochs=final_epochs,frangi_degenerate_rate=pred["frangi_degenerate_rate"],
                        causal_prefix_check_passed=causal_check["passed"],
                        causal_max_prefix_state_difference=causal_check["max_prefix_state_difference"],
                        causal_max_prefix_future_input_difference=causal_check["max_prefix_future_input_difference"]))
    summary.update({f"test_{k}":v for k,v in metric_dict(pred["label"],pred["probability"],threshold).items()})
    return result,summary,future_detail,future_summary,causal_check


def main(default_task=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config_v1.yaml")
    ap.add_argument("--task",choices=["p1","p2","p3"],default=default_task or "p1")
    ap.add_argument("--feature-manifest"); ap.add_argument("--split-manifest")
    ap.add_argument("--ablation", choices=["all", *ABLATION_NAMES], default="all",
                    help="run one ablation or the complete P2/P3 ablation suite")
    ap.add_argument("--preprocess-sensitivity", action="store_true",
                    help="re-encode raw sidecars for unregistered and brightness-corrected runs")
    ap.add_argument("--device",default="auto",
                    help="auto, cpu, or cuda; auto falls back when the GPU architecture is unsupported")
    args=ap.parse_args(); config=load_config(args.config)
    if args.task == "p3" and args.preprocess_sensitivity:
        raise ValueError("P3 ablation plan only includes Frangi and reliable-mask ablations; do not use --preprocess-sensitivity")
    config.setdefault("training", {})["inner_folds_actual"] = 4
    seed=int(config["training"].get("seed",42)); seed_all(seed)
    device = resolve_device(args.device)
    started=time.monotonic(); resources=resource_snapshot(device)
    frame,default_split=read_frame(config,args.task,args.feature_manifest)
    frames_by_variant = {"baseline": frame, "no_pred_loss": frame,
                         "mean_region": frame, "random_graph": frame}
    if args.preprocess_sensitivity:
        if not args.feature_manifest:
            raise ValueError("--preprocess-sensitivity requires --feature-manifest with raw_frames_path sidecars")
        preprocess_manifests = {}
        for mode in ("raw_unregistered", "brightness_corrected"):
            preprocess_manifests[mode] = materialize_preprocess_variant(
                config, args.task, args.feature_manifest, mode,
                device, int(config["training"].get("batch_size", 4)),
            )
            frames_by_variant[mode], _ = read_frame(config, args.task, preprocess_manifests[mode])
    feature_manifest_path=(Path(args.feature_manifest).resolve()
                           if args.feature_manifest else None)
    split_manifest_path=Path(args.split_manifest or default_split).resolve()
    split=pd.read_csv(split_manifest_path,encoding="utf-8-sig")
    split["animal_id"]=split["animal_id"].astype(str)
    folds=sorted(split["outer_fold"].dropna().astype(int).unique())
    if args.ablation == "all":
        if args.task == "p1":
            # Matched P1 comparison: identical split, budget and selection
            # rules; only the global CLS branch is removed in the ablation.
            variants = ["baseline", "no_global_cls"]
        elif args.task == "p3":
            # P3's planned ablation is deliberately limited to the two
            # independent components of the full P3 representation.
            variants = ["baseline", "no_frangi_weight", "no_reliable_mask"]
        else:
            variants = ["baseline", "no_pred_loss", "random_graph"]
            if args.preprocess_sensitivity:
                variants.extend(["raw_unregistered", "brightness_corrected"])
    else:
        variants = [args.ablation]
    p3_only = {"mean_region", "no_frangi_weight", "no_reliable_mask"}
    if args.task != "p3" and any(v in p3_only for v in variants):
        raise ValueError("Frangi/mask ablations are only valid for --task p3")
    if any(v in ("raw_unregistered", "brightness_corrected") for v in variants) and not args.preprocess_sensitivity:
        raise ValueError("preprocessing ablations require --preprocess-sensitivity")

    root = Path(config["data"]["output_dir"])/args.task
    suite_rows = []
    for variant in variants:
        output = root if variant == "baseline" else root/"ablations"/variant
        output.mkdir(parents=True,exist_ok=True)
        predictions=[]; summaries=[]; future_details=[]; future_summaries=[]; future_diagnostics=[]; future_controls=[]; property_checks=[]
        active_frame = frames_by_variant.get(variant, frame)
        for fold in folds:
            p,s,fd,fs,check=run_outer(active_frame,split,fold,config,args.task,device,output,variant)
            predictions.append(p); summaries.append(s); property_checks.append(check)
            if not fd.empty: future_details.append(fd.assign(outer_fold=fold, variant=variant))
            if not fs.empty: future_summaries.append(fs.assign(outer_fold=fold, variant=variant))
            if args.task == "p2":
                diag_path = output / f"outer_{fold}" / "future_variance_diagnostics.csv"
                control_path = output / f"outer_{fold}" / "future_control_definitions.csv"
                if diag_path.exists(): future_diagnostics.append(pd.read_csv(diag_path).assign(outer_fold=fold, variant=variant))
                if control_path.exists(): future_controls.append(pd.read_csv(control_path).assign(outer_fold=fold, variant=variant))
        allp=pd.concat(predictions,ignore_index=True)
        if allp.animal_id.duplicated().any(): raise RuntimeError("outer-test output contains duplicate animals")
        allp.to_csv(output/"all_outer_test_predictions.csv",index=False,encoding="utf-8-sig")
        summary_frame=pd.DataFrame(summaries)
        summary_frame.to_csv(output/"outer_fold_summary.csv",index=False,encoding="utf-8-sig")
        if future_details:
            pd.concat(future_details, ignore_index=True).to_csv(output/"future_prediction_details.csv", index=False, encoding="utf-8-sig")
        if future_summaries:
            pd.concat(future_summaries, ignore_index=True).to_csv(output/"future_baseline_summary.csv", index=False, encoding="utf-8-sig")
        if future_details and args.task == "p2":
            pooled_detail = pd.concat(future_details, ignore_index=True)
            summarize_future_acceptance_details(pooled_detail).to_csv(
                output / "future_acceptance_summary.csv", index=False, encoding="utf-8-sig"
            )
            pooled_detail.to_csv(output / "future_prediction_details.csv", index=False, encoding="utf-8-sig")
        if future_diagnostics:
            pd.concat(future_diagnostics, ignore_index=True).to_csv(
                output / "future_variance_diagnostics.csv", index=False, encoding="utf-8-sig"
            )
        if future_controls:
            pd.concat(future_controls, ignore_index=True).drop_duplicates(
                subset=["control", "interpretation"]
            ).to_csv(output / "future_control_definitions.csv", index=False, encoding="utf-8-sig")
        (output/"property_checks.json").write_text(json.dumps(property_checks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        split_seed=(split["split_seed"].dropna().iloc[0]
                    if "split_seed" in split.columns and split["split_seed"].notna().any() else None)
        split_version=(split["split_version"].dropna().iloc[0]
                       if "split_version" in split.columns and split["split_version"].notna().any() else None)
        resolved=copy.deepcopy(config)
        resolved["experiment"]={
            "task":args.task, "ablation":variant,
            "global_cls_branch": variant != "no_global_cls",
            "outer_folds":[int(fold) for fold in folds], "n_outer_folds":int(len(folds)),
            "inner_folds":4, "n_repeats":1, "repeat_id":0, "seed":seed,
            "animal_denominator":"unique_animals_in_frame_and_outer_test_once",
            "threshold_selection":THRESHOLD_RULE, "probability_calibration":"not_applied",
            "config_path":str(Path(args.config).resolve()),
            "feature_manifest":str(feature_manifest_path) if feature_manifest_path else None,
            "split_manifest":str(split_manifest_path),
            "future_prediction_acceptance": {
                "enabled": args.task == "p2",
                "horizons_seconds": [0.5, 1.0, 2.0],
                "target_source": "frozen_teacher_region_features",
                "input_policy": "prefix_only_at_each_origin",
                "timestamp_policy": "original_timestamps_preserved_no_reindex_after_duplicate_handling",
                "methods": ["hold_zero", "train_mean_change", "linear", "ridge", "current_predictor", "temporal_reverse", "target_shuffle_cross_animal"],
                "interval_unit": "animal",
            },
        }
        with (output/"resolved_config.yaml").open("w",encoding="utf-8") as f: yaml.safe_dump(resolved,f,allow_unicode=True,sort_keys=False)
        required_metrics=["test_accuracy","test_balanced_accuracy","test_auc","test_ap","test_f1",
                          "test_sensitivity","test_specificity","test_brier","test_threshold"]
        manifest=dict(
            task=args.task, ablation=variant,
            global_cls_branch=variant != "no_global_cls",
            outer_folds=[int(fold) for fold in folds], n_outer_folds=int(len(folds)),
            inner_folds=4, n_repeats=1, repeat_id=0, seed=seed,
            split_seed=split_seed, split_version=split_version,
            n_unique_animals=int(active_frame.animal_id.nunique()),
            animal_denominator=int(active_frame.animal_id.nunique()),
            label_counts={str(k):int(v) for k,v in active_frame.groupby("animal_id")["label"].first().value_counts().sort_index().items()},
            n_outer_test_predictions=int(len(allp)), no_duplicate_outer_predictions=True,
            prediction_file="all_outer_test_predictions.csv", outer_summary_file="outer_fold_summary.csv",
            per_animal_outer_prediction=True, reported_metrics=required_metrics,
            horizons_seconds=HORIZONS, threshold_selection=THRESHOLD_RULE,
            probability_calibration="not_applied",
            config_path=str(Path(args.config).resolve()), config_sha256=sha256_file(args.config),
            config_version="sha256_recorded_in_run_manifest",
            feature_manifest=str(feature_manifest_path) if feature_manifest_path else None,
            feature_manifest_sha256=sha256_file(feature_manifest_path) if feature_manifest_path else None,
            split_manifest=str(split_manifest_path), split_manifest_sha256=sha256_file(split_manifest_path),
            backbone=file_fingerprint(config.get("backbone",{}).get("model_name","")),
            weights_version=file_fingerprint(config.get("backbone",{}).get("model_name","")),
            causal_prefix_checks_passed=bool(all(x["passed"] for x in property_checks)),
            future_baselines_written=bool(future_details), mean_outer_auc=float(summary_frame.test_auc.mean()),
            future_acceptance_written=bool(args.task == "p2" and future_details),
            future_acceptance_files=(
                ["future_acceptance_summary.csv", "future_prediction_details.csv",
                 "future_variance_diagnostics.csv", "future_control_definitions.csv"]
                if args.task == "p2" else []
            ),
            run_seconds=float(time.monotonic()-started), resource=resources)
        (output/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        suite_rows.append({"task":args.task,"ablation":variant,"output_dir":str(output),
                           "mean_outer_auc":manifest["mean_outer_auc"],
                           "causal_prefix_checks_passed":manifest["causal_prefix_checks_passed"]})
        print(json.dumps(manifest,ensure_ascii=False,indent=2))
    root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(suite_rows).to_csv(root/"ablation_summary.csv",index=False,encoding="utf-8-sig")
    (root/"ablation_manifest.json").write_text(json.dumps({"task":args.task,"variants":variants,
        "summary":"ablation_summary.csv"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__": main()
