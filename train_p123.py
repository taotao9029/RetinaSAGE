"""Strict nested-CV runner for the P1, P2 and P3 tasks in 模型算法任务书.md.
# P1
  python train_p123.py \
    --task p1 \
    --config config_v1.yaml \
    --feature-manifest /path/to/P1/feature_extraction_manifest.csv \
    --split-manifest /path/to/split_manifest.csv

  # P2
  python train_p123.py \
    --task p2 \
    --config config_v1.yaml \
    --feature-manifest /path/to/P2/feature_extraction_manifest.csv \
    --split-manifest /path/to/split_manifest.csv

  # P3
  python train_p123.py \
    --task p3 \
    --config config_v1.yaml \
    --feature-manifest /path/to/feature_extraction_manifest_p3.csv \
    --split-manifest /path/to/split_manifest.csv
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
from model_p123 import StrictP123Model
from future_evaluation import HORIZONS as FUTURE_HORIZONS, evaluate_future, causal_prefix_check


HORIZONS = (0.5, 1.0, 2.0)


THRESHOLD_RULE = {
    "source": "inner_oof",
    "grid_start": 0.0,
    "grid_stop": 1.0,
    "grid_step": 0.01,
    "primary_metric": "balanced_accuracy",
    "tie_break_1": "f1",
    "tie_break_2": "closest_to_0.5",
}


def json_default(value):
    """Convert common NumPy/Pandas values to JSON-native types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def seed_all(seed):
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CachedAnimals(Dataset):
    def __init__(self, rows, task): self.rows, self.task = rows, task
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        p = torch.load(row["cache_path"], map_location="cpu")
        regions = p["region_features"].float()
        teacher = p.get("teacher_region_features", regions).float()
        valid = p.get("region_valid_mask", torch.ones(regions.shape[:2], dtype=torch.bool)).bool()
        degenerate = p.get("frangi_degenerate_mask", torch.zeros(regions.shape[:2], dtype=torch.bool)).bool()
        if self.task == "p3":
            if p.get("structure_weight_type") != "reference_frame_multiscale_frangi":
                raise ValueError(f"P3 cache is not fixed-reference Frangi: {row['cache_path']}")
            if p.get("structure_weight_fixed_across_frames") is not True:
                raise ValueError(f"P3 structure weights are not fixed: {row['cache_path']}")
        return dict(regions=regions, teacher=teacher,
                    global_features=p["global_features"].float(),
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


def loader(rows, task, cfg, shuffle, seed):
    g=torch.Generator().manual_seed(int(seed))
    return DataLoader(CachedAnimals(rows,task), batch_size=int(cfg.get("batch_size",4)),
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


def fit_scaler(rows, tolerance):
    sums=sumsq=count=None
    for row in rows:
        p=torch.load(row["cache_path"],map_location="cpu")
        x=p.get("teacher_region_features",p["region_features"]).float()
        ts=p["timestamps"].float()
        region_valid=p.get("region_valid_mask",torch.ones(x.shape[:2],dtype=torch.bool)).bool()
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


def build_model(dim,cfg,device,no_gat=False):
    m=cfg["model"]
    return StrictP123Model(dim,int(m.get("hidden_dim",128)),int(m.get("attention_heads",4)),
                           int(m.get("temporal_layers",2)),float(m.get("dropout",.1)),
                           no_gat,HORIZONS).to(device)


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


def run_outer(frame,split,fold,config,task,device,output):
    tr=config["training"]; seed=int(tr.get("seed",42))+fold*1000
    sf=split[split["outer_fold"].astype(int)==fold]
    test_ids=set(sf.loc[sf["role"]=="outer_test","animal_id"].astype(str))
    train_ids=set(sf.loc[sf["role"]!="outer_test","animal_id"].astype(str))
    if train_ids & test_ids: raise ValueError(f"outer fold {fold}: animal leakage in split manifest")
    outer_train=frame[frame.animal_id.isin(train_ids)].reset_index(drop=True)
    outer_test=frame[frame.animal_id.isin(test_ids)].reset_index(drop=True)
    if outer_train.empty or outer_test.empty: raise ValueError(f"outer fold {fold} is empty")
    inner_n=min(4,int(outer_train.label.value_counts().min()))
    if inner_n<2: raise ValueError(f"outer fold {fold} lacks samples for inner CV")
    inner=list(StratifiedGroupKFold(inner_n,shuffle=True,random_state=seed).split(
        outer_train,outer_train.label,groups=outer_train.animal_id))
    lrs=[float(x) for x in tr.get("learning_rate_candidates",[1e-4,3e-4])]
    lambdas=[0.0] if task=="p1" else [float(x) for x in tr.get("lambda_pred_candidates",[0,0.1,1.0])]
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
                scaler=fit_scaler(fit,tolerance).to(device) if task!="p1" else DeltaScaler(torch.zeros(3,dim,device=device),torch.ones(3,dim,device=device))
                seed_all(seed+k); model=build_model(dim,config,device)
                td=loader(fit,task,tr,True,seed+k); vd=loader(val,task,tr,False,seed+k)
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
    scaler=fit_scaler(rows,tolerance).to(device) if task!="p1" else DeltaScaler(torch.zeros(3,dim,device=device),torch.ones(3,dim,device=device))
    seed_all(seed+99); model=build_model(dim,config,device)
    train_fixed(model,loader(rows,task,tr,True,seed+99),tr,device,task,scaler,tolerance,best["lr"],best["lambda_pred"],final_epochs)
    pred=evaluate(model,loader(outer_test.to_dict("records"),task,tr,False,seed+500),device,task,scaler,tolerance,best["lambda_pred"],True)
    result=pd.DataFrame(dict(animal_id=pred["animal_id"],label=pred["label"],probability=pred["probability"],outer_fold=fold,threshold=threshold))
    result["prediction"]=(result.probability>=threshold).astype(int)
    for j in range(pred["state"].shape[1]): result[f"state_{j:03d}"]=pred["state"][:,j]
    result.to_csv(fold_root/"test_predictions.csv",index=False,encoding="utf-8-sig")
    torch.save(dict(task=task,state_dict=model.state_dict(),feature_dim=dim,threshold=threshold,
                    scaler=scaler.state(),learning_rate=best["lr"],lambda_pred=best["lambda_pred"],
                    final_epochs=final_epochs,outer_fold=fold,test_animal_ids=sorted(test_ids)),fold_root/"final_bundle.pt")

    future_detail = pd.DataFrame()
    future_summary = pd.DataFrame()
    if task in ("p2", "p3"):
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
    summary=dict(outer_fold=fold,learning_rate=best["lr"],lambda_pred=best["lambda_pred"],inner_auc=best["auc"],
                 final_epochs=final_epochs,frangi_degenerate_rate=pred["frangi_degenerate_rate"],
                 causal_prefix_check_passed=causal_check["passed"],
                 causal_max_prefix_state_difference=causal_check["max_prefix_state_difference"],
                 causal_max_prefix_future_input_difference=causal_check["max_prefix_future_input_difference"])
    summary.update({f"test_{k}":v for k,v in metric_dict(pred["label"],pred["probability"],threshold).items()})
    return result,summary,future_detail,future_summary,causal_check


def main(default_task=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config_v1.yaml")
    ap.add_argument("--task",choices=["p1","p2","p3"],default=default_task or "p1")
    ap.add_argument("--feature-manifest"); ap.add_argument("--split-manifest")
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args(); config=load_config(args.config)
    config.setdefault("training", {})["inner_folds_actual"] = 4
    seed=int(config["training"].get("seed",42)); seed_all(seed)
    frame,default_split=read_frame(config,args.task,args.feature_manifest)
    feature_manifest_path=(Path(args.feature_manifest).resolve()
                           if args.feature_manifest else None)
    split_manifest_path=Path(args.split_manifest or default_split).resolve()
    split=pd.read_csv(split_manifest_path,encoding="utf-8-sig")
    split["animal_id"]=split["animal_id"].astype(str)
    output=Path(config["data"]["output_dir"])/args.task; output.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device); started=time.monotonic(); resources=resource_snapshot(device)
    predictions=[]; summaries=[]; future_details=[]; future_summaries=[]; property_checks=[]
    folds=sorted(split["outer_fold"].dropna().astype(int).unique())
    for fold in folds:
        p,s,fd,fs,check=run_outer(frame,split,fold,config,args.task,device,output)
        predictions.append(p); summaries.append(s); property_checks.append(check)
        if not fd.empty: future_details.append(fd.assign(outer_fold=fold))
        if not fs.empty: future_summaries.append(fs.assign(outer_fold=fold))
    allp=pd.concat(predictions,ignore_index=True)
    if allp.animal_id.duplicated().any(): raise RuntimeError("outer-test output contains duplicate animals")
    allp.to_csv(output/"all_outer_test_predictions.csv",index=False,encoding="utf-8-sig")
    summary_frame=pd.DataFrame(summaries)
    summary_frame.to_csv(output/"outer_fold_summary.csv",index=False,encoding="utf-8-sig")
    if future_details:
        pd.concat(future_details, ignore_index=True).to_csv(output/"future_prediction_details.csv", index=False, encoding="utf-8-sig")
    if future_summaries:
        pd.concat(future_summaries, ignore_index=True).to_csv(output/"future_baseline_summary.csv", index=False, encoding="utf-8-sig")
    (output/"property_checks.json").write_text(
        json.dumps(property_checks, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    split_seed=(split["split_seed"].dropna().iloc[0]
                if "split_seed" in split.columns and split["split_seed"].notna().any() else None)
    split_version=(split["split_version"].dropna().iloc[0]
                   if "split_version" in split.columns and split["split_version"].notna().any() else None)
    config["experiment"]={
        "task":args.task, "outer_folds":[int(fold) for fold in folds],
        "n_outer_folds":int(len(folds)), "inner_folds":4,
        "n_repeats":1, "repeat_id":0, "seed":seed,
        "animal_denominator":"unique_animals_in_frame_and_outer_test_once",
        "threshold_selection":THRESHOLD_RULE, "probability_calibration":"not_applied",
        "config_path":str(Path(args.config).resolve()),
        "feature_manifest":str(feature_manifest_path) if feature_manifest_path else None,
        "split_manifest":str(split_manifest_path),
    }
    with (output/"resolved_config.yaml").open("w",encoding="utf-8") as f: yaml.safe_dump(config,f,allow_unicode=True,sort_keys=False)
    required_metrics=["test_accuracy","test_balanced_accuracy","test_auc","test_ap","test_f1",
                      "test_sensitivity","test_specificity","test_brier","test_threshold"]
    manifest=dict(
        task=args.task, outer_folds=[int(fold) for fold in folds], n_outer_folds=int(len(folds)),
        inner_folds=4, n_repeats=1, repeat_id=0, seed=seed,
        split_seed=split_seed, split_version=split_version,
        n_unique_animals=int(frame.animal_id.nunique()),
        animal_denominator=int(frame.animal_id.nunique()),
        label_counts={str(k):int(v) for k,v in frame.groupby("animal_id")["label"].first().value_counts().sort_index().items()},
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
        run_seconds=float(time.monotonic()-started), resource=resources)
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default)
    (output/"run_manifest.json").write_text(manifest_json + "\n", encoding="utf-8")
    print(manifest_json)


if __name__=="__main__": main()
