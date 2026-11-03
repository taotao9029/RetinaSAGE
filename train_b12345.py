"""Strict implementations of B1--B5 from 模型算法任务书.md.

B1/B2/B3 are frozen-feature logistic-regression baselines. B4/B5 are small
trainable probes on the same frozen DINOv3 global features. All selection and
normalisation are fitted inside the training animals only.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, subprocess, time
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
from torch.utils.data import DataLoader, Dataset
from train_p123 import load_config, read_frame, seed_all


TASKS = {"b1":"single_frame", "b2":"pixel_mean", "b3":"feature_mean",
         "b4":"deep_sets", "b5":"gru"}

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
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {"path": str(path), "exists": True, "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size}
    files = []
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        candidate = path / name
        if candidate.exists():
            files.append(file_fingerprint(candidate))
    return {"path": str(path), "exists": True, "files": files}


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


def auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan

def metrics(y,p,threshold=.5):
    q=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
    return {"n":len(y),"auc":auc(y,p),"ap":float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan,
            "accuracy":float((q==y).mean()),"balanced_accuracy":float(balanced_accuracy_score(y,q)),
            "f1":float(f1_score(y,q,zero_division=0)),"sensitivity":float(tp/max(tp+fn,1)),
            "specificity":float(tn/max(tn+fp,1)),"brier":float(brier_score_loss(y,p)),"threshold":threshold}

def threshold(y,p):
    xs=[(metrics(y,p,float(t)),float(t)) for t in np.arange(0,1.001,.01)]
    return max(xs,key=lambda x:(x[0]["balanced_accuracy"],x[0]["f1"],-abs(x[1]-.5)))[1]


class FeatureBank:
    def __init__(self,config,task):
        self.config,self.task,self.cache=config,task,{}
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder=None
    def _payload(self,row):
        p=torch.load(row["cache_path"],map_location="cpu")
        return p
    def _pixel_mean(self,row):
        p=self._payload(row); raw=p.get("raw_frames_path",row.get("raw_frames_path"))
        if not raw or not Path(raw).exists():
            raise FileNotFoundError("B2 requires raw_frames_path in the feature cache/manifest")
        frames=np.load(raw,allow_pickle=False)["frames"]
        image=np.asarray(frames,dtype=np.float32).mean(0).clip(0,255).astype(np.uint8)
        if self.encoder is None:
            from transformers import AutoImageProcessor,AutoModel
            name=self.config["backbone"]["model_name"]
            size=int(self.config["data"].get("image_size",256))
            self.processor=AutoImageProcessor.from_pretrained(name)
            self.encoder=AutoModel.from_pretrained(name).to(self.device).eval()
            self.image_size=size
            for q in self.encoder.parameters(): q.requires_grad_(False)
        from PIL import Image
        image=Image.fromarray(image[:,:,::-1].copy()).convert("RGB")
        try: inp=self.processor(images=image,return_tensors="pt",size={"height":self.image_size,"width":self.image_size},do_resize=True,do_center_crop=False)
        except TypeError: inp=self.processor(images=image,return_tensors="pt",size={"height":self.image_size,"width":self.image_size})
        inp={k:v.to(self.device) for k,v in inp.items()}
        with torch.no_grad(): out=self.encoder(**inp)
        return (out.pooler_output if getattr(out,"pooler_output",None) is not None else out.last_hidden_state[:,0]).squeeze(0).float().cpu().numpy()
    def get(self,row):
        key=(self.task,str(row["animal_id"]))
        if key in self.cache:return self.cache[key]
        p=self._payload(row)
        if self.task=="single_frame": x=p["global_features"].float()[0].numpy()
        elif self.task=="feature_mean": x=p["global_features"].float().mean(0).numpy()
        elif self.task=="pixel_mean": x=self._pixel_mean(row)
        elif self.task in ("deep_sets","gru"): x=p["global_features"].float().numpy()
        else: raise ValueError(self.task)
        self.cache[key]=x; return x


def rows_data(rows,bank):
    x=[bank.get(r) for r in rows]; y=np.asarray([int(r["label"]) for r in rows]); ids=[str(r["animal_id"]) for r in rows]
    return x,y,ids


class SequenceDataset(Dataset):
    def __init__(self,x,y,ids):self.x,self.y,self.ids=x,y,ids
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return torch.tensor(self.x[i],dtype=torch.float32),torch.tensor(float(self.y[i])),self.ids[i]

def seq_collate(batch):
    t=max(x.shape[0] for x,_,_ in batch); d=batch[0][0].shape[-1]
    x=torch.zeros(len(batch),t,d); m=torch.zeros(len(batch),t,dtype=torch.bool)
    for i,(z,_,_) in enumerate(batch):x[i,:len(z)]=z;m[i,:len(z)]=True
    return x,m,torch.tensor([y for _,y,_ in batch]),[i for _,_,i in batch]

class DeepSets(nn.Module):
    def __init__(self,d,h=128):
        super().__init__(); self.phi=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,h)); self.rho=nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,1))
    def forward(self,x,m):
        z=self.phi(x); z=(z*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1); return self.rho(z).squeeze(-1)

class GRUProbe(nn.Module):
    def __init__(self,d,h=128):
        super().__init__(); self.gru=nn.GRU(d,h,batch_first=True); self.head=nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,1))
    def forward(self,x,m):
        z,_=self.gru(x); last=m.sum(1).long().clamp_min(1)-1; return self.head(z[torch.arange(len(z),device=z.device),last]).squeeze(-1)

def neural_fit(train_x,train_y,val_x,val_y,kind,cfg,seed,device):
    d=train_x[0].shape[-1]; model=(DeepSets(d) if kind=="deep_sets" else GRUProbe(d)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg.get("learning_rate",3e-4)),weight_decay=.01)
    tr=DataLoader(SequenceDataset(train_x,train_y,[str(i) for i in range(len(train_y))]),batch_size=int(cfg.get("batch_size",8)),shuffle=True,collate_fn=seq_collate)
    va=DataLoader(SequenceDataset(val_x,val_y,[str(i) for i in range(len(val_y))]),batch_size=int(cfg.get("batch_size",8)),shuffle=False,collate_fn=seq_collate)
    best=-np.inf; best_state=None; stale=0; patience=int(cfg.get("early_stopping_patience",15)); best_epoch=1
    for epoch in range(1,int(cfg.get("epochs",100))+1):
        model.train()
        for x,m,y,_ in tr:
            x,m,y=x.to(device),m.to(device),y.to(device); opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(model(x,m),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval(); pp=[]
        with torch.no_grad():
            for x,m,_,_ in va:pp.extend(torch.sigmoid(model(x.to(device),m.to(device))).cpu().numpy())
        score=auc(val_y,np.asarray(pp))
        if np.isfinite(score) and score>best:best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=patience:break
    if best_state:model.load_state_dict(best_state)
    return model,best_epoch


def neural_fit_fixed(train_x, train_y, kind, cfg, epochs, seed, device):
    """Fit only on outer-training animals for an inner-CV-selected epoch count."""
    d = train_x[0].shape[-1]
    model = (DeepSets(d) if kind == "deep_sets" else GRUProbe(d)).to(device)
    seed_all(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 3e-4)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    tr = DataLoader(SequenceDataset(train_x, train_y, [str(i) for i in range(len(train_y))]), batch_size=int(cfg.get("batch_size", 8)), shuffle=True, collate_fn=seq_collate)
    for _ in range(max(1, int(epochs))):
        model.train()
        for x, m, y, _ in tr:
            x, m, y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x, m), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip_norm", 1.0)))
            opt.step()
    return model

def deepsets_permutation_error(model, x, device, seed=123):
    if not x:
        return 0.0
    rng = np.random.default_rng(seed)
    shuffled = [z[rng.permutation(len(z))] for z in x]
    original = neural_predict(model, x, device)
    permuted = neural_predict(model, shuffled, device)
    return float(np.max(np.abs(original - permuted)))

def neural_predict(model,x,device):
    model.eval(); out=[]
    with torch.no_grad():
        dl=DataLoader(SequenceDataset(x,[0]*len(x),[""]*len(x)),batch_size=32,collate_fn=seq_collate)
        for z,m,_,_ in dl:out.extend(torch.sigmoid(model(z.to(device),m.to(device))).cpu().numpy())
    return np.asarray(out)

def run_fold(frame,split,fold,config,task,output,device):
    sf=split[split.outer_fold.astype(int)==fold]; test=set(sf.loc[sf.role=="outer_test","animal_id"]); train=set(sf.loc[sf.role!="outer_test","animal_id"])
    fit=frame[frame.animal_id.isin(train)].reset_index(drop=True); testf=frame[frame.animal_id.isin(test)].reset_index(drop=True)
    inner_n=min(4,int(fit.label.value_counts().min())); seed=int(config["training"].get("seed",42))+fold*1000
    inner=list(StratifiedGroupKFold(inner_n,shuffle=True,random_state=seed).split(fit,fit.label,groups=fit.animal_id))
    bank=FeatureBank(config,TASKS[task]); bank.device=device; candidates=[.01,.1,1.,10.] if task in ("b1","b2","b3") else [1e-4,3e-4]; records=[]
    for cand in candidates:
        oof=[]; epochs=[]
        for k,(fi,vi) in enumerate(inner,1):
            rx,ry,ri=rows_data(fit.iloc[fi].to_dict("records"),bank); vx,vy,vid=rows_data(fit.iloc[vi].to_dict("records"),bank)
            if task in ("b1","b2","b3"):
                mu=np.mean(np.stack(rx),0); sd=np.std(np.stack(rx),0); sd[sd<1e-6]=1; z=(np.stack(rx)-mu)/sd; q=(np.stack(vx)-mu)/sd
                model=LogisticRegression(C=cand,max_iter=2000,solver="liblinear"); model.fit(z,ry); pp=model.predict_proba(q)[:,1]; ep=1
            else:
                model,ep=neural_fit(rx,ry,vx,vy,TASKS[task],dict(config["training"],learning_rate=cand),seed+k,device); pp=neural_predict(model,vx,device)
            oof.append(pd.DataFrame({"animal_id":vid,"label":vy,"probability":pp,"inner_fold":k}));epochs.append(ep)
        oof=pd.concat(oof,ignore_index=True); records.append((auc(oof.label,oof.probability),cand,oof,epochs))
    best=max(records,key=lambda z:(-np.inf if not np.isfinite(z[0]) else z[0],-z[1])); oof=best[2]; th=threshold(oof.label.to_numpy(),oof.probability.to_numpy())
    rx,ry,ri=rows_data(fit.to_dict("records"),bank); tx,ty,ti=rows_data(testf.to_dict("records"),bank)
    if task in ("b1","b2","b3"):
        mu=np.mean(np.stack(rx),0);sd=np.std(np.stack(rx),0);sd[sd<1e-6]=1;model=LogisticRegression(C=best[1],max_iter=2000,solver="liblinear");model.fit((np.stack(rx)-mu)/sd,ry);pp=model.predict_proba((np.stack(tx)-mu)/sd)[:,1];ep=1
    else:
        final_epochs = max(1, int(round(np.median(best[3]))))
        model = neural_fit_fixed(rx, ry, TASKS[task], dict(config["training"], learning_rate=best[1]), final_epochs, seed + 99, device)
        ep = final_epochs
        pp = neural_predict(model, tx, device)
    permutation_error = deepsets_permutation_error(model, tx, device) if task == "b4" else 0.0
    foldout=output/f"outer_{fold}";foldout.mkdir(parents=True,exist_ok=True);result=pd.DataFrame({"animal_id":ti,"label":ty,"probability":pp,"prediction":(pp>=th).astype(int),"outer_fold":fold,"threshold":th});result.to_csv(foldout/"test_predictions.csv",index=False,encoding="utf-8-sig");oof.to_csv(foldout/"inner_oof.csv",index=False)
    (foldout/"property_checks.json").write_text(
        json.dumps({"deepsets_permutation_max_probability_difference": permutation_error, "passed": permutation_error <= 1e-6}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result,{"outer_fold":fold,"selected":best[1],"inner_auc":best[0],"final_epochs":ep,"deepsets_permutation_max_probability_difference":permutation_error,"deepsets_permutation_check_passed":permutation_error<=1e-6,**{"test_"+k:v for k,v in metrics(ty,pp,th).items()}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--task",choices=list(TASKS),required=True);ap.add_argument("--config",default="config_v1.yaml");ap.add_argument("--feature-manifest");ap.add_argument("--split-manifest");ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");a=ap.parse_args()
    c=load_config(a.config)
    c.setdefault("training", {})["inner_folds_actual"] = 4
    seed = int(c["training"].get("seed", 42))
    seed_all(seed)
    frame,ds=read_frame(c,"p1",a.feature_manifest)
    split=pd.read_csv(a.split_manifest or ds,encoding="utf-8-sig")
    split.animal_id=split.animal_id.astype(str)
    device=torch.device(a.device)
    out=Path(c["data"]["output_dir"])/a.task
    out.mkdir(parents=True,exist_ok=True)
    ps=[];ss=[]
    started=time.monotonic()
    resources=resource_snapshot(device)
    outer_folds=sorted(split.outer_fold.dropna().astype(int).unique().tolist())
    for f in sorted(split.outer_fold.dropna().astype(int).unique()):
        p,s=run_fold(frame,split,f,c,a.task,out,device);ps.append(p);ss.append(s)
    allp=pd.concat(ps,ignore_index=True)
    assert not allp.animal_id.duplicated().any()
    summary=pd.DataFrame(ss)
    allp.to_csv(out/"all_outer_test_predictions.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(out/"outer_fold_summary.csv",index=False)
    checks={"task":a.task,"deepsets_permutation_checks_passed":bool(summary.get("deepsets_permutation_check_passed",pd.Series([True])).all()),"max_probability_difference":float(summary.get("deepsets_permutation_max_probability_difference",pd.Series([0.0])).max())}
    (out/"property_checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    config_path=Path(a.config).resolve()
    feature_manifest=Path(a.feature_manifest).resolve() if a.feature_manifest else None
    split_manifest=Path(a.split_manifest).resolve() if a.split_manifest else Path(ds).resolve()
    resolved=dict(c)
    resolved.setdefault("experiment", {})
    resolved["experiment"].update({
        "outer_folds": len(outer_folds),
        "inner_folds": 4,
        "n_repeats": 1,
        "seed": seed,
        "animal_denominator": int(frame.animal_id.nunique()),
        "threshold_selection": THRESHOLD_RULE,
        "probability_calibration": "not_applied",
        "config_path": str(config_path),
        "feature_manifest": str(feature_manifest) if feature_manifest else None,
        "split_manifest": str(split_manifest),
    })
    with (out/"resolved_config.yaml").open("w",encoding="utf-8") as handle:
        yaml.safe_dump(resolved,handle,allow_unicode=True,sort_keys=False)

    manifest={
        "task":a.task,
        "definition":TASKS[a.task],
        "outer_folds":outer_folds,
        "n_outer_folds":len(outer_folds),
        "inner_folds":4,
        "n_repeats":1,
        "repeat_id":0,
        "seed":seed,
        "split_seed":int(split["split_seed"].dropna().iloc[0]) if "split_seed" in split and split["split_seed"].notna().any() else None,
        "split_version":str(split["split_version"].dropna().iloc[0]) if "split_version" in split and split["split_version"].notna().any() else None,
        "n_unique_animals":int(frame.animal_id.nunique()),
        "animal_denominator":int(frame.animal_id.nunique()),
        "label_counts":{str(k):int(v) for k,v in frame["label"].value_counts().sort_index().items()},
        "n_outer_test_predictions":len(allp),
        "no_duplicate_outer_predictions":True,
        "prediction_file":"all_outer_test_predictions.csv",
        "outer_summary_file":"outer_fold_summary.csv",
        "threshold_selection":THRESHOLD_RULE,
        "probability_calibration":"not_applied",
        "config_path":str(config_path),
        "config_sha256":sha256_file(config_path),
        "feature_manifest":str(feature_manifest) if feature_manifest else None,
        "feature_manifest_sha256":sha256_file(feature_manifest) if feature_manifest and feature_manifest.exists() else None,
        "split_manifest":str(split_manifest),
        "split_manifest_sha256":sha256_file(split_manifest) if split_manifest.exists() else None,
        "backbone":file_fingerprint(c.get("backbone",{}).get("model_name","")),
        "mean_outer_auc":float(summary.test_auc.mean()),
        "run_seconds":time.monotonic()-started,
        "resource":resources,
        **checks,
    }
    (out/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__":main()
