"""Exact-B1-anchor dynamic residual experiment.

The static branch is the B1 first-frame global CLS logistic regression.  Only
the dynamic branch is trained.  Its input is a low-dimensional summary of
continuous frame changes relative to the first frame, and alpha is selected
inside each outer fold with alpha=0 as the exact B1 fallback.
"""
from __future__ import annotations

import argparse, json, random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from train_p123 import load_config


ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
PCA_DIMS = (8, 16)
LEARNING_RATES = (1e-4, 3e-4)


def seed_all(seed):
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def read_frame(config, path):
    f = pd.read_csv(path, encoding="utf-8-sig"); f["animal_id"] = f.animal_id.astype(str); f["label"] = f.label.astype(int)
    if f.animal_id.duplicated().any():
        if (f.groupby("animal_id").label.nunique() > 1).any(): raise ValueError("conflicting animal labels")
        f = f.sort_values(["animal_id", "video_id"]).drop_duplicates("animal_id")
    if "cache_path" not in f:
        cache = Path(config["data"]["cache_dir"]); f["cache_path"] = f.video_id.map(lambda x: str((cache/f"{int(x):05d}.pt").resolve()))
    if any(not Path(p).exists() for p in f.cache_path): raise FileNotFoundError("missing cache")
    return f.reset_index(drop=True)


@lru_cache(maxsize=512)
def global_sequence(path):
    p = torch.load(path, map_location="cpu")
    return p["global_features"].float().numpy().astype(np.float32)


def first_features(rows):
    return np.stack([global_sequence(str(r["cache_path"]))[0] for r in rows])


def fit_scaler(rows):
    x = first_features(rows); s = StandardScaler().fit(x); return s


def fit_delta_pca(rows, scaler, dim, seed):
    deltas=[]
    for r in rows:
        x = scaler.transform(global_sequence(str(r["cache_path"])))
        if len(x)>1: deltas.append(x[1:]-x[0])
    z=np.concatenate(deltas,axis=0) if deltas else np.zeros((1, scaler.mean_.shape[0]),np.float32)
    return PCA(n_components=min(dim,z.shape[0]-1,z.shape[1]),svd_solver="full",random_state=seed).fit(z)


def dynamic_summary(rows, scaler, pca):
    out=[]
    for r in rows:
        x=scaler.transform(global_sequence(str(r["cache_path"])))
        d=x[1:]-x[0] if len(x)>1 else np.zeros((1,x.shape[1]),np.float32)
        z=pca.transform(d)
        out.append(np.concatenate((z.mean(0),z.std(0),z[-1]),axis=0).astype(np.float32))
    return np.stack(out)


class ResidualNet(nn.Module):
    def __init__(self, dim, alpha):
        super().__init__(); self.alpha=float(alpha)
        self.body=nn.Sequential(nn.Linear(dim,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))
        nn.init.zeros_(self.body[-1].weight); nn.init.zeros_(self.body[-1].bias)
    def forward(self,x,anchor):
        correction=torch.tanh(self.body(x).squeeze(-1)); return anchor+self.alpha*correction, correction


def predict(model,x,anchor,device):
    model.eval()
    with torch.no_grad():
        logit,c=model(torch.tensor(x,dtype=torch.float32,device=device),torch.tensor(anchor,dtype=torch.float32,device=device))
    return torch.sigmoid(logit).cpu().numpy(), c.cpu().numpy()


def fit_dynamic(x,y,anchor_train,xv,yv,anchor_val,alpha,lr,epochs,seed,device):
    seed_all(seed); model=ResidualNet(x.shape[1],alpha).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    tx=torch.tensor(x,dtype=torch.float32,device=device); ta=torch.tensor(anchor_train,dtype=torch.float32,device=device); ty=torch.tensor(y,dtype=torch.float32,device=device)
    best=-np.inf; state=None; best_epoch=1; stale=0
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad(); logits,c=model(tx,ta); loss=nn.functional.binary_cross_entropy_with_logits(logits,ty)+.01*(alpha*c).square().mean(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        p,_=predict(model,xv,anchor_val,device); score=roc_auc_score(yv,p) if len(np.unique(yv))>1 else -float(loss.detach().cpu())
        if score>best: best=score; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=ep; stale=0
        else: stale+=1
        if stale>=15: break
    if state: model.load_state_dict(state)
    return model,best_epoch


def threshold(y,p):
    best=None
    for t in np.arange(0,1.001,.01):
        z=p>=t; key=(balanced_accuracy_score(y,z),f1_score(y,z,zero_division=0),-abs(t-.5))
        if best is None or key>best[0]: best=(key,float(t))
    return best[1]


def metric(y,p,t):
    z=p>=t; return {"n":len(y),"auc":roc_auc_score(y,p),"ap":average_precision_score(y,p),"accuracy":accuracy_score(y,z),"balanced_accuracy":balanced_accuracy_score(y,z),"f1":f1_score(y,z,zero_division=0),"brier":brier_score_loss(y,p),"threshold":t}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--feature-manifest",required=True); ap.add_argument("--split-manifest",required=True); ap.add_argument("--output",default="runs/B1_B5/b1_exact_residual"); ap.add_argument("--device",default="cpu"); args=ap.parse_args()
    cfg=load_config(args.config); frame=read_frame(cfg,args.feature_manifest); split=pd.read_csv(args.split_manifest,encoding="utf-8-sig"); split.animal_id=split.animal_id.astype(str); device=torch.device(args.device); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    b1sum=pd.read_csv("runs/B1_B5/b1/outer_fold_summary.csv",encoding="utf-8-sig"); summaries=[]; preds=[]
    for fold in sorted(split.outer_fold.dropna().astype(int).unique()):
        sf=split[split.outer_fold.astype(int)==fold]; test_ids=set(sf.loc[sf.role=="outer_test","animal_id"]); train_ids=set(sf.loc[sf.role!="outer_test","animal_id"]); tr=frame[frame.animal_id.isin(train_ids)].reset_index(drop=True); te=frame[frame.animal_id.isin(test_ids)].reset_index(drop=True); anchor_c=float(b1sum.loc[b1sum.outer_fold==fold,"selected"].iloc[0])
        # Reuse the persisted B1 inner animal assignment.  The split manifest
        # does not store inner memberships, so regenerating SGKF here can drift
        # from the historical B1 run and invalidate the alpha=0 fallback.
        b1_inner = pd.read_csv(f"runs/B1_B5/b1/outer_{fold}/inner_oof.csv", encoding="utf-8-sig")
        inner_map = dict(zip(b1_inner.animal_id.astype(str), b1_inner.inner_fold.astype(int)))
        inner=[]
        for k in range(1, 5):
            vi=np.asarray([i for i,a in enumerate(tr.animal_id.astype(str)) if inner_map.get(a)==k],dtype=int)
            fi=np.asarray([i for i in range(len(tr)) if i not in set(vi)],dtype=int)
            if len(vi)==0: raise ValueError(f"missing persisted B1 inner fold {fold}/{k}")
            inner.append((fi,vi))
        records=[]
        for alpha in ALPHAS:
            for dim in PCA_DIMS:
                for lr in LEARNING_RATES:
                    oof=[]; epochs=[]
                    for k,(fi,vi) in enumerate(inner,1):
                        fr=tr.iloc[fi].to_dict("records"); vr=tr.iloc[vi].to_dict("records"); sc=fit_scaler(fr); pca=fit_delta_pca(fr,sc,dim,42+fold+k); xs=dynamic_summary(fr,sc,pca); xv=dynamic_summary(vr,sc,pca); y=tr.iloc[fi].label.to_numpy(); yv=tr.iloc[vi].label.to_numpy(); anchor=LogisticRegression(C=anchor_c,max_iter=2000,solver="liblinear").fit(sc.transform(first_features(fr)),y); a=anchor.decision_function(sc.transform(first_features(fr))); av=anchor.decision_function(sc.transform(first_features(vr)))
                        if alpha==0: pp=1/(1+np.exp(-av)); ep=1
                        else: model,ep=fit_dynamic(xs,y,a,xv,yv,av,alpha,lr,80,42+fold+k,device); pp,_=predict(model,xv,av,device)
                        oof.append(pd.DataFrame({"animal_id":tr.iloc[vi].animal_id.to_numpy(),"label":yv,"probability":pp,"inner_fold":k})); epochs.append(ep)
                    oof=pd.concat(oof,ignore_index=True); records.append((roc_auc_score(oof.label,oof.probability),-alpha,oof,alpha,dim,lr,epochs))
        best=max(records,key=lambda z:(z[0],z[1])); _,_,oof,alpha,dim,lr,epochs=best; t=threshold(oof.label.to_numpy(),oof.probability.to_numpy()); foldout=out/f"outer_{fold}"; foldout.mkdir(exist_ok=True); oof.to_csv(foldout/"inner_oof.csv",index=False,encoding="utf-8-sig")
        fr=tr.to_dict("records"); ter=te.to_dict("records"); sc=fit_scaler(fr); pca=fit_delta_pca(fr,sc,dim,42+fold+99); xs=dynamic_summary(fr,sc,pca); xt=dynamic_summary(ter,sc,pca); y=tr.label.to_numpy(); yt=te.label.to_numpy(); anchor=LogisticRegression(C=anchor_c,max_iter=2000,solver="liblinear").fit(sc.transform(first_features(fr)),y); a=anchor.decision_function(sc.transform(first_features(fr))); at=anchor.decision_function(sc.transform(first_features(ter)))
        if alpha==0: p=1/(1+np.exp(-at)); residual=np.zeros(len(xt)); ep=1
        else: model=ResidualNet(xs.shape[1],alpha).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01); tx=torch.tensor(xs,dtype=torch.float32,device=device); ta=torch.tensor(a,dtype=torch.float32,device=device); ty=torch.tensor(y,dtype=torch.float32,device=device)
        if alpha!=0:
            for _ in range(max(1,int(round(np.median(epochs))))): model.train(); opt.zero_grad(); logits,c=model(tx,ta); loss=nn.functional.binary_cross_entropy_with_logits(logits,ty)+.01*(alpha*c).square().mean(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            p,residual=predict(model,xt,at,device); ep=max(1,int(round(np.median(epochs))))
        result=pd.DataFrame({"animal_id":te.animal_id,"label":yt,"probability":p,"prediction":p>=t,"dynamic_residual":residual,"outer_fold":fold,"alpha":alpha,"threshold":t}); result.to_csv(foldout/"test_predictions.csv",index=False,encoding="utf-8-sig"); preds.append(result); summaries.append({"outer_fold":fold,"anchor_C":anchor_c,"selected_alpha":alpha,"selected_pca_dim":dim,"selected_lr":lr,"inner_auc":best[0],"residual_mean_abs":float(np.mean(np.abs(residual)),),**{"test_"+k:v for k,v in metric(yt,p,t).items()}})
    summary=pd.DataFrame(summaries); summary.to_csv(out/"outer_fold_summary.csv",index=False); allp=pd.concat(preds,ignore_index=True); allp.to_csv(out/"all_outer_test_predictions.csv",index=False); manifest={"method":"exact_b1_anchor_dynamic_residual","outer_folds":[1,2,3,4,5],"inner_folds":4,"n_repeats":1,"animal_denominator":int(allp.animal_id.nunique()),"anchor":"exact B1 first-frame global CLS logistic regression","dynamic":"PCA summary of continuous global-feature differences relative to first frame","alpha_grid":ALPHAS,"selection":"inner OOF AUC; alpha=0 is exact B1 fallback","no_outer_test_used_for_selection":True,"mean_outer_auc":float(summary.test_auc.mean()),"mean_outer_accuracy":float(summary.test_accuracy.mean()),"mean_outer_balanced_accuracy":float(summary.test_balanced_accuracy.mean())}; (out/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
