"""Strict B8/B9 probes with the same animal-level split as B1--B6 and P1--P3."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_p123 import load_config, read_frame, seed_all
from train_b12345 import auc, metrics, threshold


class Probe(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Dropout(.1), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def load_features(rows):
    vectors=[]; labels=[]; ids=[]
    for row in rows:
        payload=torch.load(row["cache_path"],map_location="cpu")
        if "global_features" not in payload:
            raise KeyError(f"cache lacks global_features: {row['cache_path']}")
        vectors.append(payload["global_features"].float().mean(0).numpy())
        labels.append(int(row["label"])); ids.append(str(row["animal_id"]))
    return np.stack(vectors),np.asarray(labels),ids


def fit_probe(x,y,xv,yv,lr,epochs,device,seed):
    seed_all(seed); model=Probe(x.shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    train=DataLoader(TensorDataset(torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)),batch_size=8,shuffle=True)
    best=-np.inf; state=None; stale=0
    for epoch in range(1,epochs+1):
        model.train()
        for a,b in train:
            a,b=a.to(device),b.to(device);optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(a),b);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.no_grad(): pred=torch.sigmoid(model(torch.tensor(xv,dtype=torch.float32,device=device))).cpu().numpy()
        score=auc(yv,pred)
        if np.isfinite(score) and score>best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=15:break
    if state:model.load_state_dict(state)
    return model,best_epoch


def predict_probe(model,x,device):
    model.eval()
    with torch.no_grad(): return torch.sigmoid(model(torch.tensor(x,dtype=torch.float32,device=device))).cpu().numpy()


def fit_probe_fixed(x, y, lr, epochs, device, seed):
    """Fit B9 only on outer-training animals for a fixed inner-CV epoch count."""
    seed_all(seed)
    model = Probe(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    train = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=8, shuffle=True,
    )
    for _ in range(max(1, int(epochs))):
        model.train()
        for a, b in train:
            a, b = a.to(device), b.to(device)
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(a), b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def run_fold(frame,split,fold,task,config,out,device):
    sf=split[split.outer_fold.astype(int)==fold]
    test=set(sf.loc[sf.role=="outer_test","animal_id"]); train=set(sf.loc[sf.role!="outer_test","animal_id"])
    fit=frame[frame.animal_id.isin(train)].reset_index(drop=True); testf=frame[frame.animal_id.isin(test)].reset_index(drop=True)
    inner_n=min(4,int(fit.label.value_counts().min())); seed=int(config["training"].get("seed",42))+fold*1000
    inner=list(StratifiedGroupKFold(inner_n,shuffle=True,random_state=seed).split(fit,fit.label,groups=fit.animal_id))
    candidates=[.01,.1,1.,10.] if task=="b8" else [1e-4,3e-4]; records=[]
    for candidate in candidates:
        parts=[]; epochs=[]
        for k,(fi,vi) in enumerate(inner,1):
            x,y,ids=load_features(fit.iloc[fi].to_dict("records")); xv,yv,vid=load_features(fit.iloc[vi].to_dict("records"))
            if task=="b8":
                mu=x.mean(0);sd=x.std(0);sd[sd<1e-6]=1.;model=LogisticRegression(C=candidate,max_iter=2000,solver="liblinear");model.fit((x-mu)/sd,y);pred=model.predict_proba((xv-mu)/sd)[:,1];ep=1
            else:
                model,ep=fit_probe(x,y,xv,yv,candidate,int(config["training"].get("epochs",100)),device,seed+k);pred=predict_probe(model,xv,device)
            parts.append(pd.DataFrame({"animal_id":vid,"label":yv,"probability":pred,"inner_fold":k}));epochs.append(ep)
        oof=pd.concat(parts,ignore_index=True);records.append((auc(oof.label,oof.probability),candidate,oof,epochs))
    best=max(records,key=lambda z:(-np.inf if not np.isfinite(z[0]) else z[0],-z[1]));oof=best[2];th=threshold(oof.label.to_numpy(),oof.probability.to_numpy())
    x,y,ids=load_features(fit.to_dict("records"));xt,yt,tid=load_features(testf.to_dict("records"))
    if task=="b8":
        mu=x.mean(0);sd=x.std(0);sd[sd<1e-6]=1.;model=LogisticRegression(C=best[1],max_iter=2000,solver="liblinear");model.fit((x-mu)/sd,y);pred=model.predict_proba((xt-mu)/sd)[:,1];ep=1
    else:
        final_epochs = max(1, int(round(np.median(best[3]))))
        model = fit_probe_fixed(x, y, best[1], final_epochs, device, seed + 99)
        ep = final_epochs
        pred = predict_probe(model, xt, device)
    foldout=out/f"outer_{fold}";foldout.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame({"animal_id":tid,"label":yt,"probability":pred,"prediction":(pred>=th).astype(int),"outer_fold":fold,"threshold":th})
    result.to_csv(foldout/"test_predictions.csv",index=False,encoding="utf-8-sig");oof.to_csv(foldout/"inner_oof.csv",index=False)
    return result,{"outer_fold":fold,"selected":best[1],"inner_auc":best[0],"final_epochs":ep,**{"test_"+k:v for k,v in metrics(yt,pred,th).items()}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--task",choices=["b8","b9"],required=True);ap.add_argument("--config",default="config_v1.yaml");ap.add_argument("--feature-manifest");ap.add_argument("--split-manifest");ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");a=ap.parse_args()
    c=load_config(a.config);seed_all(int(c["training"].get("seed",42)));frame,ds=read_frame(c,"p1",a.feature_manifest);split=pd.read_csv(a.split_manifest or ds,encoding="utf-8-sig");split.animal_id=split.animal_id.astype(str);out=Path(c["data"]["output_dir"])/a.task;out.mkdir(parents=True,exist_ok=True);ps=[];ss=[]
    for fold in sorted(split.outer_fold.dropna().astype(int).unique()):
        p,s=run_fold(frame,split,fold,a.task,c,out,torch.device(a.device));ps.append(p);ss.append(s)
    allp=pd.concat(ps,ignore_index=True);assert not allp.animal_id.duplicated().any();summary=pd.DataFrame(ss);allp.to_csv(out/"all_outer_test_predictions.csv",index=False,encoding="utf-8-sig");summary.to_csv(out/"outer_fold_summary.csv",index=False);m={"task":a.task,"definition":"RETFound natureCFP mean pooling + logistic regression" if a.task=="b8" else "V-JEPA 2.1 frozen mean pooling + two-layer probe","n_unique_animals":int(frame.animal_id.nunique()),"n_outer_test_predictions":len(allp),"no_duplicate_outer_predictions":True,"mean_outer_auc":float(summary.test_auc.mean())};(out/"run_manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(m,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
