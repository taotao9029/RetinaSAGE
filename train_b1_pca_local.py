"""Nested B1 + low-dimensional local-feature fusion.

The B1 global CLS branch is kept intact.  Local Frangi-weighted regions are
mean/max pooled, PCA is fitted inside each inner training split, and a small
block weight gamma is selected by inner OOF AUC.
"""
from __future__ import annotations

import argparse
import json
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

from train_p123 import load_config


CS = (0.1, 1.0, 10.0)
GAMMAS = (0.0, 0.1, 0.25, 0.5)
PCA_DIMS = (8, 16)


def read_frame(config, manifest_path):
    frame = pd.read_csv(manifest_path, encoding="utf-8-sig")
    frame["animal_id"] = frame["animal_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    if frame["animal_id"].duplicated().any():
        if (frame.groupby("animal_id")["label"].nunique() > 1).any():
            raise ValueError("animal has conflicting labels")
        frame = frame.sort_values(["animal_id", "video_id"]).drop_duplicates("animal_id")
    missing = [p for p in frame["cache_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return frame.reset_index(drop=True)


@lru_cache(maxsize=512)
def cached_parts(cache_path):
    p = torch.load(cache_path, map_location="cpu")
    g = p["global_features"].float()[0].numpy().astype(np.float32)
    local = p["region_features"].float()[0]
    valid = p.get("region_valid_mask", torch.ones(local.shape[0], dtype=torch.bool))[0].bool()
    local = local * valid[:, None].float()
    mean = local.mean(0).numpy().astype(np.float32)
    maximum = local.max(0).values.numpy().astype(np.float32)
    return g, np.concatenate((mean, maximum)).astype(np.float32)


def parts(row):
    return cached_parts(str(row["cache_path"]))


def transform(fit_rows, query_rows, pca_dim, seed):
    fg = np.stack([parts(r)[0] for r in fit_rows])
    fl = np.stack([parts(r)[1] for r in fit_rows])
    qg = np.stack([parts(r)[0] for r in query_rows])
    ql = np.stack([parts(r)[1] for r in query_rows])
    gs, ls = StandardScaler().fit(fg), StandardScaler().fit(fl)
    fls, qls = ls.transform(fl), ls.transform(ql)
    pca = PCA(n_components=min(int(pca_dim), fls.shape[0] - 1, fls.shape[1]),
              svd_solver="full", random_state=seed).fit(fls)
    return gs.transform(fg), gs.transform(qg), pca.transform(fls), pca.transform(qls), gs, ls, pca


def compose(g, l, gamma):
    return np.concatenate((g, float(gamma) * l), axis=1)


def threshold(y, p):
    best = None
    for t in np.arange(0, 1.001, 0.01):
        pred = p >= t
        key = (balanced_accuracy_score(y, pred), f1_score(y, pred, zero_division=0), -abs(t - .5))
        if best is None or key > best[0]: best = (key, float(t))
    return best[1]


def metric(y, p, t):
    pred = p >= t
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p)), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)), "brier": float(brier_score_loss(y, p)),
            "threshold": float(t)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--feature-manifest", required=True); ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--output", default="runs/B1_B5/b1_plus_pca_local")
    args = ap.parse_args(); cfg = load_config(args.config); frame = read_frame(cfg, args.feature_manifest)
    split = pd.read_csv(args.split_manifest, encoding="utf-8-sig"); split["animal_id"] = split["animal_id"].astype(str)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); summaries=[]; predictions=[]
    for fold in sorted(split.outer_fold.dropna().astype(int).unique()):
        sf=split[split.outer_fold.astype(int)==fold]; test_ids=set(sf.loc[sf.role=="outer_test","animal_id"]); train_ids=set(sf.loc[sf.role!="outer_test","animal_id"])
        tr=frame[frame.animal_id.isin(train_ids)].reset_index(drop=True); te=frame[frame.animal_id.isin(test_ids)].reset_index(drop=True)
        inner=list(StratifiedGroupKFold(4,shuffle=True,random_state=42+fold*1000).split(tr,tr.label,groups=tr.animal_id))
        records=[]
        for c in CS:
            for gamma in GAMMAS:
                for dim in PCA_DIMS:
                    oof=[]
                    for k,(fi,vi) in enumerate(inner,1):
                        fr=tr.iloc[fi].to_dict("records"); vr=tr.iloc[vi].to_dict("records")
                        gtr,gva,ltr,lva,_,_,_=transform(fr,vr,dim,42+fold+k)
                        xtr=compose(gtr,ltr,gamma); xva=compose(gva,lva,gamma)
                        clf=LogisticRegression(C=c,max_iter=2000,solver="liblinear").fit(xtr,tr.iloc[fi].label.to_numpy())
                        oof.append(pd.DataFrame({"animal_id":tr.iloc[vi].animal_id.to_numpy(),"label":tr.iloc[vi].label.to_numpy(),"probability":clf.predict_proba(xva)[:,1],"inner_fold":k}))
                    oof=pd.concat(oof,ignore_index=True); auc=float(roc_auc_score(oof.label,oof.probability)); t=threshold(oof.label.to_numpy(),oof.probability.to_numpy()); bacc=balanced_accuracy_score(oof.label,oof.probability>=t)
                    records.append((auc,bacc,-abs(gamma),-c,oof,c,gamma,dim,t))
        best=max(records,key=lambda z:z[:4]); _,_,_,_,oof,c,gamma,dim,t=best
        foldout=out/f"outer_{fold}"; foldout.mkdir(exist_ok=True); oof.to_csv(foldout/"inner_oof.csv",index=False,encoding="utf-8-sig")
        fr=tr.to_dict("records"); te_rows=te.to_dict("records"); gtr,gte,ltr,lte,gs,ls,pca=transform(fr,te_rows,dim,42+fold+99)
        clf=LogisticRegression(C=c,max_iter=2000,solver="liblinear").fit(compose(gtr,ltr,gamma),tr.label.to_numpy())
        p=clf.predict_proba(compose(gte,lte,gamma))[:,1]; result=pd.DataFrame({"animal_id":te.animal_id,"label":te.label,"probability":p,"prediction":p>=t,"outer_fold":fold,"selected_C":c,"selected_gamma":gamma,"selected_pca_dim":dim,"threshold":t}); result.to_csv(foldout/"test_predictions.csv",index=False,encoding="utf-8-sig"); predictions.append(result)
        summaries.append({"outer_fold":fold,"selected_C":c,"selected_gamma":gamma,"selected_pca_dim":dim,"inner_auc":best[0],**{"test_"+k:v for k,v in metric(te.label.to_numpy(),p,t).items()}})
    summary=pd.DataFrame(summaries); summary.to_csv(out/"outer_fold_summary.csv",index=False); allp=pd.concat(predictions,ignore_index=True); allp.to_csv(out/"all_outer_test_predictions.csv",index=False)
    manifest={"method":"b1_plus_pca_local","outer_folds":[1,2,3,4,5],"inner_folds":4,"n_repeats":1,"animal_denominator":int(allp.animal_id.nunique()),"selection":"inner OOF AUC; threshold inner balanced accuracy","local_features":"first-frame Frangi-weighted local mean+max, PCA fitted inside each inner fold","global_features":"exact B1 first-frame global CLS","no_outer_test_used_for_selection":True,"mean_outer_auc":float(summary.test_auc.mean()),"mean_outer_accuracy":float(summary.test_accuracy.mean()),"mean_outer_balanced_accuracy":float(summary.test_balanced_accuracy.mean())}; (out/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
