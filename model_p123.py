"""Strict P1/P2/P3 graph-temporal model from 模型算法任务书.md."""
from __future__ import annotations
import math
import torch
from torch import nn


def _sinusoidal(x: torch.Tensor, dim: int) -> torch.Tensor:
    half=max(1,dim//2)
    freq=torch.exp(torch.arange(half,device=x.device,dtype=x.dtype)*(-math.log(10000.0)/max(half-1,1)))
    z=x.unsqueeze(-1)*freq
    out=torch.cat((torch.sin(z),torch.cos(z)),-1)
    if out.shape[-1]<dim: out=torch.nn.functional.pad(out,(0,dim-out.shape[-1]))
    return out[...,:dim]


class GATv2Layer(nn.Module):
    def __init__(self,dim:int,heads:int,dropout:float):
        super().__init__()
        if dim%heads: raise ValueError("hidden_dim must be divisible by attention_heads")
        self.heads,self.head_dim=heads,dim//heads
        self.src=nn.Linear(dim,dim,bias=False); self.dst=nn.Linear(dim,dim,bias=False)
        self.attn=nn.Parameter(torch.empty(heads,2*self.head_dim)); nn.init.xavier_uniform_(self.attn)
        self.out=nn.Linear(dim,dim); self.norm=nn.LayerNorm(dim); self.drop=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor,adjacency:torch.Tensor)->torch.Tensor:
        b,n,_=x.shape
        q=self.src(x).view(b,n,self.heads,self.head_dim)
        k=self.dst(x).view(b,n,self.heads,self.head_dim)
        pair=torch.cat((q.unsqueeze(2).expand(-1,-1,n,-1,-1),k.unsqueeze(1).expand(-1,n,-1,-1,-1)),-1)
        score=torch.nn.functional.leaky_relu((pair*self.attn.view(1,1,1,self.heads,-1)).sum(-1),.2).permute(0,1,3,2)
        score=score.masked_fill(~adjacency.bool().view(1,n,1,n),torch.finfo(score.dtype).min)
        alpha=torch.softmax(score,-1)
        y=torch.einsum("bihj,bjhd->bihd",alpha,k).reshape(b,n,-1)
        return self.norm(x+self.drop(self.out(y)))


class StrictP123Model(nn.Module):
    def __init__(self,feature_dim:int,hidden_dim=128,heads=4,temporal_layers=2,dropout=.1,no_gat=False,horizons=(.5,1.,2.)):
        super().__init__(); self.feature_dim=feature_dim; self.hidden_dim=hidden_dim; self.no_gat=no_gat
        self.horizons=tuple(float(x) for x in horizons)
        self.region_in=nn.Linear(feature_dim,hidden_dim)
        self.gat=nn.ModuleList([GATv2Layer(hidden_dim,heads,dropout),GATv2Layer(hidden_dim,heads,dropout)])
        layer=nn.TransformerEncoderLayer(hidden_dim,heads,hidden_dim*2,dropout,batch_first=True,norm_first=True,activation="gelu")
        self.temporal=nn.TransformerEncoder(layer,num_layers=temporal_layers)
        self.time_proj=nn.Linear(hidden_dim,hidden_dim); self.global_proj=nn.Linear(feature_dim,hidden_dim)
        self.classifier=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1))
        self.future=nn.Sequential(nn.Linear(hidden_dim+1,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,feature_dim))

    @staticmethod
    def adjacency(n:int,device):
        side=int(round(n**.5))
        if side*side!=n:return torch.eye(n,device=device,dtype=torch.bool)
        a=torch.zeros(n,n,device=device,dtype=torch.bool)
        for i in range(n):
            r,c=divmod(i,side)
            for j in range(n):
                rr,cc=divmod(j,side)
                if abs(r-rr)+abs(c-cc)<=1:a[i,j]=True
        return a

    def encode(self,regions,global_features,timestamps,valid,region_valid=None):
        b,t,n,_=regions.shape
        if region_valid is None:region_valid=valid[:,:,None].expand(-1,-1,n)
        x=self.region_in(regions); adj=self.adjacency(n,regions.device); spatial=[]
        for i in range(t):
            z=x[:,i]
            if not self.no_gat:
                for layer in self.gat:z=layer(z,adj)
            spatial.append(z)
        spatial=torch.stack(spatial,1)
        clean=timestamps.masked_fill(~valid,0.0)
        first=clean.gather(1,valid.float().argmax(1).view(-1,1)).squeeze(1)
        rel=(clean-first[:,None]).clamp_min(0); dt=torch.zeros_like(rel); dt[:,1:]=(rel[:,1:]-rel[:,:-1]).clamp_min(0)
        time=torch.cat((_sinusoidal(rel,self.hidden_dim//2),_sinusoidal(dt,self.hidden_dim-self.hidden_dim//2)),-1)
        z=spatial+self.time_proj(time).unsqueeze(2)
        # Each fixed region has its own causal time sequence.
        flat=z.permute(0,2,1,3).reshape(b*n,t,self.hidden_dim)
        pad=(~valid).unsqueeze(1).expand(-1,n,-1).reshape(b*n,t)
        causal=torch.triu(torch.ones(t,t,device=regions.device,dtype=torch.bool),1)
        encoded=self.temporal(flat,mask=causal,src_key_padding_mask=pad)
        return encoded.reshape(b,n,t,self.hidden_dim).permute(0,2,1,3),region_valid

    def forward(self,regions,global_features,timestamps,valid,region_valid=None,return_state=False):
        state,region_valid=self.encode(regions,global_features,timestamps,valid,region_valid)
        w=region_valid.to(state.dtype)*valid[:,:,None].to(state.dtype)
        frame=(state*w[...,None]).sum(2)/w.sum(2,keepdim=True).clamp_min(1)
        tw=valid.to(state.dtype)
        pooled=(frame*tw[...,None]).sum(1)/tw.sum(1,keepdim=True).clamp_min(1)
        gf=(global_features*tw[...,None]).sum(1)/tw.sum(1,keepdim=True).clamp_min(1)
        logits=self.classifier(torch.cat((pooled,self.global_proj(gf)),-1)).squeeze(-1)
        return {"logits":logits,"state":state,"region_valid":region_valid}

    def predict_delta(self,state,horizon:float):
        h=torch.full((*state.shape[:-1],1),float(horizon),device=state.device,dtype=state.dtype)
        return self.future(torch.cat((state,h),-1))
