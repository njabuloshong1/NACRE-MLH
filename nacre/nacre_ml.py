"""
NACRE-ML : NACRE with the rare-cell rescue replaced by a LEARNED refinement.
  UNANIMOUS(4/4)->lock, STRONG(3/1)->plurality  (identical to NACRE-4, do-no-harm),
  SPLIT(2/2,2/1/1,1/1/1/1) -> an MLP trained on the confident (lock+strong) cells decides the label,
     restricted to the candidate labels the 4 tools proposed for that cell.
Improvements over the prototype MLP:
  * features = expression SVD(50) + scaled spatial + spatial-neighborhood cell-type COMPOSITION
  * class-weighted cross-entropy (inverse-sqrt freq) => rare-aware, replaces the hand rare-lock rule
  * 2-hidden-layer net + BatchNorm + dropout, early-stopping on a held-out slice of confident cells (GPU)
"""
import os, sys, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sps
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
TOOLS=["Seurat","Azimuth","RCTD","SingleR"]; TIEBREAK=["Azimuth","RCTD","SingleR","Seurat"]

def _embed(adata,n_comp=50):
    X=adata.X
    if sps.issparse(X):
        X=X.tocsr().astype(np.float32); s=np.asarray(X.sum(1)).ravel(); s[s==0]=1
        X=X.multiply(1.0/s[:,None]).tocsr(); X=X.multiply(1e4); X.data=np.log1p(X.data)
        return TruncatedSVD(n_components=int(min(n_comp,X.shape[1]-1)),random_state=0).fit_transform(X)
    X=np.asarray(X,np.float32); s=X.sum(1,keepdims=True); s[s==0]=1; X=np.log1p(X/s*1e4)
    return TruncatedSVD(n_components=int(min(n_comp,X.shape[1]-1,X.shape[0]-1)),random_state=0).fit_transform(X)

def run_nacre_ml(name, query_h5ad, base_df, rare_set=None, use_spatial=True, k_neighbors=15,
                 n_pcs=50, spatial_weight=0.3, restrict_candidates=True, epochs=200, seed=0, verbose=True,
                 use_composition=True, class_weighted=True, return_proba=False):
    # use_composition / class_weighted exist for the ablations (see ablations.py); the defaults
    # reproduce the shipped method exactly. return_proba additionally returns the softmax over
    # candidates for split cells, used for the distribution-shift and confidence analyses.
    # NOTE: a REFERENCE-rare class-weight boost was tested (rare_boost) and REVERTED -- it lowered
    # concordance on the reference-rare split cells in 3/4 tissues (Breast -7.6, COAD/HCC small; only OV +4.6).
    import torch, torch.nn as nn, torch.nn.functional as Fn
    torch.manual_seed(seed); np.random.seed(seed)
    a=ad.read_h5ad(query_h5ad)
    if "high_quality" in a.obs: a=a[a.obs["high_quality"].astype(int)==1].copy()
    a.obs_names=a.obs_names.astype(str)
    df=base_df.set_index("cell").reindex(a.obs_names)
    ok=df[TOOLS].notna().all(1).values; a=a[ok].copy(); df=df.loc[ok]
    V=df[TOOLS].values.astype(str); n=len(V)
    nun=np.array([len(set(r)) for r in V]); plu=np.empty(n,object); maxcnt=np.zeros(n,int)
    for i in range(n):
        vals,cnt=np.unique(V[i],return_counts=True); mx=cnt.max(); maxcnt[i]=mx; win=set(vals[cnt==mx])
        if len(win)==1: plu[i]=next(iter(win))
        else:
            for t in TIEBREAK:
                if V[i,TOOLS.index(t)] in win: plu[i]=V[i,TOOLS.index(t)]; break
    lock=(nun==1); strong=(maxcnt==3); split=~lock&~strong
    final=plu.copy(); refined=np.zeros(n,bool)
    pmax=np.full(n,np.nan,np.float32); pent=np.full(n,np.nan,np.float32)   # filled only for refined cells
    if verbose: print(f"[NACRE-ML {name}] n={n} lock={lock.mean()*100:.1f}% strong={strong.mean()*100:.1f}% split={split.mean()*100:.1f}%",flush=True)
    if split.sum() and (lock|strong).sum()>=200:
        # ---- features: SVD(expr) + spatial + neighborhood composition ----
        emb=StandardScaler().fit_transform(_embed(a,n_pcs))
        feats=[emb]
        if use_spatial and "spatial" in a.obsm:
            feats.append(StandardScaler().fit_transform(np.asarray(a.obsm["spatial"],np.float32))*spatial_weight)
        Z=np.hstack(feats).astype(np.float32)
        classes=sorted(set(plu.astype(str)) | set(V.ravel())); ci={c:j for j,c in enumerate(classes)}; C=len(classes)
        conf=lock|strong; y=np.array([ci[str(x)] for x in plu])
        # neighborhood composition over confident labels
        if use_composition:
            nn_=NearestNeighbors(n_neighbors=k_neighbors+1,n_jobs=-1).fit(Z); idx=nn_.kneighbors(Z)[1][:,1:]
            Yc=np.zeros((n,C),np.float32); Yc[conf,y[conf]]=1.0
            rows=np.repeat(np.arange(n),k_neighbors); cols=idx.ravel()
            A=sps.coo_matrix((np.full(len(rows),1.0/k_neighbors,np.float32),(rows,cols)),shape=(n,n)).tocsr()
            comp=(A@Yc).astype(np.float32)
            F=np.hstack([Z,comp]).astype(np.float32)
        else:
            F=Z.astype(np.float32)
        # ---- torch MLP, class-weighted, early stop ----
        dev="cuda" if torch.cuda.is_available() else "cpu"
        Xt=torch.tensor(F,device=dev); yt=torch.tensor(y,device=dev)
        cidx=np.where(conf)[0]; np.random.shuffle(cidx); nv=int(0.1*len(cidx))
        vi=torch.tensor(cidx[:nv],device=dev); tri=torch.tensor(cidx[nv:],device=dev)
        cw=np.bincount(y[conf],minlength=C).astype(np.float32); cw=(cw.sum()/(cw+1.0))**0.5; cw=cw/cw.mean()
        if not class_weighted: cw=np.ones(C,np.float32)      # ablation: unweighted cross-entropy
        cwt=torch.tensor(cw,device=dev)
        class Net(nn.Module):
            def __init__(s,d,C):
                super().__init__()
                s.net=nn.Sequential(nn.Linear(d,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.3),
                                    nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,C))
            def forward(s,x): return s.net(x)
        net=Net(F.shape[1],C).to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
        best=1e9; bad=0; beststate=None
        for ep in range(epochs):
            net.train(); opt.zero_grad(); out=net(Xt[tri]); loss=Fn.cross_entropy(out,yt[tri],weight=cwt)
            loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): vl=Fn.cross_entropy(net(Xt[vi]),yt[vi],weight=cwt).item()
            if vl<best-1e-4: best=vl; bad=0; beststate={k:v.clone() for k,v in net.state_dict().items()}
            else:
                bad+=1
                if bad>=20: break
        if beststate: net.load_state_dict(beststate)
        net.eval()
        sp_idx=np.where(split)[0]
        with torch.no_grad():
            proba=torch.softmax(net(Xt[torch.tensor(sp_idx,device=dev)]),1).cpu().numpy()
        for k2,i in enumerate(sp_idx):
            cands=list(dict.fromkeys(V[i])) if restrict_candidates else classes
            cj=[ci[c] for c in cands if c in ci]
            if not cj: continue
            pv=np.array([proba[k2,j] for j in cj],np.float64); pv=pv/max(pv.sum(),1e-12)
            best_c=cands[int(np.argmax([proba[k2,j] for j in cj]))]
            final[i]=best_c; refined[i]=True
            pmax[i]=pv.max(); pent[i]=float(-(pv*np.log(pv+1e-12)).sum())
    res=pd.DataFrame({"cell":a.obs_names,"nacre_ml":final.astype(str),"lock":lock,"refined":refined,
                      "gt":df["gt"].astype(str).values if "gt" in df else np.nan})
    for t in TOOLS: res[t]=V[:,TOOLS.index(t)]
    if return_proba:
        res["mlp_pmax"]=pmax; res["mlp_entropy"]=pent
    return res
