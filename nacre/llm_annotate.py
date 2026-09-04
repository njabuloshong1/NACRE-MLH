"""
LLM as a BASE annotator for the whole tissue (GPTCelltype-style):
cluster the query (Leiden) -> per-cluster top markers -> LLM assigns ONE cell type from the reference
vocabulary -> propagate to all cells. Produces <tissue>_llm.csv (cell, LLM) for every tissue, so the
LLM can be used like Seurat/Azimuth/RCTD/SingleR in leave-one-out consensus experiments.
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd, anndata as ad, scanpy as sc

def _load_key():
    """Self-contained key lookup: argument -> environment -> a key file beside this package."""
    import os
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in (".env", "openai_key.txt", "api_key.txt"):
        p = os.path.join(os.path.dirname(here), fn)
        if os.path.exists(p):
            for line in open(p, encoding="utf8").read().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip("\"'")
                if line.startswith("sk-"):
                    return line
    return None


class _KeyShim:
    _load_key = staticmethod(_load_key)


nacre5 = _KeyShim()


def llm_label_cluster(tissue, valid_types, markers, model="gpt-4o"):
    key=nacre5._load_key(); os.environ["OPENAI_API_KEY"]=key
    from openai import OpenAI; client=OpenAI()
    sysp=("You are an expert annotating single cells in spatial transcriptomics. Assign the cluster to "
          "EXACTLY ONE cell type from the provided list (use the list's spelling verbatim). Base the call "
          "on the ranked marker genes (highest first). Return JSON {\"label\":<one of the list>,\"reason\":<short>}.")
    usr=json.dumps({"tissue":tissue,"allowed_cell_types":valid_types,"cluster_top_markers_high_to_low":markers})
    kw=dict(model=model,response_format={"type":"json_object"},
            messages=[{"role":"system","content":sysp},{"role":"user","content":usr}])
    if not str(model).startswith("gpt-5"): kw["temperature"]=0   # gpt-5 rejects temperature=0 (forces default)
    r=client.chat.completions.create(**kw)
    d=json.loads(r.choices[0].message.content); return d.get("label")

def run(name, query_h5ad, nacre4_csv, tissue, resolution=1.0, topn=20, outdir="results", model="gpt-4o", suffix=""):
    os.makedirs(outdir,exist_ok=True)
    a=ad.read_h5ad(query_h5ad)
    if "high_quality" in a.obs: a=a[a.obs["high_quality"].astype(int)==1].copy()
    a.obs_names=a.obs_names.astype(str)
    # valid cell-type vocabulary = union of the 4 tools' labels
    nb=pd.read_csv(nacre4_csv); nb["cell"]=nb["cell"].astype(str)
    vocab=sorted(set(pd.unique(nb[["Seurat","Azimuth","RCTD","SingleR"]].astype(str).values.ravel()))-{"nan"})
    sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a,n_top_genes=min(2000,a.n_vars))
    sc.pp.pca(a,n_comps=50,use_highly_variable=True)
    sc.pp.neighbors(a,n_neighbors=15)
    sc.tl.leiden(a,resolution=resolution,flavor="igraph",n_iterations=2,directed=False)
    ncl=a.obs["leiden"].nunique()
    sc.tl.rank_genes_groups(a,"leiden",method="wilcoxon",n_genes=topn)
    print(f"[{name}] {a.n_obs} cells -> {ncl} leiden clusters | vocab={len(vocab)} types",flush=True)
    cl2lab={}
    for cl in a.obs["leiden"].cat.categories:
        mk=[g for g in a.uns["rank_genes_groups"]["names"][cl][:topn]]
        lab=None
        for attempt in range(5):
            try: lab=llm_label_cluster(tissue,vocab,mk,model=model); break
            except Exception as e:
                import time; print(f"   cluster {cl} LLM error (try {attempt+1}): {str(e)[:90]}",flush=True); time.sleep(12)
        if lab not in vocab:      # snap to vocab if the LLM drifted
            lab=next((v for v in vocab if v.lower()==str(lab).lower()), vocab[0] if lab is None else lab)
        cl2lab[cl]=lab
        print(f"   cluster {cl:>3} (n={int((a.obs['leiden']==cl).sum()):6d}): {mk[:6]} -> {lab}",flush=True)
    a.obs["LLM"]=a.obs["leiden"].map(cl2lab).astype(str)
    out=pd.DataFrame({"cell":a.obs_names,"LLM":a.obs["LLM"].values,"leiden":a.obs["leiden"].values})
    fn=f"{name}_llm{suffix}.csv"; out.to_csv(os.path.join(outdir,fn),index=False)
    print(f"[{name}] saved {fn} ({ncl} clusters labelled, model={model})",flush=True)

# demo __main__ removed: this copy is driven by nacre_mlh_cp.py.
