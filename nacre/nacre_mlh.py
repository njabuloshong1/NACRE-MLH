"""
NACRE-MLH : NACRE-ML + multi-resolution output.  (NACRE-ML in nacre_ml.py is UNCHANGED.)
Runs NACRE-ML to get the subtype label, then deterministically maps it up a fixed hierarchy
(assets/hierarchy.csv, extended per reference as needed) to produce:
    predicted_subtype     (= NACRE-ML, finest)
    predicted_lineage     (parent)
    predicted_compartment (grandparent; Myeloid / Lymphoid / Stromal / Epithelial)
plus a per-cell RESOLUTION CERTIFICATE -- how deep THIS cell's label can be trusted:
    conc_<level>          raw per-cell panel concordance (25/50/75/100), no threshold
    resolution            finest level with >=3/4 tool backing, else 'unresolved'
    usable_label          the label at that level ('Unresolved' if none) -- the column to consume
    usable                resolution != 'unresolved'
    resolution_strict     same cascade at 4/4 (conservative bound)
See multilevel_resolution.py for the batch/analysis version and the independent validation.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nacre_ml as NML
HIER=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"assets","hierarchy.csv")
LEVELS=["subtype","lineage","compartment"]
THRESH=0.75   # NACRE's own confident line: lock(4/4) or strong(3/1); see nacre_ml.py

def load_hierarchy(path=HIER):
    H=pd.read_csv(path)
    return dict(zip(H.label.astype(str),H.lineage.astype(str))), dict(zip(H.label.astype(str),H.compartment.astype(str)))

def check_hierarchy(labels, path=HIER):
    """Labels absent from the hierarchy fall back to THEMSELVES (never a shared '?' -- two distinct
    unmapped labels must not compare as equal, or they'd score as concordant). Warn so a new dataset
    gets its hierarchy extended (see ensure_hierarchy in nacre_mlh_cp.py) instead of silently
    coarsening to nothing."""
    lin,_=load_hierarchy(path); miss=sorted({str(x) for x in labels}-set(lin))
    if miss:
        print(f"[NACRE-MLH] WARNING: {len(miss)} label(s) not in {os.path.basename(path)}: {', '.join(miss[:12])}"
              f"{' ...' if len(miss)>12 else ''}\n             they stay unchanged at lineage/compartment "
              f"(no coarsening, so they can never gain concordance by backing off).\n"
              f"             Supply an LLM key so the hierarchy is extended for this dataset.",flush=True)
    return miss

def add_resolution(res, tools=NML.TOOLS, thresh=THRESH, hierarchy_path=None):
    """Per-cell resolution certificate from the already-present tool columns + the 3 labels.

    hierarchy_path must be threaded through, not defaulted. A default argument binds once at
    definition time, so `load_hierarchy()` here would silently read the packaged asset even when the
    caller extended the hierarchy for this dataset: the emitted labels would be correct while every
    concordance was computed against unmapped names, making c_lineage collapse instead of rise.
    """
    lab={lev:res[f"predicted_{lev}"].astype(str).values for lev in LEVELS}
    T=res[tools].astype(str).values
    lin,comp=load_hierarchy(hierarchy_path or HIER)
    # otypes is required or np.vectorize refuses a size-0 input, turning an empty frame into an
    # opaque ValueError several frames from the actual cause.
    up={"subtype":lambda a:a,
        "lineage":np.vectorize(lambda x:lin.get(x,x),otypes=[object]),
        "compartment":np.vectorize(lambda x:comp.get(x,x),otypes=[object])}
    conc={lev:(up[lev](T)==lab[lev][:,None]).mean(1) for lev in LEVELS}
    for lev in LEVELS: res[f"conc_{lev}"]=(100*conc[lev]).round(1)
    def tier(t):
        return np.where(conc["subtype"]>=t,"subtype",
               np.where(conc["lineage"]>=t,"lineage",
               np.where(conc["compartment"]>=t,"compartment","unresolved")))
    res["resolution"]=tier(thresh)
    res["usable_label"]=[lab[r][i] if r!="unresolved" else "Unresolved"
                         for i,r in enumerate(res["resolution"].values)]
    res["usable"]=(res["resolution"].values!="unresolved").astype(np.int8)
    # per-level filters, 1=usable / 0=not. A user working at ONE level filters on that level's
    # flag -- failing at subtype says nothing about whether the lineage call is sound.
    for lev in LEVELS: res[f"usable_at_{lev}"]=(conc[lev]>=thresh).astype(np.int8)
    res["resolution_strict"]=tier(1.0)
    return res

def run_nacre_mlh(name, query_h5ad, base_df, rare_set=None, hierarchy_path=None, **kw):
    hierarchy_path = hierarchy_path or HIER
    res=NML.run_nacre_ml(name, query_h5ad, base_df, rare_set=rare_set, **kw)   # has 'nacre_ml'
    lin,comp=load_hierarchy(hierarchy_path)
    res=res.rename(columns={"nacre_ml":"predicted_subtype"})
    # identity fallback, NOT "?" -- must match the mapping add_resolution() applies to the tool labels
    check_hierarchy(set(res["predicted_subtype"].astype(str))|set(res[NML.TOOLS].astype(str).values.ravel()),hierarchy_path)
    res["predicted_lineage"]=res["predicted_subtype"].map(lambda x: lin.get(str(x),str(x)))
    res["predicted_compartment"]=res["predicted_subtype"].map(lambda x: comp.get(str(x),str(x)))
    return add_resolution(res, hierarchy_path=hierarchy_path)

# The original hard-wired __main__ block is removed here; this copy is driven by
# nacre_mlh_cp.py, which supplies platform, paths and thresholds explicitly.
