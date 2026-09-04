"""Write the annotation back onto the data, as objects you can open directly.

The per-cell CSV is the record, but nobody wants to re-join a CSV by hand before plotting. These are
the same numbers attached to the counts:

    <name>_annotated.h5ad   scanpy / anndata
    <name>_annotated.rds    Seurat  (written only if asked, and only if R can do it)
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# Everything NACRE produces per cell. Absent columns are skipped, so a run without ground truth
# simply carries fewer.
CARRY = ["predicted_subtype", "predicted_lineage", "predicted_compartment",
         "usable_label", "resolution", "resolution_strict",
         "conc_subtype", "conc_lineage", "conc_compartment",
         "usable_at_subtype", "usable_at_lineage", "usable_at_compartment",
         "lock", "refined", "Seurat", "Azimuth", "RCTD", "SingleR",
         "truth", "truth_lineage", "truth_compartment", "acc_lineage", "acc_compartment"]
CATEGORICAL = {"predicted_subtype", "predicted_lineage", "predicted_compartment",
               "usable_label", "resolution", "resolution_strict",
               "Seurat", "Azimuth", "RCTD", "SingleR", "truth"}


def annotated_h5ad(d, q_h5, out_path, name):
    """Query counts + every NACRE column, aligned on cell id."""
    import anndata as ad
    a = ad.read_h5ad(q_h5)
    idx = pd.Index(a.obs_names.astype(str))
    t = d.set_index(d.cell.astype(str))
    t = t[~t.index.duplicated(keep="first")].reindex(idx)

    # Cells below the query count floor are dropped before the annotators run, so they have no
    # labels and reindexing leaves NaN. That turns a bool column into mixed bool/float, which anndata
    # refuses to write. Coerce by the values actually present rather than by a fixed list of names,
    # so a column added later cannot reintroduce the same failure.
    missing = int(t[CARRY[0]].isna().sum()) if CARRY[0] in t.columns else 0

    def coerce(v):
        if pd.api.types.is_bool_dtype(v) or set(map(type, v.dropna().head(200))) == {bool}:
            return v.map({True: "True", False: "False"}).fillna("NA").astype("category"), "bool"
        if pd.api.types.is_numeric_dtype(v):
            return v.astype(float), "numeric"
        return v.astype(str).replace({"nan": "NA", "None": "NA"}).fillna("NA").astype("category"), "text"

    added = []
    for c in CARRY:
        if c not in t.columns:
            continue
        v, _ = coerce(t[c])
        a.obs[c if c.startswith(("predicted_", "conc_", "usable")) else f"nacre_{c}"] = v.values
        added.append(c)

    a.obs["nacre_annotated"] = pd.Categorical(
        np.where(t[CARRY[0]].isna().values, "no", "yes") if CARRY[0] in t.columns else "yes")
    a.uns["nacre_mlh"] = {
        "run": str(name), "columns": [str(c) for c in added],
        "cells_annotated": int(len(t) - missing), "cells_below_count_floor": int(missing),
        "note": "resolution names the deepest level the certificate endorses; usable_label is the "
                "label at that depth; cells with nacre_annotated == 'no' fell below the count "
                "floor and were never annotated"}
    if missing:
        print(f"    note: {missing:,} cell(s) fell below the count floor and carry no label; "
              f"marked nacre_annotated = 'no'", flush=True)
    a.write_h5ad(out_path)
    return out_path, added


def annotated_rds(h5ad_path, out_path, rscript):
    """Convert to a Seurat object. Returns (path, None) or (None, reason)."""
    import subprocess, tempfile
    script = f'''
suppressMessages({{library(zellkonverter); library(Seurat); library(SingleCellExperiment)}})
sce <- readH5AD("{h5ad_path.replace(os.sep, '/')}")
m <- assay(sce, "X")
so <- CreateSeuratObject(counts = m, meta.data = as.data.frame(colData(sce)))
# spatial coordinates travel as a plain reduction: it survives saveRDS without needing the image
# machinery a real Seurat FOV object would demand
rd <- reducedDims(sce)
if ("spatial" %in% names(rd)) {{
  xy <- as.matrix(rd[["spatial"]])[, 1:2, drop = FALSE]
  colnames(xy) <- c("spatial_1", "spatial_2"); rownames(xy) <- colnames(so)
  so[["spatial"]] <- CreateDimReducObject(embeddings = xy, key = "spatial_",
                                          assay = DefaultAssay(so))
}}
saveRDS(so, "{out_path.replace(os.sep, '/')}")
cat("OK\\n")
'''
    fh = tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8")
    fh.write(script); fh.close()
    try:
        r = subprocess.run([rscript, "--vanilla", fh.name], capture_output=True, text=True,
                           timeout=7200)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        try: os.unlink(fh.name)
        except OSError: pass
    if os.path.exists(out_path):
        return out_path, None
    tail = (r.stderr or r.stdout or "").strip().splitlines()
    return None, (tail[-1] if tail else "R produced no object")
