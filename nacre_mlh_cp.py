"""NACRE-MLH_cp: a command-line build of NACRE-MLH that runs on Xenium, MERSCOPE or CosMx output.

    python nacre_mlh_cp.py --platform X --query DIR --ref DIR --out DIR [--llm-key KEY]

Stages, all resumable, each skipped when its output already exists:

  1. build     vendor directory        -> query.h5ad (counts, coordinates, QC flag)
     build     reference directory     -> reference.h5ad harmonized to the label vocabulary
  2. annotate  four reference-based tools (Seurat, Azimuth, RCTD, SingleR) -> bases.csv
  3. resolve   consensus, learned resolver on contested cells, hierarchy    -> nacre_mlh.csv

The platform code selects the loader and the filter thresholds; see nacre/platforms.py for why
min_umi in particular must be set per platform rather than left at RCTD's default.

The reference directory must contain one .h5ad with a cell-type column. Pass --ref-label to name it,
otherwise the first of major_annotation / cell_type / celltype / Final_CellTypes / annotation is used.

--llm-key is needed only when the reference vocabulary contains labels absent from
assets/hierarchy.csv, in which case the missing labels are mapped to lineage and compartment once and
cached. Supply it as an argument or via OPENAI_API_KEY. With --llm-panel it also runs the held-out
cluster-level LLM annotation used for validation.
"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "nacre"))
from nacre import platforms  # noqa: E402

# PATH first, so a container or a Linux install is found without special-casing. The Windows globs
# are a fallback for machines where R is installed but not on PATH; they simply never match
# elsewhere. No entry may name a specific user or a single R version.
RSCRIPT_CANDIDATES = [
    "Rscript",
    r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
    r"C:\Program Files\R\R-*\bin\Rscript.exe",
]
LABEL_COLS = ("major_annotation", "cell_type", "celltype", "Final_CellTypes", "annotation",
              "cell_types", "CellType")
MIN_CELLS_PER_TYPE = 25
PER_TYPE = 3000
SEED = 0


def find_rscript(user):
    import shutil, glob
    if user and os.path.exists(user):
        return user
    # Several R versions are commonly installed and only one carries Seurat/spacexr/SingleR. The
    # pipeline layer picks by probing for those packages; reuse it when present so running this
    # script directly does not silently choose an R that cannot run the annotators.
    try:
        sys.path.insert(0, HERE)
        from run_pipeline import find_rscript as smart
        r = smart(None, verbose=False)
        if r:
            return r
    except Exception:
        pass
    for c in RSCRIPT_CANDIDATES:
        if "*" in c:
            hits = sorted(glob.glob(c), reverse=True)
            if hits:
                return hits[0]
        elif shutil.which(c):
            return shutil.which(c)
        elif os.path.exists(c):
            return c
    raise SystemExit("Rscript not found; install R or pass --rscript")


def counts_matrix(a):
    """Integer counts. Many deposited h5ads normalize X and keep counts in a layer or .raw; feeding
    normalized values to RCTD fails silently rather than loudly."""
    import scipy.sparse as sps
    for src, X in (("layers['counts']", a.layers.get("counts") if hasattr(a, "layers") else None),
                   ("X", a.X),
                   (".raw", a.raw.X if getattr(a, "raw", None) is not None else None)):
        if X is None:
            continue
        v = X[:200]
        v = v.toarray() if sps.issparse(v) else np.asarray(v)
        if v.size and np.allclose(v, np.round(v)):
            return X, src
    raise SystemExit("no integer counts found in layers['counts'], X or .raw")



def find_h5ads(d):
    """Recursive, largest-first: users pass a folder, and the .h5ad often sits one level inside."""
    import glob
    if os.path.isfile(d) and d.endswith(".h5ad"):
        return [d]
    hits = glob.glob(os.path.join(d, "**", "*.h5ad"), recursive=True)
    return sorted(hits, key=lambda p: -os.path.getsize(p))



SYMBOL_COLS = ("feature_name", "gene_name", "gene_symbols", "gene_symbol", "Symbol", "symbol",
               "gene_ids", "GeneSymbol")


def to_symbols(a):
    """CZ CELLxGENE and many GEO deposits index var by Ensembl ID while spatial panels use symbols,
    which yields an empty intersection and four failed annotators. Switch to the symbol column when
    the index is clearly Ensembl."""
    names = pd.Index(a.var_names.astype(str))
    if not names.str.startswith("ENSG").mean() > 0.5:
        return a, None
    col = next((c for c in SYMBOL_COLS if c in a.var.columns), None)
    if col is None:
        return a, None
    a = a.copy()
    a.var_names = a.var[col].astype(str).to_numpy()
    a = a[:, ~a.var_names.duplicated()].copy()
    return a, col


class _LazyH5AD:
    """Minimal stand-in for a backed AnnData: obs, var and row-indexable X, nothing else.

    Only what build_reference touches is provided, so the file is never read whole.
    """

    def __init__(self, path):
        import h5py
        from anndata.io import read_elem, sparse_dataset
        self._f = h5py.File(path, "r")
        self.obs = read_elem(self._f["obs"])
        self.var = read_elem(self._f["var"])
        def wrap(node):
            return sparse_dataset(node) if isinstance(node, h5py.Group) else node

        self.X = wrap(self._f["X"])
        # X is frequently normalized while the integer counts sit in layers['counts']; carry the
        # layers through, or counts_matrix() below has nothing to find and RCTD gets log values.
        self.layers = {k: wrap(self._f["layers"][k]) for k in self._f.get("layers", {})}
        self.raw = None
        self.n_obs, self.n_vars = len(self.obs), len(self.var)
        self.var_names = pd.Index(self.var.index.astype(str))
        self.obs_names = pd.Index(self.obs.index.astype(str))

    def __getitem__(self, idx):
        """Row subset, materialized. Callers subsample first, so this stays small."""
        import anndata as ad
        rows = np.asarray(idx)
        a = ad.AnnData(X=self.X[rows], obs=self.obs.iloc[rows].copy(), var=self.var.copy())
        a.obs_names = self.obs_names[rows]
        a.var_names = self.var_names
        for k, v in self.layers.items():
            a.layers[k] = v[rows]
        return a

    def to_memory(self):
        return self

    class _File:
        def __init__(self, h):
            self._h = h

        def close(self):
            try:
                self._h.close()
            except Exception:
                pass

    @property
    def file(self):
        return _LazyH5AD._File(self._f)


def read_h5ad_safe(path, backed=None):
    """Read an .h5ad whose /uns this anndata cannot decode.

    Published references often carry uns entries encoded as 'null': a scanpy log1p record whose
    `base` is None, a stored neighbor graph. anndata aborts the entire read on them even though the
    pipeline only ever needs X, obs and var. On that specific failure fall back to a lazy reader,
    which matters because references run to tens of gigabytes and must not be materialized whole.
    """
    import anndata as ad
    try:
        return ad.read_h5ad(path, backed=backed) if backed else ad.read_h5ad(path)
    except Exception as e:
        if "encoding_type='null'" not in str(e) and "No read method registered" not in str(e):
            raise
        print(f"  note: {os.path.basename(path)} has uns metadata this anndata cannot decode; "
              f"reading X/obs/var only", flush=True)
        lazy = _LazyH5AD(path)
        return lazy if backed else lazy[np.arange(lazy.n_obs)]


def build_query(plat, qdir, out, min_counts):
    import anndata as ad
    if os.path.exists(out):
        import anndata
        return list(anndata.read_h5ad(out, backed="r").var_names)
    h5ads = find_h5ads(qdir)
    if h5ads:
        a = read_h5ad_safe(h5ads[0])
        X, src = counts_matrix(a)
        genes = np.array(a.var_names, str)
        keep = plat.drop_controls(genes)
        X, genes = X[:, keep], genes[keep]
        cells = np.array(a.obs_names, str)
        if "spatial" not in a.obsm:
            raise SystemExit(f"{h5ads[0]} has no obsm['spatial']")
        xy = np.asarray(a.obsm["spatial"], np.float32)[:, :2]
        tx = (a.obs["transcript_counts"].to_numpy(float) if "transcript_counts" in a.obs
              else np.asarray(X.sum(1)).ravel())
        print(f"  query from {os.path.basename(h5ads[0])} (counts in {src})", flush=True)
    else:
        X, genes, cells, xy, tx = plat.load(qdir)
        print(f"  query from {plat.name} vendor files", flush=True)
    q = ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(cells)),
                   var=pd.DataFrame(index=pd.Index(genes)))
    q.obsm["spatial"] = xy
    q.obs["transcript_counts"] = tx
    q.obs["high_quality"] = (tx >= min_counts).astype(int)
    q.write_h5ad(out)
    print(f"  query: {q.n_obs:,} cells x {q.n_vars:,} genes | high_quality "
          f"{int(q.obs.high_quality.sum()):,} ({100*q.obs.high_quality.mean():.1f}%)", flush=True)
    return list(q.var_names)


def build_reference(rdir, out, panel, label_col, min_umi):
    """Read backed, choose the subsample from obs alone, and only then materialize.

    Order matters: atlas references run to tens of gigabytes, so any operation that touches X before
    subsampling (a copy, a symbol rename) will exhaust memory on the full matrix.
    """
    import anndata as ad
    if os.path.exists(out):
        return
    h5ads = find_h5ads(rdir)
    if not h5ads:
        raise SystemExit(f"no .h5ad found under {rdir} (searched recursively)")
    src_path = h5ads[0]
    big = read_h5ad_safe(src_path, backed="r")
    col = label_col or next((c for c in LABEL_COLS if c in big.obs), None)
    if col is None:
        raise SystemExit(f"no cell-type column in {os.path.basename(src_path)}; "
                         f"tried {LABEL_COLS}. Use --ref-label.")
    lab_all = big.obs[col].astype(str).to_numpy()
    # Cluster-annotated references often prefix the label with its cluster index ("18: SPP1+
    # Macrophages", "2: SPP1+ Macrophages"). Left alone, one population splits across several
    # vocabulary entries: the hierarchy has to be extended once per entry, and the panel records
    # disagreement whenever two tools pick different cluster indices for the same cell type.
    stripped = np.array([re.sub(r"^\s*\d+\s*:\s*", "", s) for s in lab_all])
    if (stripped != lab_all).any():
        n_before, n_after = len(set(lab_all)), len(set(stripped))
        print(f"  reference labels carry cluster-index prefixes; stripped, "
              f"{n_before} -> {n_after} distinct types", flush=True)
        lab_all = stripped

    rng = np.random.default_rng(SEED)
    keep = []
    for t in sorted(set(lab_all)):
        idx = np.flatnonzero(lab_all == t)
        keep.append(np.sort(idx if len(idx) <= PER_TYPE else rng.choice(idx, PER_TYPE, replace=False)))
    keep = np.sort(np.concatenate(keep))
    print(f"  reference {os.path.basename(src_path)}: {big.n_obs:,} cells, labels in '{col}'; "
          f"subsampling {len(keep):,} at up to {PER_TYPE:,}/type", flush=True)

    sub = big[keep].to_memory()
    try:
        big.file.close()
    except Exception:
        pass
    sub, symcol = to_symbols(sub)
    if symcol:
        print(f"  var switched from Ensembl IDs to symbols via '{symcol}' ({sub.n_vars:,} genes)",
              flush=True)
    X, csrc = counts_matrix(sub)
    print(f"  counts from {csrc}", flush=True)

    r = ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(np.array(sub.obs_names, str))),
                   var=pd.DataFrame(index=pd.Index(np.array(sub.var_names, str))))
    r = r[:, ~r.var_names.duplicated()].copy()
    r.obs["major_annotation"] = lab_all[keep]
    del sub

    shared = [g for g in panel if g in set(r.var_names)]
    if not shared:
        raise SystemExit("query and reference share no gene symbols. A reference indexed by Ensembl "
                         "ID does this; no usable symbol column was found in var.")
    tot = np.asarray(r[:, shared].X.sum(1)).ravel()
    r = r[tot >= min_umi].copy()
    vc = r.obs["major_annotation"].value_counts()
    drop = sorted(vc[vc < MIN_CELLS_PER_TYPE].index)
    if drop:
        r = r[~r.obs["major_annotation"].isin(drop)].copy()
    r.write_h5ad(out)
    vc = r.obs["major_annotation"].value_counts()
    print(f"  reference: {len(shared)}/{len(panel)} panel genes shared | {r.n_obs:,} cells, "
          f"{len(vc)} types | dropped below {MIN_CELLS_PER_TYPE} cells: {drop or 'none'}", flush=True)


def ensure_hierarchy(ref_h5ad, hier_path, llm_key):
    """Every emitted label needs a lineage and a compartment or it can never be coarsened."""
    import anndata as ad
    H = pd.read_csv(hier_path)
    known = set(H.label.astype(str))
    labs = set(ad.read_h5ad(ref_h5ad, backed="r").obs["major_annotation"].astype(str))
    missing = sorted(labs - known)
    if not missing:
        print(f"  hierarchy covers all {len(labs)} reference labels", flush=True)
        return hier_path
    print(f"  {len(missing)} label(s) absent from the hierarchy: {missing}", flush=True)
    key = llm_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("These labels have no lineage/compartment mapping, so they could never be "
                         "coarsened. Supply --llm-key to extend the hierarchy, or rename them to "
                         "existing labels.")
    os.environ["OPENAI_API_KEY"] = key
    from openai import OpenAI
    client = OpenAI()
    comps = sorted(H.compartment.unique())
    lins = sorted(H.lineage.unique())
    prompt = ("Map each cell type to an immediate biological lineage and one compartment. "
              "Reuse the existing values verbatim where they fit.\n"
              + json.dumps({"cell_types": missing, "existing_lineages": lins,
                            "compartments": comps})
              + '\nReturn JSON {"map":[{"label":..,"lineage":..,"compartment":..}]}')
    r = client.chat.completions.create(model="gpt-5", response_format={"type": "json_object"},
                                       messages=[{"role": "user", "content": prompt}])
    add = pd.DataFrame(json.loads(r.choices[0].message.content)["map"])
    out = os.path.join(os.path.dirname(ref_h5ad), "hierarchy_extended.csv")
    pd.concat([H, add[["label", "lineage", "compartment"]]], ignore_index=True).to_csv(out, index=False)
    print(f"  extended hierarchy -> {out}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description="NACRE-MLH_cp: consensus annotation with a resolution "
                                            "certificate, for Xenium / MERSCOPE / CosMx.")
    p.add_argument("--platform", required=True, help="X (Xenium), M (MERSCOPE) or C (CosMx)")
    p.add_argument("--query", required=True, help="directory of vendor output, or one .h5ad")
    p.add_argument("--ref", required=True, help="directory holding the single-cell reference .h5ad")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--name", default=None, help="run name (default: query directory name)")
    p.add_argument("--llm-key", default=None, help="LLM API key; needed only to extend the hierarchy")
    p.add_argument("--llm-panel", action="store_true", help="also run the held-out LLM panel")
    p.add_argument("--ref-label", default=None, help="cell-type column in the reference")
    p.add_argument("--min-umi", type=int, default=None, help="override the RCTD reference floor")
    p.add_argument("--min-counts", type=int, default=None, help="override the query cell floor")
    p.add_argument("--chunk", type=int, default=None,
                   help="query cells per annotator pass; omit to size it from free memory")
    p.add_argument("--rscript", default=None)
    a = p.parse_args()

    plat = platforms.get(a.platform)
    min_umi = a.min_umi if a.min_umi is not None else plat.min_umi
    min_counts = a.min_counts if a.min_counts is not None else plat.min_counts
    name = a.name or os.path.basename(os.path.normpath(a.query))
    os.makedirs(a.out, exist_ok=True)
    q_h5 = os.path.join(a.out, f"query_{name}.h5ad")
    r_h5 = os.path.join(a.out, f"ref_{name}.h5ad")
    bases = os.path.join(a.out, f"bases_{name}.csv")
    mlh = os.path.join(a.out, f"nacre_mlh_{name}.csv")

    print(f"\n=== NACRE-MLH_cp | {plat.name} | {name} ===")
    print(f"min_umi={min_umi} (RCTD reference floor)  min_counts={min_counts} (query cell floor)\n")

    print("[1/3] building inputs", flush=True)
    panel = build_query(plat, a.query, q_h5, min_counts)
    build_reference(a.ref, r_h5, panel, a.ref_label, min_umi)
    hier = ensure_hierarchy(r_h5, os.path.join(HERE, "assets", "hierarchy.csv"), a.llm_key)

    print("\n[2/3] four base annotators", flush=True)
    t0 = time.time()
    if os.path.exists(bases):
        print("  [skip] bases exist", flush=True)
    else:
        # Size the annotator pass from free memory and the panel width. The dense intermediates
        # Seurat, SCTransform and RCTD build scale with cells x shared genes, so a 5,000-plex panel
        # needs far smaller passes than a 1,000-plex one. A section that fits runs in a single pass,
        # which is identical to not chunking; only a section that would not fit gets split.
        chunk = a.chunk
        if chunk is None:
            try:
                sys.path.insert(0, HERE)
                from pipeline.resources import plan_chunk
                import anndata as _ad
                _q = _ad.read_h5ad(q_h5, backed="r")
                chunk, note = plan_chunk(int(_q.n_obs), int(_q.n_vars)); del _q
                print(f"  memory: {note}", flush=True)
            except Exception as e:
                chunk = 0
                print(f"  memory: could not size chunks ({e}); running in one pass", flush=True)

        log = os.path.join(a.out, f"log_{name}.txt")
        with open(log, "w", encoding="utf8") as fh:
            subprocess.run([find_rscript(a.rscript), os.path.join(HERE, "annotators",
                                                                  "run_four_bases.R"),
                            q_h5, r_h5, bases, os.path.join(a.out, f"runtime_{name}.csv"),
                            name, str(min_umi), str(int(chunk or 0))],
                           stdout=fh, stderr=subprocess.STDOUT, check=False)
        if not os.path.exists(bases):
            raise SystemExit(f"annotators produced nothing; see {log}")
        print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)

    print("\n[3/3] consensus, resolver and certificate", flush=True)
    import anndata as ad
    import nacre_mlh as MLH, nacre_ml as NML
    MLH.HIER = hier
    vc = ad.read_h5ad(r_h5, backed="r").obs["major_annotation"].astype(str).value_counts()
    rare = set((vc / vc.sum())[lambda s: s < 0.01].index)
    B = pd.read_csv(bases, dtype={"cell": str})
    # A tool can fail inside R while the other three succeed, and the run then continues on a panel
    # of three. The vote structure is defined over four annotators, so 4/4, 3/1 and 2/2 would all be
    # unreachable or mean something different; refuse rather than emit a certificate that looks
    # normal and is not comparable. The R log names the cause.
    dead = [t for t in ("Seurat", "Azimuth", "RCTD", "SingleR")
            if t in B.columns and B[t].notna().sum() == 0]
    if dead:
        raise SystemExit(
            f"{', '.join(dead)} produced no labels for any cell, so the panel is not four "
            f"annotators and the vote structure is undefined. See {log} for the reason, fix it, "
            f"delete {os.path.basename(bases)} and rerun.")
    res = MLH.run_nacre_mlh(name, q_h5, B, rare_set=rare, hierarchy_path=hier)
    res.to_csv(mlh, index=False)

    lin, comp = MLH.load_hierarchy(hier)
    up = {"subtype": lambda x: x,
          "lineage": np.vectorize(lambda v: lin.get(v, v)),
          "compartment": np.vectorize(lambda v: comp.get(v, v))}
    V = res[NML.TOOLS].astype(str).values
    rows = []
    print(f"\n{'level':<13}{'before':>9}{'after':>9}{'delta':>9}")
    for lev in MLH.LEVELS:
        Vl = up[lev](V)
        before = np.mean([(Vl[:, i] == Vl[:, j]).mean()
                          for i in range(4) for j in range(i + 1, 4)]) * 100
        Ll = up[lev](res[f"predicted_{lev}"].astype(str).values)
        after = np.mean([(Vl[:, t] == Ll).mean() for t in range(4)]) * 100
        rows.append((lev, before, after, after - before))
        print(f"{lev:<13}{before:>9.2f}{after:>9.2f}{after-before:>+9.2f}")
    pd.DataFrame(rows, columns=["level", "before", "after", "delta"]).to_csv(
        mlh.replace(".csv", "_concordance.csv"), index=False)

    print("\nresolution certificate:")
    rc = res["resolution"].value_counts()
    for k in ("subtype", "lineage", "compartment", "unresolved"):
        n = int(rc.get(k, 0))
        print(f"   {k:<13}{n:>9,}  {100*n/len(res):>6.2f}%")

    if a.llm_panel:
        print("\n[extra] held-out LLM panel", flush=True)
        os.environ.setdefault("OPENAI_API_KEY", a.llm_key or "")
        import llm_annotate
        llm_annotate.run(name, q_h5, bases, name, outdir=a.out, model="gpt-4o")

    print(f"\nwrote {mlh}")


if __name__ == "__main__":
    main()
