"""Platform adapters: turn a vendor output directory into the one object NACRE-MLH needs.

NACRE-MLH itself is platform-agnostic. It reads exactly three things from a query:

    X                     integer counts, cells x genes
    obsm["spatial"]       2D coordinates
    obs["high_quality"]   optional QC flag

Everything vendor-specific lives here: which files to look for, which columns hold the coordinates,
which features are negative controls rather than genes, and what filter thresholds are sensible given
the panel size.

Two thresholds vary by platform and both have bitten us in practice.

`min_umi` is passed to RCTD's Reference(). RCTD computes it over *shared panel genes only*, so a fixed
value is stricter on a narrow panel than a wide one: 10 counts across 5,001 Xenium probes is easy, the
same 10 across a 140-gene MERSCOPE panel is not. Its default of 100 once cut a reference from 10,689
cells to 1,950 and silently erased four cell types, so it is set explicitly here, never left implicit.

`min_counts` is the query QC floor and scales the same way.

Both are defaults, not findings. Override them with --min-umi / --min-counts when a panel is unusual.
"""
import os, glob, gzip, io, re
import numpy as np, pandas as pd

PLATFORMS = {"X": "Xenium", "M": "MERSCOPE", "C": "CosMx"}


class Platform:
    code, name = None, None
    min_umi = 10        # RCTD reference floor, over shared panel genes
    min_counts = 10     # query cell floor, transcripts over the panel
    control_patterns = ()

    @staticmethod
    def find(d, patterns):
        for p in patterns:
            hits = sorted(glob.glob(os.path.join(d, p)))
            if hits:
                return hits[0]
        return None

    @classmethod
    def drop_controls(cls, genes):
        """Negative-control probes are not genes; if they survive into X they become features the
        reference can never match."""
        keep = np.ones(len(genes), bool)
        for pat in cls.control_patterns:
            keep &= ~pd.Series(genes).str.match(pat, case=False, na=False).to_numpy()
        return keep

    @classmethod
    def load(cls, d):
        raise NotImplementedError


class Xenium(Platform):
    code, name = "X", "Xenium"
    min_umi, min_counts = 10, 10
    # the h5 carries feature_type, so controls are filtered by type rather than by name
    control_patterns = (r"^NegControl", r"^BLANK", r"^Unassigned", r"^DeprecatedCodeword")

    @classmethod
    def load(cls, d):
        import h5py, scipy.sparse as sps
        h5 = cls.find(d, ["*cell_feature_matrix.h5", "cell_feature_matrix.h5"])
        if h5 is None:
            raise FileNotFoundError(f"no cell_feature_matrix.h5 under {d}")
        with h5py.File(h5, "r") as h:
            g = h["matrix"]
            ft = np.array([x.decode() for x in g["features"]["feature_type"][:]])
            genes = np.array([x.decode() for x in g["features"]["name"][:]])
            barcodes = np.array([x.decode() for x in g["barcodes"][:]])
            shape = tuple(int(v) for v in g["shape"][:])
            M = sps.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        keep = ft == "Gene Expression"
        X = M[keep, :].T.tocsr()
        genes = genes[keep]

        cf = cls.find(d, ["*cells.csv.gz", "*cells.parquet.gz", "*cells.csv", "*cells.parquet"])
        if cf is None:
            raise FileNotFoundError(f"no cells table under {d}")
        if cf.endswith(".parquet.gz"):
            meta = pd.read_parquet(io.BytesIO(gzip.open(cf, "rb").read()))
        elif cf.endswith(".parquet"):
            meta = pd.read_parquet(cf)
        else:
            meta = pd.read_csv(cf)
        meta = meta.set_index("cell_id").reindex(barcodes)
        xy = meta[["x_centroid", "y_centroid"]].to_numpy(np.float32)
        tx = (meta["transcript_counts"].to_numpy(float) if "transcript_counts" in meta
              else np.asarray(X.sum(1)).ravel())
        return X, genes, np.asarray(barcodes, str), xy, tx


class MERSCOPE(Platform):
    code, name = "M", "MERSCOPE"
    # panels run 140-500 genes and cells are correspondingly shallow, so a floor of 10 over the panel
    # would discard a large share of otherwise usable reference cells
    min_umi, min_counts = 5, 5
    control_patterns = (r"^Blank[-_]", r"^Blank\d")

    @classmethod
    def load(cls, d):
        import scipy.sparse as sps
        from . import schema
        # Identified by column signature, not by name, for the same reason as CosMx: filenames and
        # layout vary between MERSCOPE software versions and between deposits.
        roles = schema.detect(d)
        cbg, cm = roles.get("expression"), roles.get("metadata")
        if cbg is None or cm is None:
            have = ", ".join(f"{r}={os.path.basename(p)}" for r, p in sorted(roles.items())) or "none"
            raise FileNotFoundError(
                f"could not identify a cell-by-gene matrix and a cell metadata table under {d}. "
                f"Recognized: {have}.")
        print(f"  expression <- {os.path.basename(cbg)} | metadata <- {os.path.basename(cm)}",
              flush=True)
        # MERSCOPE writes the cell id as the CSV index in both files.
        E = pd.read_csv(cbg, index_col=0)
        genes = np.array(E.columns, dtype=str)
        keep = cls.drop_controls(genes)
        X = sps.csr_matrix(E.to_numpy()[:, keep].astype(np.float32))
        genes = genes[keep]
        meta = pd.read_csv(cm, index_col=0).reindex(E.index)
        pair = schema.find_xy(meta.columns)
        if pair is None:
            raise KeyError(f"no centroid columns in {os.path.basename(cm)}; "
                           f"recognized spellings: {schema.XY_PAIRS}")
        xy = meta[list(pair)].to_numpy(np.float32)
        # A cell present in the matrix but absent from the metadata has no position; dropping it here
        # keeps NaN coordinates out of the resolver's neighborhood features.
        ok = np.isfinite(xy).all(1)
        idx = np.asarray(E.index, str)
        if not ok.all():
            print(f"  dropped {int((~ok).sum())} cell(s) with no metadata entry", flush=True)
            X, xy, idx = X[ok], xy[ok], idx[ok]
        return X, genes, idx, xy, np.asarray(X.sum(1)).ravel()


class CosMx(Platform):
    code, name = "C", "CosMx"
    min_umi, min_counts = 10, 10
    # Legacy exports name background probes NegPrb*; AtoMx SIP uses Negative* and adds SystemControl*;
    # both may carry FalseCode*, barcodes absent from the probe mix. None of them are genes.
    control_patterns = (r"^NegPrb", r"^SystemControl", r"^Negative", r"^FalseCode")
    # Every non-gene column the documented exports put in the expression matrix. `cell` and `cell_id`
    # hold strings such as "c_1_3_86" and would otherwise be parsed as a gene and crash the cast.
    ID_COLS = ("fov", "cell_ID", "cell_id", "cell", "slide_ID")

    @classmethod
    def _matrix_from_transcripts(cls, tx_path, d):
        """Pivot a transcript table into a cell-by-gene matrix.

        Deposits that ship only transcript-level data are common enough to be worth supporting: the
        matrix is derivable, and refusing the dataset over a missing file it does not need would be
        arbitrary. Rows with no cell assignment are excluded here rather than downstream.
        """
        out = os.path.join(d, "_derived_exprMat.csv")
        if os.path.exists(out):
            return out
        tx = pd.read_csv(tx_path)
        low = {str(c).strip().lower(): c for c in tx.columns}
        gcol = next((low[c] for c in ("target", "gene", "gene_name", "target_name") if c in low),
                    None)
        idc = [low[c] for c in ("fov", "cell_id", "cell") if c in low]
        if gcol is None or not idc:
            raise KeyError(f"{os.path.basename(tx_path)} has no gene column or no cell identifier")
        keep = ~tx[gcol].isna()
        # Keep the transcript table's own identifier columns rather than collapsing them into one.
        # The metadata table keys on those same columns, and a single joined "cell" column would
        # share no identifier with it.
        grp = tx.loc[keep, idc].astype(str)
        cell = grp.agg("_".join, axis=1) if len(idc) > 1 else grp[idc[0]]
        M = pd.crosstab(cell, tx.loc[keep, gcol])
        M.index.name = "_key"
        M = M.reset_index()
        parts = M["_key"].str.split("_", n=len(idc) - 1, expand=True)
        for i, c in enumerate(idc):
            M[c] = parts[i]
        M = M.drop(columns="_key")
        M = M[idc + [c for c in M.columns if c not in idc]]
        M.to_csv(out, index=False)
        print(f"  no cell-by-gene matrix found; derived one from "
              f"{os.path.basename(tx_path)} ({M.shape[0]:,} cells x {M.shape[1]-1:,} genes)",
              flush=True)
        return out

    @classmethod
    def load(cls, d):
        import scipy.sparse as sps
        from . import schema
        # Files are identified by their column signature, not their name: export version, instrument
        # versus repository, and each author's own reorganization all change the filenames and the
        # directory layout while leaving the schemas recognizable.
        roles = schema.detect(d)
        ef, mf = roles.get("expression"), roles.get("metadata")
        if ef is None and roles.get("transcripts"):
            # Some deposits ship transcript-level data and no cell-by-gene matrix. Build it.
            ef = cls._matrix_from_transcripts(roles["transcripts"], d)
        if ef is None or mf is None:
            have = ", ".join(f"{r}={os.path.basename(p)}" for r, p in sorted(roles.items())) or "none"
            raise FileNotFoundError(
                f"could not identify a cell-by-gene matrix and a cell metadata table under {d}. "
                f"Recognized: {have}. Files are matched on their columns, so a table needs a cell "
                f"identifier plus either many numeric feature columns (expression) or a centroid "
                f"(metadata).")
        print(f"  expression <- {os.path.basename(ef)} | metadata <- {os.path.basename(mf)}",
              flush=True)
        E = pd.read_csv(ef)
        meta = pd.read_csv(mf)
        # Key on every identifier both files share. fov+cell_ID is unique only within a slide, so an
        # export holding more than one slide needs slide_ID or the study-wide cell/cell_id as well.
        idcols = [c for c in cls.ID_COLS if c in E.columns and c in meta.columns]
        if not idcols:
            raise KeyError(f"no shared cell identifier between {os.path.basename(ef)} and "
                           f"{os.path.basename(mf)}; looked for {cls.ID_COLS}")
        join = lambda df: (df[idcols].astype(str).agg("_".join, axis=1) if len(idcols) > 1
                           else df[idcols[0]].astype(str))
        key, mkey = join(E), join(meta)
        if mkey.duplicated().any():
            raise KeyError(f"{os.path.basename(mf)} has {int(mkey.duplicated().sum())} duplicate "
                           f"ids under {idcols}; a multi-slide export needs a slide_ID or cell_id "
                           f"column to disambiguate")
        # Genes are the numeric columns that are not identifiers: an export may also carry string
        # annotation columns, and those must not be cast into the count matrix.
        drop = set(cls.ID_COLS)
        genes = np.array([c for c in E.columns
                          if c not in drop and pd.api.types.is_numeric_dtype(E[c])], dtype=str)
        keep = cls.drop_controls(genes)
        X = sps.csr_matrix(E[genes].to_numpy()[:, keep].astype(np.float32))
        genes = genes[keep]
        meta = meta.set_index(mkey).reindex(key)
        # One vocabulary of centroid spellings, shared with the detector, so a file the detector
        # recognized can never then fail here for want of a column name it already matched. Global
        # coordinates rank first: CosMx local coordinates restart at the origin in every field of
        # view, so on a multi-FOV slide they stack unrelated cells on one another and the resolver's
        # neighborhood feature, a k-NN in joint expression-position space, would treat cells from
        # different FOVs as neighbors. Local is correct only for a single FOV.
        pair = schema.find_xy(meta.columns)
        if pair is None:
            raise KeyError(f"no centroid columns in {os.path.basename(mf)}; "
                           f"recognized spellings: {schema.XY_PAIRS}")
        cx, cy = pair
        if cx.endswith("local_px") and "fov" in meta.columns and meta["fov"].nunique() > 1:
            print(f"  WARNING: only FOV-local coordinates found, and this slide has "
                  f"{meta['fov'].nunique()} FOVs; spatial features will be unreliable", flush=True)
        xy = meta[[cx, cy]].to_numpy(np.float32)
        # The expression matrix carries one extra row per FOV with cell_ID = 0: the transcripts
        # assigned to no cell. It has no metadata entry, so it survives the join only as NaN
        # coordinates. Drop anything the metadata does not describe, which covers that row and any
        # other unmatched id, rather than letting a non-cell reach the annotators.
        ok = np.isfinite(xy).all(1)
        if not ok.all():
            print(f"  dropped {int((~ok).sum())} row(s) with no metadata entry "
                  f"(CosMx exports one per FOV for off-cell transcripts)", flush=True)
            X, xy, key = X[ok], xy[ok], np.asarray(key, str)[ok]
        return X, genes, np.asarray(key, str), xy, np.asarray(X.sum(1)).ravel()


ADAPTERS = {"X": Xenium, "M": MERSCOPE, "C": CosMx}


def get(code):
    code = str(code).strip().upper()[:1]
    if code not in ADAPTERS:
        raise SystemExit(f"unknown platform '{code}'. Use X (Xenium), M (MERSCOPE) or C (CosMx).")
    return ADAPTERS[code]
