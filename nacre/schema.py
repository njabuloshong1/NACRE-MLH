"""Identify vendor files by what is inside them, not by what they are called.

Vendor exports differ in filename, directory layout and which files are even present: instrument
output versus a processed deposit, one software version versus another, an author's own
reorganization. Two real CosMx layouts look like this:

    a/  exprMat_file.csv  metadata_file.csv  tx_file.csv  fov_positions_file.csv  polygons.csv
    b/  CellStats.csv  RawTranscriptData.csv  expression_matrix/  CellComposite/  Morphology2D/

Nothing about the names is stable, so matching on them is guesswork. Every one of those files does
have a recognizable column signature, which is stable: an expression matrix is wide and almost
entirely numeric, cell metadata pairs an identifier with a centroid, a transcript table is long and
carries one gene name plus one coordinate per row.

This module reads only the header of each candidate file, scores it against those signatures, and
returns the best file for each role. Callers get a role, not a filename.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

TABULAR = ("*.csv", "*.csv.gz", "*.tsv", "*.tsv.gz", "*.txt", "*.txt.gz")

# Column vocabularies, lowercased for matching. Vendors vary the spelling of every one of these.
ID_NAMES = {"cell_id", "cellid", "cell", "fov", "slide_id", "cell_index", "cellcomposite",
            "entityid", "entity_id", "barcode"}


def _is_index_col(name):
    """MERSCOPE writes the cell id as the CSV index, so pandas names it "Unnamed: 0".

    An unnamed leading column is an identifier, not a measurement, and without this a real
    cell_metadata.csv is unrecognizable: its only other columns are a centroid and morphology.
    """
    return bool(re.fullmatch(r"unnamed:\s*\d+", str(name).strip().lower()))


XY_PAIRS = [
    ("centerx_global_px", "centery_global_px"), ("centerx_local_px", "centery_local_px"),
    ("x_global_px", "y_global_px"), ("x_local_px", "y_local_px"),
    ("center_x", "center_y"), ("x_centroid", "y_centroid"),
    ("x_slide_mm", "y_slide_mm"), ("global_x", "global_y"), ("x", "y"),
]
GENE_NAMES = {"target", "gene", "gene_name", "target_name", "feature_name", "featurename"}
MORPH_NAMES = {"area", "aspectratio", "width", "height", "circularity", "eccentricity",
               "perimeter", "solidity", "nucarea", "area.um2", "mean.dapi", "max.dapi",
               # MERSCOPE cell_metadata carries a volume and a bounding box instead
               "volume", "min_x", "max_x", "min_y", "max_y"}


def _peek(path, n=5):
    """Header plus a few rows. Never reads a 700 MB transcript table whole."""
    for kw in ({}, {"sep": "\t"}):
        try:
            d = pd.read_csv(path, nrows=n, **kw)
            if d.shape[1] > 1:
                return d
        except Exception:
            continue
    return None


def _cols(d):
    return [str(c).strip().lower() for c in d.columns]


def find_xy(columns):
    """Return the centroid column pair, preferring a global frame over a per-FOV one."""
    low = {str(c).strip().lower(): c for c in columns}
    for a, b in XY_PAIRS:
        if a in low and b in low:
            return low[a], low[b]
    return None


def classify(path):
    """Score one file against each role. Returns (role, score, dataframe-head) or None."""
    d = _peek(path)
    if d is None or d.empty:
        return None
    cols = _cols(d)
    ncol = len(cols)
    ids = [c for c in cols if c in ID_NAMES or _is_index_col(c)]
    xy = find_xy(d.columns)
    genes = [c for c in cols if c in GENE_NAMES]
    morph = [c for c in cols if c in MORPH_NAMES]
    numeric = [c for c, dt in zip(cols, d.dtypes) if pd.api.types.is_numeric_dtype(dt)]

    scores = {}

    # Expression matrix: a cell identifier, many numeric columns that are feature names rather than
    # measurements, and no centroid. The absent centroid is what separates it from cell metadata,
    # which also pairs an identifier with numeric columns; a counts table never carries positions.
    non_id_numeric = [c for c in numeric
                      if c not in ID_NAMES and c not in MORPH_NAMES and not _is_index_col(c)]
    # The ratio stays loose deliberately: an export may carry several identifier and annotation
    # columns beside the counts, and on a small panel a strict ratio rejects a valid matrix.
    if ids and not xy and len(non_id_numeric) >= 4 and len(non_id_numeric) / ncol > 0.5:
        scores["expression"] = 2.5 + min(len(non_id_numeric) / 1000.0, 1.0)

    # Transcript table: one gene name and one position per row, and narrow.
    if genes and xy and ncol <= 20:
        scores["transcripts"] = 3.0

    # Cell metadata: one row per cell, so it needs a cell-level identifier and a centroid. `fov`
    # alone does not qualify, or an FOV position table would look like sparse cell metadata.
    cell_ids = [c for c in ids if c not in ("fov", "slide_id")]
    if cell_ids and xy and len(non_id_numeric) < 200:
        scores["metadata"] = 2.0 + 0.5 * bool(morph) + (0.5 if ncol <= 100 else 0)

    # FOV positions: keyed by fov, a centroid, and no cell identifier at all.
    if "fov" in cols and xy and ncol <= 8 and not cell_ids:
        scores["fov_positions"] = 2.5

    if not scores:
        return None
    role = max(scores, key=scores.get)
    return role, scores[role], d


def detect(root, verbose=True):
    """Map every tabular file under `root` to a role, best candidate per role.

    Name is used only to break ties between equally-scoring candidates, never to decide.
    """
    cands = []
    for pat in TABULAR:
        cands += glob.glob(os.path.join(root, "**", pat), recursive=True)
    cands = sorted(set(cands))

    found = {}
    report = []
    for p in cands:
        got = classify(p)
        if not got:
            report.append((os.path.relpath(p, root), "-", 0))
            continue
        role, score, _ = got
        # a bigger file wins a tie: the real matrix beats a small excerpt
        score += min(os.path.getsize(p) / 1e10, 0.4)
        report.append((os.path.relpath(p, root), role, score))
        if role not in found or score > found[role][1]:
            found[role] = (p, score)

    if verbose:
        print("  schema detection:")
        for rel, role, score in sorted(report, key=lambda r: (r[1], -r[2])):
            mark = "->" if role != "-" and found.get(role, ("",))[0].endswith(rel.split(os.sep)[-1]) \
                else "  "
            print(f"    {mark} {rel:<46} {role:<14}{score:.2f}" if role != "-"
                  else f"       {rel:<46} (unrecognized)")
    return {r: p for r, (p, _) in found.items()}
