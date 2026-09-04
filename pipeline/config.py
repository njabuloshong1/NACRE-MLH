"""Read the run sheet.

One row per dataset. CSV, TSV or a whitespace/`key = value` text file all parse to the same thing,
because the point of the run sheet is that a person edits it by hand and should not have to remember
which delimiter this pipeline wanted.

Required per row:
    name        run name; becomes the output subdirectory
    platform    X | M | C  (or xenium / merscope / cosmx, case-insensitive)
    query       vendor output directory, or a single .h5ad
    reference   directory holding the reference .h5ad, or the .h5ad itself

Optional per row:
    ref_label   cell-type column in the reference    (default: the pipeline picks one)
    truth       ground-truth cell-type column, in the query or in truth_file
    truth_file  a CSV/TSV of ground-truth labels, if they live outside the query
    min_counts  query cell floor                     (default: per-platform)
    min_umi     RCTD reference floor                 (default: per-platform)
    skip        y/1/true to leave the row out without deleting it

Ground truth is entirely optional. Give truth_file, or name a column with truth, or leave both blank
and the pipeline will look for a likely label column in the query. With none of the three, accuracy
is simply not reported; everything else runs unchanged.
"""
import os, csv, re

PLATFORMS = {"X": "X", "XENIUM": "X", "M": "M", "MERSCOPE": "M", "MERFISH": "M",
             "C": "C", "COSMX": "C", "SMI": "C"}
REQUIRED = ("name", "platform", "query", "reference")
KNOWN = REQUIRED + ("ref_label", "truth", "truth_file", "min_counts", "min_umi", "skip")
ALIAS = {"ref": "reference", "run": "name", "dataset": "name", "id": "name",
         "gt_file": "truth_file", "truth_path": "truth_file", "gt_path": "truth_file",
         "gtfile": "truth_file", "ground_truth_file": "truth_file", "labels_file": "truth_file",
         "plat": "platform", "tech": "platform", "technology": "platform",
         "query_dir": "query", "querypath": "query", "query_path": "query",
         "reference_dir": "reference", "ref_path": "reference", "refpath": "reference",
         "label": "ref_label", "reflabel": "ref_label", "ref_column": "ref_label",
         "gt": "truth", "truth_column": "truth", "ground_truth": "truth"}


class ConfigError(Exception):
    pass


def _norm(k):
    k = re.sub(r"[^a-z0-9_]", "", str(k).strip().lower().replace(" ", "_"))
    return ALIAS.get(k, k)


def _truthy(v):
    return str(v).strip().lower() in {"1", "y", "yes", "true", "t"}


def _content_lines(path):
    """Lines with comments and blanks removed. The template ships with a comment header, so this
    has to happen before csv sees the file or the first '#' line becomes the field names."""
    out = []
    for raw in open(path, encoding="utf-8-sig"):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(raw.rstrip("\r\n"))
    return out


def _rows_from_delimited(lines):
    try:
        dial = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t|")
    except csv.Error:
        dial = csv.excel                          # single column, or one row: comma is a safe default
    return [dict(r) for r in csv.DictReader(lines, dialect=dial)]


def _rows_from_blocks(path):
    """`key = value` lines, blank line or [name] header between runs.

    Reads the file itself rather than _content_lines, because here a blank line is meaningful: it
    separates one run from the next.
    """
    rows, cur = [], {}
    for raw in open(path, encoding="utf-8-sig"):
        line = raw.split("#")[0].strip()
        if not line:
            if cur: rows.append(cur); cur = {}
            continue
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            if cur: rows.append(cur)
            cur = {"name": m.group(1).strip()}; continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line and not re.match(r"^[A-Za-z]:[\\/]", line):
            k, v = line.split(":", 1)             # not a Windows drive letter
        else:
            continue
        cur[k.strip()] = v.strip()
    if cur: rows.append(cur)
    return rows


def load(path):
    """-> list of dicts, validated. Raises ConfigError naming the row and the problem."""
    if not os.path.exists(path):
        raise ConfigError(f"run sheet not found: {path}")
    lines = _content_lines(path)
    if not lines:
        raise ConfigError(f"{path} is empty (or entirely comments)")
    # a delimited sheet has its field names on the first real line; a block file starts with
    # '[name]' or 'key = value'
    first = lines[0]
    delimited = bool(re.search(r"[,;\t|]", first)) and not first.startswith("[") and "=" not in first
    raw = _rows_from_delimited(lines) if delimited else _rows_from_blocks(path)

    runs, seen = [], set()
    for i, r0 in enumerate(raw, 1):
        r = {_norm(k): (v.strip() if isinstance(v, str) else v)
             for k, v in r0.items() if k is not None and str(k).strip()}
        r = {k: v for k, v in r.items() if v not in (None, "")}
        if not r:
            continue
        if _truthy(r.get("skip", "")):
            continue
        where = f"row {i}" + (f" ({r['name']})" if "name" in r else "")

        missing = [k for k in REQUIRED if not r.get(k)]
        if missing:
            raise ConfigError(f"{where}: missing {', '.join(missing)}")
        unknown = sorted(set(r) - set(KNOWN))
        if unknown:
            raise ConfigError(f"{where}: unrecognised column(s) {', '.join(unknown)}. "
                              f"Allowed: {', '.join(KNOWN)}")

        p = PLATFORMS.get(str(r["platform"]).strip().upper())
        if p is None:
            raise ConfigError(f"{where}: platform '{r['platform']}' is not one of "
                              f"X/Xenium, M/MERSCOPE, C/CosMx")
        r["platform"] = p

        if r["name"] in seen:
            raise ConfigError(f"{where}: duplicate name '{r['name']}'; names become output folders")
        seen.add(r["name"])
        if not re.match(r"^[A-Za-z0-9._-]+$", r["name"]):
            raise ConfigError(f"{where}: name '{r['name']}' must be letters, digits, . _ or - only")

        # Paths are checked now rather than an hour into the annotators.
        for k in ("query", "reference", "truth_file"):
            if not r.get(k):
                continue
            r[k] = os.path.abspath(os.path.expandvars(os.path.expanduser(r[k])))
            if not os.path.exists(r[k]):
                raise ConfigError(f"{where}: {k} does not exist: {r[k]}")
        for k in ("min_counts", "min_umi"):
            if k in r:
                try:
                    r[k] = int(float(r[k]))
                except ValueError:
                    raise ConfigError(f"{where}: {k} must be a whole number, got '{r[k]}'")
        runs.append(r)

    if not runs:
        raise ConfigError(f"{path} defines no runs (all rows blank or skipped)")
    return runs


TEMPLATE = """\
# NACRE-MLH run sheet. One row per dataset. Lines starting with # are ignored.
#
#   platform    X (Xenium) | M (MERSCOPE) | C (CosMx)
#   query       vendor output folder, or a single .h5ad
#   reference   folder holding the reference .h5ad, or the .h5ad itself
#   ref_label   OPTIONAL cell-type column in the reference
#   truth       OPTIONAL ground-truth column name (in the query, or inside truth_file)
#   truth_file  OPTIONAL CSV/TSV of ground-truth labels, if they are not in the query
#   skip        OPTIONAL put y here to leave a row out without deleting it
#
# Ground truth is optional. Point at a file, name a column, or leave both blank and the pipeline
# looks for a likely label column in your query. With none of the three, accuracy is simply not
# reported and everything else runs the same.
#
name,platform,query,reference,ref_label,truth,truth_file,skip
MyXenium,X,D:/data/xenium_run,D:/refs/breast_ref.h5ad,celltype,,,
MyCosMx,C,D:/data/cosmx_export,D:/refs/ovarian_ref.h5ad,Cluster_Detailed,ori_celltype,,
MyLabelled,X,D:/data/other_run,D:/refs/ref.h5ad,,cell_type,D:/data/my_labels.csv,
MyMERSCOPE,M,D:/data/merscope_run,D:/refs/lung_ref.h5ad,,,,y
"""
