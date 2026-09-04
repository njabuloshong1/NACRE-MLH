"""NACRE-MLH pipeline. One command, a run sheet, and everything else is produced for you.

    python run_pipeline.py --config runs.csv --out E:/results/mynewruns

The run sheet names, per dataset, the platform, the query and the reference. Nothing else is
required. An LLM key is needed only when a reference uses cell-type names the shipped hierarchy has
not seen, and the pipeline says so before it starts rather than an hour in.

Each run produces, in <out>/<name>/:
    nacre_mlh_<name>.csv              per-cell labels, concordance and certificate
    nacre_mlh_<name>_concordance.csv  before/after concordance at each level
    figures/                          the figure set
    query_<name>.h5ad, ref_<name>.h5ad, bases_<name>.csv    intermediates, reused on rerun
    log_<name>.txt                    the annotators' own log

and across runs, in <out>/:
    summary_*.csv, SUMMARY.md, NACRE-MLH_report.docx

Reruns skip work already done; --fresh forces a run to be rebuilt.
"""
import os, sys, time, json, argparse, subprocess, traceback, warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "nacre"))

import numpy as np, pandas as pd
from pipeline import (config as cfgmod, figures as figmod, summary as summod,
                      resources as resmod, export as expmod)

CLI = os.path.join(HERE, "nacre_mlh_cp.py")
TIERS = ["subtype", "lineage", "compartment", "unresolved"]


def r_missing(rscript):
    """Which required R packages this interpreter lacks. None means the probe itself failed."""
    found, missing = resmod.check_r(rscript)
    return None if found is None and missing == list(resmod.R_REQUIRED) else missing


def find_rscript(explicit=None, verbose=True):
    """Pick an R that actually has the annotator packages.

    Several R versions are commonly installed side by side and only one of them carries Seurat,
    spacexr and the rest. Choosing by version number picks the newest, which is usually the emptiest;
    the failure then surfaces an hour later inside the annotators. Probe instead, and prefer an
    interpreter that can actually run the job.
    """
    import shutil, glob
    for c in [explicit, os.environ.get("RSCRIPT")]:
        if c and os.path.exists(c):
            return c                       # explicit choice is respected, and checked in preflight
    cands, seen = [], set()
    w = shutil.which("Rscript")
    if w: cands.append(w); seen.add(os.path.normcase(w))
    for p in [r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
              r"C:\Program Files\R\R-*\bin\Rscript.exe",
              r"C:\Program Files\Microsoft\R Open\R-*\bin\x64\Rscript.exe"]:
        for f in sorted(glob.glob(p), reverse=True):       # newest first, then fall back
            if os.path.normcase(f) not in seen:
                cands.append(f); seen.add(os.path.normcase(f))
    if not cands:
        return None
    partial = None
    for c in cands:
        miss = r_missing(c)
        if miss == []:
            if verbose and c != cands[0]:
                print(f"  note: using {c}\n        (earlier R installs lack the annotator packages)",
                      flush=True)
            return c
        if miss and partial is None:
            partial = (c, miss)
    if partial:                            # nothing complete: return the best one so preflight can
        return partial[0]                  # report exactly which packages are missing
    return cands[0]


def read_key(explicit=None):
    """Key from the flag, then the environment, then key.txt beside this script."""
    if explicit: return explicit.strip()
    e = os.environ.get("OPENAI_API_KEY")
    if e: return e.strip()
    p = os.path.join(HERE, "key.txt")
    if os.path.exists(p):
        k = open(p, encoding="utf-8").read().strip()
        if k: return k
    return None


def preflight(runs, rscript, key):
    """Fail before the first annotator starts, not an hour into it."""
    problems = []
    found_py, missing_py, outdated_py = resmod.check_python()
    found_r, missing_r = (None, None)

    if rscript is None:
        problems.append("Rscript not found. Install R, or pass --rscript C:/path/to/Rscript.exe")
    else:
        # The annotators run in a subprocess an hour into the job. A missing R package there costs
        # the whole run, so check it here where the fix is cheap.
        found_r, missing_r = resmod.check_r(rscript)
        if found_r is None and missing_r == list(resmod.R_REQUIRED):
            problems.append(f"{rscript} could not be run. Check the path, or pass --rscript.")
        elif missing_r:
            problems.append(
                f"{rscript} is missing R package(s): {', '.join(missing_r)}\n"
                f"    NOT installed for you. Install them in THAT R, or point --rscript at an R "
                f"that already has them:\n"
                f'      "{rscript}" -e \'BiocManager::install(c('
                f'{", ".join(chr(34)+m+chr(34) for m in missing_r)}))\'')
        for p, have, need in resmod.r_outdated(found_r):
            problems.append(f"R package {p} is {have}, needs >= {need}")

    print(resmod.describe(found_py, rscript, found_r), flush=True)
    avail, how = resmod.available_bytes()
    print(f"memory  {avail/1024**3:.1f} GB free ({how})\n", flush=True)

    if not os.path.exists(CLI):
        problems.append(f"missing {CLI}")
    for m in missing_py:
        problems.append(f"python package '{m}' is not installed (in {sys.executable})")
    for p, have, need in outdated_py:
        problems.append(f"python package {p} is {have}, needs >= {need}")
    try:
        H = set(pd.read_csv(os.path.join(HERE, "assets", "hierarchy.csv")).label.astype(str))
    except Exception as e:
        problems.append(f"cannot read assets/hierarchy.csv: {e}")
        H = None
    if H is not None and not key:
        # only a warning: many references are already covered, and we cannot know which without
        # opening each one, which is slow. The per-run step raises if a key turns out to be needed.
        print("  note: no LLM key supplied. Runs whose reference uses unfamiliar cell-type names "
              "will stop and tell you.", flush=True)
    if problems:
        print("\nCannot start:\n" + "\n".join(f"  - {p}" for p in problems), file=sys.stderr)
        raise SystemExit(2)


def parse_stdout(text):
    """Pull the numbers the CLI prints, so the summary does not depend on re-deriving them."""
    got = {}
    for line in text.splitlines():
        s = line.strip()
        if "lock=" in s and "strong=" in s:
            for k in ("lock", "strong", "split"):
                try: got[k] = float(s.split(f"{k}=")[1].split("%")[0])
                except Exception: pass
        if s.startswith("query:") and "cells" in s:
            try:
                got["cells"] = int(s.split("query:")[1].split("cells")[0].strip().replace(",", ""))
                got["genes"] = int(s.split("x")[1].split("genes")[0].strip().replace(",", ""))
            except Exception: pass
        if s.startswith("reference:") and "panel genes shared" in s:
            try:
                got["shared"] = s.split("panel genes shared")[0].split(":")[1].strip()
                got["ref_cells"] = int(s.split("|")[1].split("cells")[0].strip().replace(",", ""))
                got["ref_types"] = int(s.split("cells,")[1].split("types")[0].strip())
            except Exception: pass
    return got


# Columns that look like a cell-type annotation. Ordered: an earlier pattern wins.
TRUTH_HINTS = ["ori_celltype", "ori.celltype", "ground_truth", "groundtruth", "gt",
               "cell_type", "celltype", "cell_types", "annotation", "annotations",
               "major_annotation", "label", "labels", "class", "subclass", "cluster_name"]
# Never auto-select these: pipeline bookkeeping, or the output of a tool we are being scored against.
TRUTH_NEVER = {"high_quality", "transcript_counts", "cell", "cell_id", "fov", "sample", "batch",
               "seurat", "azimuth", "rctd", "singler", "nacre", "predicted_subtype",
               "predicted_lineage", "predicted_compartment", "resolution", "usable_label"}


def _id_and_label(df, want=None):
    """In a table of labels, work out which column holds the cell id and which the label."""
    idc = None
    for c in df.columns:
        if str(c).strip().lower() in ("cell", "cell_id", "cellid", "barcode", "cell_ID", "id"):
            idc = c; break
    if idc is None:
        idc = df.columns[0]                       # exported label tables put the id first
    if want and want in df.columns:
        return idc, want
    rest = [c for c in df.columns if c != idc]
    if len(rest) == 1:
        return idc, rest[0]
    for c in rest:                                # otherwise the most annotation-looking column
        if str(c).strip().lower().replace(" ", "_") in TRUTH_HINTS:
            return idc, c
    return idc, (rest[0] if rest else None)


def _obs_candidates(a):
    """obs columns that plausibly hold a cell-type annotation.

    Auto-selection is deliberately restricted to columns whose NAME says they are annotations.
    Accepting any categorical column is far too loose: a Xenium export carries `segmentation_method`
    with three values, which satisfies every structural test and is not a cell type, and silently
    scoring against it produces a confident, meaningless accuracy. A column nobody named and whose
    name gives no indication is not ground truth, so it is offered as a suggestion instead.

    Returns (auto, suggestions): `auto` may be used without asking, `suggestions` only reported.
    """
    auto, suggest = [], []
    for c in a.obs.columns:
        key = str(c).strip().lower().replace(" ", "_").replace(".", "_")
        if key in TRUTH_NEVER or any(t in key for t in ("seurat", "azimuth", "rctd", "singler",
                                                        "segmentation", "codeword", "probe")):
            continue
        s = a.obs[c]
        if s.dtype.kind in "biufc" and str(s.dtype) != "category":
            continue                              # numeric columns are not annotations
        n = s.astype(str).nunique()
        if not (2 <= n <= 300):
            continue
        if key in TRUTH_HINTS:
            auto.append((TRUTH_HINTS.index(key), str(c), n))
        elif any(h in key for h in ("celltype", "cell_type", "annotation", "ident", "_type")):
            auto.append((len(TRUTH_HINTS), str(c), n))
        else:
            suggest.append((str(c), n))
    return sorted(auto), suggest


def find_truth(source, rebuilt, truth_col=None, truth_file=None):
    """Locate ground-truth labels, in order: an explicit file, a named column, then a likely column.

    The core rebuilds the query and keeps only its own QC columns, so a label column present in the
    user's data does not survive into query_<name>.h5ad. The original source is therefore searched
    first, and the rebuilt file only as a fallback.

    Returns (Series indexed by cell id, description) or (None, reason).
    """
    import anndata as ad, glob

    if truth_file:
        sep = "\t" if truth_file.lower().endswith((".tsv", ".txt")) else None
        df = pd.read_csv(truth_file, sep=sep, engine="python", dtype=str)
        idc, lab = _id_and_label(df, truth_col)
        if lab is None:
            return None, f"{os.path.basename(truth_file)} has no label column beside '{idc}'"
        s = pd.Series(df[lab].astype(str).values, index=df[idc].astype(str))
        s = s[~s.index.duplicated(keep="first")]
        return s, f"{os.path.basename(truth_file)} [{lab}]"

    cands = []
    if os.path.isfile(source) and source.endswith(".h5ad"):
        cands.append(source)
    elif os.path.isdir(source):
        cands += sorted(glob.glob(os.path.join(source, "*.h5ad")))
    if rebuilt and os.path.exists(rebuilt):
        cands.append(rebuilt)

    seen_cols, maybe = [], []
    for p in cands:
        try:
            a = ad.read_h5ad(p, backed="r")
        except Exception:
            continue
        if truth_col:
            if truth_col in a.obs.columns:
                s = pd.Series(a.obs[truth_col].astype(str).values, index=a.obs_names.astype(str))
                del a
                return s, f"{os.path.basename(p)} [{truth_col}]"
        else:
            auto, suggest = _obs_candidates(a)
            if auto:
                _, col, n = auto[0]
                s = pd.Series(a.obs[col].astype(str).values, index=a.obs_names.astype(str))
                others = ", ".join(c for _, c, _ in auto[1:4])
                del a
                note = f"{os.path.basename(p)} [{col}, {n} classes, auto-detected]"
                if others:
                    note += f"  (other candidates: {others})"
                return s, note
            maybe += [f"{c} ({n} classes)" for c, n in suggest]
            seen_cols += [str(c) for c in a.obs.columns]
        del a

    if truth_col:
        return None, f"column '{truth_col}' not found in the query data"
    msg = "no ground-truth column found"
    if maybe:
        msg += (f". These exist but do not look like cell-type annotations, so they were NOT used: "
                f"{', '.join(sorted(set(maybe))[:5])}. Set truth= if one of them is your ground truth")
    elif seen_cols:
        msg += f" (query obs has: {', '.join(sorted(set(seen_cols))[:8])})"
    return None, msg


def extend_for_truth(labels, hier_path, cache, key):
    """Place a truth vocabulary into the hierarchy's lineage/compartment scheme.

    A user's own labels are rarely the reference's labels: ours were Cancer a..i and Fibroblast a..d
    against a reference offering one epithelial type. Coarsening both sides through the same
    hierarchy is what makes them comparable, so unfamiliar truth labels have to be placed by the
    same LLM step the reference uses rather than by a hand-written map.

    Called once on the whole vocabulary. Extending lazily as labels appear would rebuild the map
    partway through and score some cells against a different mapping than others.
    """
    H = pd.read_csv(hier_path)
    if os.path.exists(cache):
        E = pd.read_csv(cache)
        if set(map(str, labels)) <= set(E.label.astype(str)):
            return E
        H = E
    missing = sorted(set(map(str, labels)) - set(H.label.astype(str)))
    if not missing:
        return H
    if not key:
        return H            # caller reports them as unmapped rather than guessing
    from openai import OpenAI
    prompt = ("Map each cell type to an immediate biological lineage and one compartment. "
              "Reuse the existing values verbatim where they fit.\n"
              + json.dumps({"cell_types": missing,
                            "existing_lineages": sorted(H.lineage.astype(str).unique()),
                            "compartments": sorted(H.compartment.astype(str).unique())})
              + '\nReturn JSON {"map":[{"label":..,"lineage":..,"compartment":..}]}')
    r = OpenAI(api_key=key).chat.completions.create(
        model="gpt-5", response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}])
    add = pd.DataFrame(json.loads(r.choices[0].message.content)["map"])[
        ["label", "lineage", "compartment"]].astype(str)

    # Constrain the answer to the vocabulary that already exists. Left unchecked the model will
    # sometimes invent a category ("Progenitors" came back once as a new lineage "Progenitor"), and
    # an invented lineage is one no prediction can ever equal, so every cell carrying it is scored
    # wrong for a reason that has nothing to do with the annotation. Compartments are a closed set of
    # four and are remapped to the nearest existing value; a genuinely new lineage is kept but
    # reported, because it means those cells are unreachable rather than misannotated.
    comps, lins = set(H.compartment.astype(str)), set(H.lineage.astype(str))
    bad_c = add[~add.compartment.isin(comps)]
    for _, x in bad_c.iterrows():
        near = [c for c in comps if c.lower() in x.compartment.lower()
                or x.compartment.lower() in c.lower()]
        fixed = near[0] if near else sorted(comps)[0]
        print(f"    note: '{x.label}' was given compartment '{x.compartment}', which is not one of "
              f"{sorted(comps)}; using '{fixed}'", flush=True)
        add.loc[add.label == x.label, "compartment"] = fixed
    new_lin = sorted(set(add.lineage) - lins)
    if new_lin:
        print(f"    note: new lineage(s) {new_lin} were introduced for the truth vocabulary. "
              f"No prediction can map to them, so cells carrying them are reported as unreachable "
              f"rather than counted as annotation errors.", flush=True)

    E = pd.concat([H, add], ignore_index=True)
    E.to_csv(cache, index=False)
    # An ambiguous name ("Progenitors" could be haematopoietic or epithelial) is not always placed
    # the same way twice, and the placement moves the accuracy it feeds. Print every placement and
    # say where the file is: the cache makes reruns stable, and the file is editable if a placement
    # is wrong for this tissue.
    print(f"    placed {len(add)} truth label(s) into the hierarchy "
          f"-> {os.path.basename(cache)} (edit it to override):", flush=True)
    for _, x in add.iterrows():
        print(f"      {str(x.label):<28}{str(x.lineage):<22}{x.compartment}", flush=True)
    return E


def score_truth(d, source, rebuilt, truth_col, hier_path, cache=None, key=None, truth_file=None):
    """Attach accuracy columns when the query carries its own labels.

    The truth vocabulary is usually not the reference vocabulary, so subtype comparison is not
    generally defined; the shared ground is the lineage and compartment the hierarchy assigns. Truth
    labels absent from the hierarchy are reported, not silently dropped.
    """
    truth, src = find_truth(source, rebuilt, truth_col, truth_file)
    if truth is None:
        return d, {"note": src}
    d = d.copy()
    d["truth"] = d.cell.astype(str).map(truth)
    if d.truth.isna().all():
        return d, {"note": f"found labels in {src} but their cell ids do not match the "
                           f"annotated cells, so nothing could be scored"}
    matched = float(d.truth.notna().mean())

    vocab = set(d.truth.dropna()) | set(d.predicted_subtype.astype(str))
    H = (extend_for_truth(vocab, hier_path, cache, key) if cache
         else pd.read_csv(hier_path))
    lin = dict(zip(H.label.astype(str), H.lineage.astype(str)))
    comp = dict(zip(H.label.astype(str), H.compartment.astype(str)))
    unmapped = sorted(set(d.truth.dropna()) - set(lin))
    out = {"truth from": src}
    if matched < 0.999:
        out["cells matched"] = f"{100*matched:.1f}%"
    if unmapped:
        out["unmapped truth labels"] = ", ".join(unmapped[:8]) + ("..." if len(unmapped) > 8 else "")
        # An unmapped label coarsens to itself, so it can never equal a prediction and every cell
        # carrying it scores wrong. Past a small fraction the accuracy stops measuring the
        # annotation and starts measuring the gap in the hierarchy, so withhold it rather than
        # print a number that looks real. The usual cause is no LLM key to place the vocabulary.
        share = float(d.truth.isin(unmapped).mean())
        if share > 0.10:
            out["accuracy withheld"] = (
                f"{100*share:.0f}% of cells carry a truth label with no place in the hierarchy, so "
                f"an accuracy would measure the mapping, not the annotation. Supply an LLM key so "
                f"the truth vocabulary can be placed, or edit hierarchy_truth_*.csv by hand.")
            return d, out
    for lev, m in (("lineage", lin), ("compartment", comp)):
        t = d.truth.map(m); p = d[f"predicted_{lev}"].astype(str).map(lambda v: m.get(v, v))
        ok = (t == p)
        ok[t.isna()] = np.nan
        d[f"acc_{lev}"] = ok.astype(float)
        d[f"truth_{lev}"] = t
        out[f"{lev} accuracy"] = float(100 * np.nanmean(ok.values.astype(float)))

    # A truth label whose lineage no prediction can produce is wrong by construction: the reference
    # has no such type to emit. That is a property of the reference vocabulary, not of the
    # annotation, so report it separately instead of letting it sit silently inside the headline.
    reach = set(d.predicted_lineage.astype(str).map(lambda v: lin.get(v, v)))
    unreach = ~d["truth_lineage"].isin(reach) & d["truth_lineage"].notna()
    if unreach.any():
        miss = sorted(set(d.loc[unreach, "truth_lineage"]))
        out["unreachable"] = (f"{100*unreach.mean():.2f}% of cells "
                              f"({', '.join(miss[:4])}{'...' if len(miss) > 4 else ''})")
        v = d["acc_lineage"].values.astype(float)
        out["lineage accuracy, reachable"] = float(100 * np.nanmean(v[~unreach.values]))

    cert = (d.resolution == "subtype").values
    v = d["acc_lineage"].values.astype(float)
    out["certified"] = float(100 * np.nanmean(v[cert]))
    out["withheld"] = float(100 * np.nanmean(v[~cert]))
    out["gap"] = out["certified"] - out["withheld"]
    return d, out


def run_one(r, outdir, rscript, key, python, fresh, no_umap, no_markers,
            chunk_override=None, no_export=False, want_rds=False):
    name = r["name"]
    d_out = os.path.join(outdir, name)
    os.makedirs(d_out, exist_ok=True)
    mlh = os.path.join(d_out, f"nacre_mlh_{name}.csv")

    if fresh:
        for f in os.listdir(d_out):
            if f.startswith(("bases_", "nacre_mlh_", "query_", "ref_")):
                try: os.remove(os.path.join(d_out, f))
                except OSError: pass

    rec = {"name": name, "platform": r["platform"], "status": "ok"}
    t0 = time.time()

    if os.path.exists(mlh) and not fresh:
        print(f"  [skip] {name} already has {os.path.basename(mlh)}", flush=True)
        rec["skipped"] = True
    else:
        cmd = [python, "-u", CLI, "--platform", r["platform"], "--query", r["query"],
               "--ref", r["reference"], "--out", d_out, "--name", name]
        if chunk_override: cmd += ["--chunk", str(int(chunk_override))]
        if r.get("ref_label"):  cmd += ["--ref-label", r["ref_label"]]
        if r.get("min_counts") is not None: cmd += ["--min-counts", str(r["min_counts"])]
        if r.get("min_umi") is not None:    cmd += ["--min-umi", str(r["min_umi"])]
        if rscript: cmd += ["--rscript", rscript]
        if key:     cmd += ["--llm-key", key]

        env = dict(os.environ)
        if key: env["OPENAI_API_KEY"] = key
        buf = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        for line in proc.stdout:
            sys.stdout.write("    " + line); sys.stdout.flush(); buf.append(line)
        proc.wait()
        text = "".join(buf)
        open(os.path.join(d_out, f"pipeline_{name}.log"), "w", encoding="utf-8").write(text)
        rec.update(parse_stdout(text))
        if proc.returncode != 0 or not os.path.exists(mlh):
            rec["status"] = f"failed (exit {proc.returncode})"
            rec["minutes"] = (time.time() - t0) / 60
            return rec

    if not rec.get("skipped"):
        rec["minutes"] = round((time.time() - t0) / 60, 1)

    d = pd.read_csv(mlh)
    rec["cells"] = rec.get("cells") or len(d)
    rec["res"] = {t: int((d.resolution == t).sum()) for t in TIERS}
    # Derive the vote structure from the table rather than from the core's stdout, so a resumed run
    # reports it as fully as a fresh one. With four annotators concordance is quantised to
    # 25/50/75/100, and LOCK/STRONG/SPLIT are exactly those buckets.
    if "conc_subtype" in d.columns:
        c = d.conc_subtype.values
        rec["lock"] = float(100 * np.mean(c >= 100 - 1e-9))
        rec["strong"] = float(100 * np.mean((c >= 75 - 1e-9) & (c < 100 - 1e-9)))
        rec["split"] = float(100 * np.mean(c < 75 - 1e-9))
    cpath = mlh.replace(".csv", "_concordance.csv")
    conc = pd.read_csv(cpath) if os.path.exists(cpath) else None
    rec["conc"] = conc

    q_h5 = os.path.join(d_out, f"query_{name}.h5ad")
    if os.path.exists(q_h5):
        try:
            import anndata as ad
            a = ad.read_h5ad(q_h5, backed="r")
            if "transcript_counts" in a.obs.columns:
                tx = pd.Series(np.asarray(a.obs["transcript_counts"]).ravel(),
                               index=a.obs_names.astype(str))
                d["transcript_counts"] = d.cell.astype(str).map(tx)
                rec["median_tx"] = float(np.nanmedian(d.transcript_counts))
            if "spatial" in a.obsm:
                xy = pd.DataFrame(np.asarray(a.obsm["spatial"])[:, :2],
                                  index=a.obs_names.astype(str), columns=["x", "y"])
                d[["x", "y"]] = xy.reindex(d.cell.astype(str)).values
            rec["genes"] = rec.get("genes") or int(a.n_vars)
            del a
        except Exception as e:
            print(f"    note: could not read {os.path.basename(q_h5)} for extras: {e}", flush=True)

    r_h5 = os.path.join(d_out, f"ref_{name}.h5ad")
    if not rec.get("ref_cells") and os.path.exists(r_h5):
        try:
            import anndata as ad
            a = ad.read_h5ad(r_h5, backed="r")
            rec["ref_cells"] = int(a.n_obs)
            if "major_annotation" in a.obs.columns:
                rec["ref_types"] = int(a.obs["major_annotation"].astype(str).nunique())
            ref_genes = set(a.var_names.astype(str))
            del a
            # The reference keeps its full transcriptome, so its gene count is not the overlap.
            # Intersect against the panel to get the number the annotators actually work from.
            if os.path.exists(q_h5):
                qa = ad.read_h5ad(q_h5, backed="r")
                panel = set(qa.var_names.astype(str)); del qa
                rec["shared"] = f"{len(panel & ref_genes)}/{len(panel)}"
        except Exception:
            pass

    # Ground truth is optional and may arrive three ways: an explicit file, a named column, or a
    # likely column found in the query. Accuracy is reported when labels are found and quietly
    # omitted when they are not, so a dataset without ground truth runs exactly as well.
    hp = os.path.join(d_out, "hierarchy_extended.csv")
    hp = hp if os.path.exists(hp) else os.path.join(HERE, "assets", "hierarchy.csv")
    try:
        d, acc = score_truth(d, r["query"], q_h5, r.get("truth"), hp,
                             cache=os.path.join(d_out, f"hierarchy_truth_{name}.csv"), key=key,
                             truth_file=r.get("truth_file"))
        if "lineage accuracy" in acc:
            rec["acc"] = acc
            print("    ground truth: " +
                  "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in acc.items()), flush=True)
        else:
            # Say why. "none" on its own leaves the user unable to tell an absent label column from
            # a vocabulary that could not be placed, which have completely different remedies.
            why = acc.get("note") or acc.get("accuracy withheld") or "none found"
            print(f"    ground truth: accuracy not reported -- {why}", flush=True)
            for k in ("truth from", "unmapped truth labels"):
                if k in acc:
                    print(f"      {k}: {acc[k]}", flush=True)
    except Exception as e:
        print(f"    note: ground-truth scoring failed: {e}; accuracy not reported", flush=True)

    # Write the annotation back onto the counts, so the result is an object rather than a CSV to
    # re-join. Done after truth scoring so the accuracy columns travel with it.
    if not no_export:
        try:
            ah5 = os.path.join(d_out, f"{name}_annotated.h5ad")
            p, added = expmod.annotated_h5ad(d, q_h5, ah5, name)
            print(f"    wrote {os.path.basename(p)} ({len(added)} annotation columns)", flush=True)
            rec["annotated_h5ad"] = p
            if want_rds:
                rp, err = expmod.annotated_rds(p, os.path.join(d_out, f"{name}_annotated.rds"),
                                               rscript)
                if rp:
                    print(f"    wrote {os.path.basename(rp)}", flush=True)
                    rec["annotated_rds"] = rp
                else:
                    print(f"    note: .rds not written: {err}", flush=True)
        except Exception as e:
            print(f"    note: annotated object not written: {e}", flush=True)

    print("    drawing figures", flush=True)
    rec["figures"] = figmod.draw_all(d, conc, q_h5, os.path.join(d_out, "figures"), name,
                                     want_umap=not no_umap, want_markers=not no_markers)
    for k, v in rec["figures"].items():
        if not str(v).lower().endswith(".png"):
            print(f"      {k}: {v}", flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(
        description="NACRE-MLH pipeline for Xenium / MERSCOPE / CosMx.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", "-c", default="runs.csv", help="run sheet (CSV, TSV or text)")
    p.add_argument("--out", "-o", default=None, help="output root (default: ./results)")
    p.add_argument("--llm-key", default=None, help="LLM API key; else $OPENAI_API_KEY or key.txt")
    p.add_argument("--rscript", default=None, help="path to Rscript.exe if not on PATH")
    p.add_argument("--python", default=sys.executable, help="python to run the core with")
    p.add_argument("--only", default=None, help="comma-separated run names to process")
    p.add_argument("--fresh", action="store_true", help="rebuild runs instead of resuming")
    p.add_argument("--chunk", type=int, default=None,
                   help="cells per annotator pass; omit to size it from free memory automatically")
    p.add_argument("--rds", action="store_true",
                   help="also write <name>_annotated.rds for Seurat (needs R; slower)")
    p.add_argument("--no-export", action="store_true", help="do not write the annotated .h5ad")
    p.add_argument("--no-umap", action="store_true", help="skip the UMAP panel (the slow figure)")
    p.add_argument("--no-markers", action="store_true", help="skip the marker dotplot")
    p.add_argument("--figures-only", action="store_true",
                   help="redraw figures and summaries from finished runs, annotators not rerun")
    p.add_argument("--init", action="store_true", help="write a template run sheet and exit")
    a = p.parse_args()

    if a.init:
        if os.path.exists(a.config):
            raise SystemExit(f"{a.config} already exists; delete it or choose another --config")
        open(a.config, "w", encoding="utf-8").write(cfgmod.TEMPLATE)
        print(f"wrote {a.config}\nEdit it, then: python run_pipeline.py --config {a.config}")
        return

    try:
        runs = cfgmod.load(a.config)
    except cfgmod.ConfigError as e:
        print(f"\nRun sheet problem:\n  {e}\n\nStart from a template with:  "
              f"python run_pipeline.py --init", file=sys.stderr)
        raise SystemExit(2)

    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        missing = want - {r["name"] for r in runs}
        if missing:
            raise SystemExit(f"--only names not in the run sheet: {', '.join(sorted(missing))}")
        runs = [r for r in runs if r["name"] in want]

    outdir = os.path.abspath(a.out or os.path.join(HERE, "results"))
    os.makedirs(outdir, exist_ok=True)
    rscript = find_rscript(a.rscript)
    key = read_key(a.llm_key)

    print(f"\n=== NACRE-MLH pipeline ===")
    print(f"run sheet : {os.path.abspath(a.config)}  ({len(runs)} run(s))")
    print(f"output    : {outdir}")
    print(f"Rscript   : {rscript or 'NOT FOUND'}")
    print(f"LLM key   : {'supplied' if key else 'none'}\n")
    for r in runs:
        print(f"  {r['name']:<22}{summod.PLATFORM_NAME.get(r['platform'], r['platform']):<10}"
              f"{r['query']}")
    print()
    preflight(runs, rscript, key)

    if a.figures_only:
        print("--figures-only: annotators will not be run\n")

    results, t0 = [], time.time()
    for i, r in enumerate(runs, 1):
        print(f"\n--- [{i}/{len(runs)}] {r['name']} ---", flush=True)
        try:
            if a.figures_only and not os.path.exists(
                    os.path.join(outdir, r["name"], f"nacre_mlh_{r['name']}.csv")):
                print("  [skip] no finished run to draw from", flush=True)
                continue
            rec = run_one(r, outdir, rscript, key, a.python,
                          a.fresh and not a.figures_only, a.no_umap, a.no_markers,
                          chunk_override=a.chunk, no_export=a.no_export, want_rds=a.rds)
        except KeyboardInterrupt:
            print("\ninterrupted; finished runs are kept and will be resumed next time")
            break
        except Exception:
            traceback.print_exc()
            rec = {"name": r["name"], "platform": r["platform"], "status": "failed (pipeline)"}
        results.append(rec)
        print(f"  {rec['name']}: {rec['status']}", flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        print("\n--- summary ---", flush=True)
        T = summod.build_tables(ok)
        summod.write_csvs(T, outdir)
        md = summod.write_markdown(T, outdir, "NACRE-MLH results")
        figs = {r["name"]: r.get("figures", {}) for r in ok}
        dx = summod.write_docx(T, outdir, "NACRE-MLH results", figures=figs)
        print(f"  {md}")
        print(f"  {dx}" if dx else
              "  (python-docx not installed, so no .docx; CSVs and SUMMARY.md are written)")
        for k in T: print(f"  summary_{k}.csv")

    print(f"\ndone in {(time.time()-t0)/60:.1f} min: "
          f"{len(ok)} ok, {len(results)-len(ok)} failed, {len(runs)-len(results)} not reached")
    for r in results:
        if r.get("status") != "ok":
            print(f"  {r['name']}: {r['status']}  (see {os.path.join(outdir, r['name'])})")
    return 0 if len(ok) == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
