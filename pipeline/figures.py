"""Per-run figures, drawn from whatever the run actually produced.

Adapted from the manuscript's figure set. The difference is that nothing here knows which dataset it
is looking at: labels, palettes and groupings are read off the outputs, so a new tissue with a new
reference vocabulary draws correctly without edits. Figures whose inputs are absent are skipped
rather than faked, and the caller is told which.

  1 fig_certificate   how far down the certificate reaches, and where the annotators converge
  2 fig_concordance   agreement before and after the resolver, at all three levels
  3 fig_spatial       the section, coloured by label and by certificate tier
  4 fig_umap          one panel per annotator plus NACRE at each level
  5 fig_markers       marker dotplot per label at each level
  6 fig_depth         transcript counts against certificate tier
  7 fig_accuracy      only when the run sheet named a truth column
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOOLS = ["Seurat", "Azimuth", "RCTD", "SingleR"]
LEVELS = ["subtype", "lineage", "compartment"]
TIERS = LEVELS + ["unresolved"]
TIER_COL = {"subtype": "#238b45", "lineage": "#a1d99b",
            "compartment": "#fdae6b", "unresolved": "#cb181d"}
GREY = (0.86, 0.86, 0.88)


def palette(labels):
    """Deterministic colour per label: sorted, so a type keeps its colour across every panel."""
    base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    return {t: base[i % len(base)] for i, t in enumerate(sorted(set(map(str, labels))))}


def _scatter(ax, xy, labels, pal, title, s=0.6, spatial=False):
    rng = np.random.default_rng(0)
    order = rng.permutation(len(xy))          # else the last-drawn type hides the rest
    cols = np.array([pal.get(str(l), GREY) for l in labels])
    ax.scatter(xy[order, 0], xy[order, 1], c=cols[order], s=s, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if spatial:
        ax.set_aspect("equal"); ax.invert_yaxis()


def _legend_below(fig, pal, keys, ncol=9, size=7.5):
    h = [plt.Line2D([0], [0], marker="o", ls="", mfc=pal[k], mec="none", ms=7, label=k) for k in keys]
    fig.legend(handles=h, loc="lower center", ncol=min(ncol, len(keys)), fontsize=size,
               frameon=False, bbox_to_anchor=(0.5, -0.015))


def _save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return os.path.basename(path)


def _lognorm(X):
    import scipy.sparse as sps
    X = X.astype(np.float32)
    if sps.issparse(X):
        s = np.asarray(X.sum(1)).ravel(); s[s == 0] = 1
        X = X.multiply(1e4 / s[:, None]).tocsr(); X.data = np.log1p(X.data); return X
    s = X.sum(1, keepdims=True); s[s == 0] = 1
    return np.log1p(X / s * 1e4)


# --------------------------------------------------------------------------- figures
def fig_certificate(d, out, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    vc = d.resolution.value_counts()
    n = len(d)
    ax = axes[0]
    vals = [vc.get(t, 0) for t in TIERS]
    ax.bar(range(4), [100 * v / n for v in vals], color=[TIER_COL[t] for t in TIERS], width=.62)
    for i, v in enumerate(vals):
        ax.text(i, 100 * v / n + 1.2, f"{100*v/n:.1f}%\n{v:,}", ha="center", fontsize=8.4)
    ax.set_xticks(range(4)); ax.set_xticklabels(["subtype", "lineage", "compart.", "unresolved"])
    ax.set_ylabel("% of cells"); ax.set_ylim(0, max(100 * max(vals) / n + 12, 20))
    ax.set_title("Deepest level the certificate reaches", fontsize=10.5, fontweight="bold")

    ax = axes[1]
    for lev in LEVELS:
        c = f"conc_{lev}"
        if c not in d: continue
        v = d[c].round(2).value_counts(normalize=True).mul(100).sort_index()
        ax.plot(v.index, v.values, marker="o", label=lev, lw=1.8)
    ax.axvline(75, color="0.4", ls="--", lw=1, zorder=0)
    ax.text(75.6, ax.get_ylim()[1] * .92, "threshold", fontsize=7.6, color="0.4")
    ax.set_xlabel("annotator concordance (%)"); ax.set_ylabel("% of cells")
    ax.set_title("Where the four annotators converge", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle(f"{name}  |  n = {n:,} cells", fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, os.path.join(out, f"fig_certificate_{name}.png"))


def fig_concordance(conc, out, name):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(conc)); w = .36
    ax.bar(x - w/2, conc.before, w, label="four annotators alone", color="#bdbdbd")
    ax.bar(x + w/2, conc.after,  w, label="after NACRE", color="#3d7ea6")
    for i, r in conc.reset_index(drop=True).iterrows():
        ax.text(i + w/2, r.after + .6, f"+{r.delta:.2f}", ha="center", fontsize=8.4,
                color="#1b4f6b", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(conc.level)
    ax.set_ylabel("mean pairwise concordance (%)")
    ax.set_ylim(0, min(100, conc.after.max() + 12))
    ax.set_title(f"Concordance before and after the resolver  |  {name}",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _save(fig, os.path.join(out, f"fig_concordance_{name}.png"))


def fig_spatial(d, out, name):
    if not {"x", "y"} <= set(d.columns):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    xy = d[["x", "y"]].values
    keys = sorted(set(d.usable_label.astype(str)))
    pal = palette(keys)
    _scatter(axes[0], xy, d.usable_label.astype(str), pal,
             "Label, at the depth the certificate allows", s=1.1, spatial=True)
    _scatter(axes[1], xy, d.resolution.astype(str), TIER_COL,
             "Certificate tier", s=1.1, spatial=True)
    h = [plt.Line2D([0], [0], marker="o", ls="", mfc=TIER_COL[t], mec="none", ms=7, label=t)
         for t in TIERS]
    axes[1].legend(handles=h, fontsize=8, frameon=False, loc="upper right", markerscale=1.1)
    _legend_below(fig, pal, keys, ncol=7, size=7)
    fig.suptitle(f"{name}  |  spatial distribution of certified and withheld calls",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return _save(fig, os.path.join(out, f"fig_spatial_{name}.png"))


def fig_umap(d, q_h5, out, name, cap=40000):
    """Shared embedding, one panel per annotator plus NACRE at each level.

    The annotators scatter across the contested regions and the NACRE panels tighten as the level
    coarsens; that contrast is the point of the panel, so all of them share one embedding.
    """
    import anndata as ad
    from sklearn.decomposition import TruncatedSVD
    a = ad.read_h5ad(q_h5, backed="r")
    obs = a.obs_names.astype(str)
    pos = pd.Series(np.arange(len(obs)), index=obs)
    idx = pos.reindex(d.cell.astype(str)).dropna().astype(int)
    if len(idx) < 50:
        return None
    take = np.sort(np.random.default_rng(0).choice(len(idx), min(cap, len(idx)), replace=False))
    sub = d.iloc[take].reset_index(drop=True)
    X = _lognorm(a[idx.values[take]].to_memory().X)
    del a
    Z = TruncatedSVD(n_components=min(30, X.shape[1] - 1), random_state=0).fit_transform(X)
    try:
        import umap
        E = umap.UMAP(random_state=0, n_neighbors=15, min_dist=.3).fit_transform(Z)
    except Exception:
        E = Z[:, :2]          # SVD projection is a fair fallback; no extra dependency required

    cols = [(t, t) for t in TOOLS if t in sub] + \
           [(f"predicted_{l}", f"NACRE  {l}") for l in LEVELS if f"predicted_{l}" in sub]
    ncol = 4; nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    pal = palette(pd.concat([sub[c].astype(str) for c, _ in cols]))
    for ax, (c, title) in zip(axes, cols):
        _scatter(ax, E, sub[c].astype(str), pal, title, s=1.4)
    for ax in axes[len(cols):]:
        ax.axis("off")
    keys = sorted({k for c, _ in cols for k in sub[c].astype(str).unique()})
    _legend_below(fig, pal, keys, ncol=7, size=6.8)
    fig.suptitle(f"{name}  |  {len(sub):,} cells", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return _save(fig, os.path.join(out, f"fig_umap_{name}.png"))


def fig_markers(d, q_h5, out, name, topn=4, cap=40000):
    import anndata as ad
    a = ad.read_h5ad(q_h5, backed="r")
    obs = a.obs_names.astype(str)
    pos = pd.Series(np.arange(len(obs)), index=obs)
    idx = pos.reindex(d.cell.astype(str)).dropna().astype(int)
    if len(idx) < 50:
        return None
    take = np.sort(np.random.default_rng(0).choice(len(idx), min(cap, len(idx)), replace=False))
    sub = d.iloc[take].reset_index(drop=True)
    genes = np.array(a.var_names.astype(str))
    X = _lognorm(a[idx.values[take]].to_memory().X)
    del a
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, lev in zip(axes, LEVELS):
        col = f"predicted_{lev}"
        if col not in sub:
            ax.axis("off"); continue
        labs = sorted(set(sub[col].astype(str)))
        M = np.vstack([X[(sub[col].astype(str) == l).values].mean(0) for l in labs])
        z = (M - M.mean(0)) / (M.std(0) + 1e-9)
        pick, seen = [], set()
        for i in range(len(labs)):                       # top markers per label, no repeats
            for g in np.argsort(-z[i])[:topn * 3]:
                if g not in seen:
                    pick.append(g); seen.add(g)
                    if sum(1 for _ in pick) % topn == 0: break
        pick = pick[:topn * len(labs)]
        frac = np.vstack([(X[(sub[col].astype(str) == l).values][:, pick] > 0).mean(0) for l in labs])
        zz = z[:, pick]
        yy, xx = np.meshgrid(np.arange(len(labs)), np.arange(len(pick)), indexing="ij")
        sc = ax.scatter(xx, yy, s=frac * 90, c=zz, cmap="RdBu_r", vmin=-2, vmax=2,
                        edgecolors="none")
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7)
        ax.set_xticks(range(len(pick)))
        ax.set_xticklabels(genes[pick], rotation=90, fontsize=5.6)
        ax.set_title(lev, fontsize=10.5, fontweight="bold")
        ax.invert_yaxis()
    fig.colorbar(sc, ax=axes, fraction=.012, pad=.01, label="mean expression (z)")
    fig.suptitle(f"{name}  |  markers per label at each level  (dot size = % expressing)",
                 fontsize=11.5, fontweight="bold")
    return _save(fig, os.path.join(out, f"fig_markers_{name}.png"))


def fig_depth(d, out, name):
    if "transcript_counts" not in d.columns or d.transcript_counts.isna().all():
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    data = [d.transcript_counts[d.resolution == t].dropna().values for t in TIERS]
    keep = [(t, v) for t, v in zip(TIERS, data) if len(v)]
    bp = ax.boxplot([v for _, v in keep], labels=[t for t, _ in keep], showfliers=False,
                    patch_artist=True, widths=.55, medianprops=dict(color="black"))
    for patch, (t, _) in zip(bp["boxes"], keep):
        patch.set_facecolor(TIER_COL[t]); patch.set_alpha(.85); patch.set_edgecolor("white")
    ax.set_yscale("log"); ax.set_ylabel("transcripts per cell")
    med = {t: float(np.median(v)) for t, v in keep}
    if "subtype" in med and "unresolved" in med and med["subtype"] > 0:
        drop = 100 * (1 - med["unresolved"] / med["subtype"])
        ax.set_title(f"Unresolved cells carry {drop:.0f}% fewer transcripts  |  {name}",
                     fontsize=10.5, fontweight="bold")
    else:
        ax.set_title(f"Transcript depth by certificate tier  |  {name}",
                     fontsize=10.5, fontweight="bold")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _save(fig, os.path.join(out, f"fig_depth_{name}.png"))


def fig_accuracy(d, out, name):
    """Only drawn when the run sheet named a truth column, so `acc_*` columns exist."""
    if "acc_lineage" not in d.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    xs, ys, ns = [], [], []
    for t in TIERS:
        m = (d.resolution == t).values
        if m.sum() < 10: continue
        xs.append(t); ys.append(100 * d.acc_lineage[m].mean()); ns.append(int(m.sum()))
    ax.bar(range(len(xs)), ys, color=[TIER_COL[t] for t in xs], width=.62)
    for i, (v, n) in enumerate(zip(ys, ns)):
        ax.text(i, v + 1.2, f"{v:.1f}%\nn={n:,}", ha="center", fontsize=8.2)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
    ax.set_ylabel("lineage accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("Accuracy by certificate tier", fontsize=10.5, fontweight="bold")

    ax = axes[1]
    cert = (d.resolution == "subtype").values
    a, b = 100 * d.acc_lineage[cert].mean(), 100 * d.acc_lineage[~cert].mean()
    ax.bar([0, 1], [a, b], color=["#238b45", "#cb181d"], width=.55)
    for i, (v, n) in enumerate(zip([a, b], [cert.sum(), (~cert).sum()])):
        ax.text(i, v + 1.2, f"{v:.2f}%\nn={n:,}", ha="center", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["certified at subtype", "withheld"])
    ax.set_ylim(0, 108); ax.set_ylabel("lineage accuracy (%)")
    ax.set_title(f"Certified vs withheld   gap {a-b:+.2f} points",
                 fontsize=10.5, fontweight="bold")
    for x in axes:
        for s in ("top", "right"): x.spines[s].set_visible(False)
    fig.suptitle(f"{name}  |  scored against the supplied truth column", fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, os.path.join(out, f"fig_accuracy_{name}.png"))


def draw_all(d, conc, q_h5, out, name, want_umap=True, want_markers=True):
    """Draw everything the run supports. Returns {figure: filename or reason skipped}."""
    os.makedirs(out, exist_ok=True)
    made = {}
    jobs = [("certificate", lambda: fig_certificate(d, out, name)),
            ("concordance", lambda: fig_concordance(conc, out, name) if conc is not None else None),
            ("spatial",     lambda: fig_spatial(d, out, name)),
            ("depth",       lambda: fig_depth(d, out, name)),
            ("accuracy",    lambda: fig_accuracy(d, out, name))]
    if want_umap:
        jobs.append(("umap", lambda: fig_umap(d, q_h5, out, name)))
    if want_markers:
        jobs.append(("markers", lambda: fig_markers(d, q_h5, out, name)))
    for key, fn in jobs:
        try:
            r = fn()
            made[key] = r if r else "skipped (inputs absent)"
        except Exception as e:
            # one figure failing must not cost the run its other figures or its tables
            made[key] = f"failed: {type(e).__name__}: {e}"
    return made
