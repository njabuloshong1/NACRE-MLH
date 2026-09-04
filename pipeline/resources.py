"""Memory sizing and environment inspection.

Two jobs:
  * work out how many query cells the annotators can hold at once, so a large section is split
    before it fails rather than after;
  * report what is already installed, with versions, so nobody is told to install something they
    have.
"""
import os, sys, re, subprocess, importlib

# The annotators the R script needs, with the minimum version where one actually matters.
R_REQUIRED = {"zellkonverter": None, "SingleCellExperiment": None, "scuttle": None,
              "SingleR": None, "Seurat": "4.0.0", "Matrix": None, "spacexr": None}
PY_REQUIRED = {"anndata": "0.8", "scipy": "1.7", "sklearn": "1.0",
               "matplotlib": "3.4", "pandas": "1.3", "numpy": "1.20",
               # the learned resolver; absent, a run does an hour of annotation and then dies
               # at the final step, so it belongs in the required set and not the optional one
               "torch": "1.13"}
PY_OPTIONAL = {"umap": "UMAP embedding (falls back to an SVD projection)",
               "docx": "the .docx report (CSVs and SUMMARY.md are written regardless)",
               "openai": "extending the hierarchy for unfamiliar cell-type names",
               "psutil": "accurate free-memory reading (a conservative default is used without it)"}


def _cgroup_limit():
    """Memory available to this container, or None outside one.

    psutil reports the HOST's memory inside a container, so a 4 GB container on a 64 GB machine
    would size its chunks for 64 GB and be killed. cgroup v2 first, then v1.
    """
    for lim_p, use_p in (("/sys/fs/cgroup/memory.max",                        # cgroup v2
                          "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes",      # cgroup v1
                          "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            raw = open(lim_p).read().strip()
            if raw in ("max", ""):
                continue
            lim = int(raw)
            if lim <= 0 or lim > (1 << 62):      # v1 writes a sentinel meaning "unlimited"
                continue
            try:
                use = int(open(use_p).read().strip())
            except Exception:
                use = 0
            return max(lim - use, 0)
        except Exception:
            continue
    return None


def available_bytes():
    """Free memory. A container limit wins over the host figure, then psutil, then the OS."""
    c = _cgroup_limit()
    if c is not None:
        return c, "container limit (cgroup)"
    try:
        import psutil
        return int(psutil.virtual_memory().available), "psutil"
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return int(m.ullAvailPhys), "GlobalMemoryStatusEx"
        except Exception:
            pass
    try:
        import resource  # noqa: F401  (POSIX)
        pages = os.sysconf("SC_AVPHYS_PAGES"); size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * size), "sysconf"
    except Exception:
        return 4 * 1024**3, "assumed 4 GB"


# Peak working set is dominated by the dense intermediates Seurat, SCTransform and RCTD build over
# the shared panel genes, not by the sparse counts. Measured across the Xenium, CosMx and MERSCOPE
# runs, roughly this many bytes per query cell per shared gene, including transient copies.
BYTES_PER_CELL_PER_GENE = 96
SAFETY = 0.55            # leave the OS and the reference-side objects room
MIN_CHUNK, MAX_CHUNK = 2000, 200000


def plan_chunk(n_cells, n_genes, avail=None, floor_gb=2.0):
    """Cells per pass. Returns (chunk, note). chunk >= n_cells means a single unchunked pass."""
    if avail is None:
        avail, _ = available_bytes()
    budget = max(avail - int(floor_gb * 1024**3), int(0.25 * avail)) * SAFETY
    per_cell = max(1, n_genes * BYTES_PER_CELL_PER_GENE)
    chunk = int(budget // per_cell)
    chunk = max(MIN_CHUNK, min(MAX_CHUNK, chunk))
    if chunk >= n_cells:
        return n_cells, (f"{n_cells:,} cells x {n_genes:,} genes fits in one pass "
                         f"(~{n_cells*per_cell/1024**3:.1f} GB of ~{avail/1024**3:.1f} GB free)")
    return chunk, (f"{n_cells:,} cells x {n_genes:,} genes needs ~"
                   f"{n_cells*per_cell/1024**3:.1f} GB but only ~{avail/1024**3:.1f} GB is free; "
                   f"splitting into chunks of {chunk:,}")


def _ver(mod):
    for a in ("__version__", "version", "VERSION"):
        v = getattr(mod, a, None)
        if isinstance(v, str):
            return v
    return "?"


def _cmp(have, need):
    """True when `have` >= `need`, comparing numeric parts only."""
    if not need or have == "?":
        return True
    def parts(s):
        return [int(x) for x in re.findall(r"\d+", s)[:3]]
    h, n = parts(have), parts(need)
    h += [0] * (len(n) - len(h)); n += [0] * (len(h) - len(n))
    return h >= n


def check_python():
    """-> (found, missing, outdated) with versions; nothing is installed or modified."""
    found, missing, outdated = {}, [], []
    for name, need in PY_REQUIRED.items():
        try:
            m = importlib.import_module(name)
        except Exception:
            missing.append(name); continue
        v = _ver(m); found[name] = v
        if not _cmp(v, need):
            outdated.append((name, v, need))
    for name, why in PY_OPTIONAL.items():
        try:
            found[name] = _ver(importlib.import_module(name))
        except Exception:
            found[name] = None
    return found, missing, outdated


def check_r(rscript):
    """-> (found {pkg: version}, missing [pkg]) for one R interpreter. Never installs anything."""
    pkgs = list(R_REQUIRED)
    expr = ('for (p in c(%s)) { v <- tryCatch(as.character(utils::packageVersion(p)), '
            'error=function(e) NA); cat(p, "\\t", ifelse(is.na(v), "MISSING", v), "\\n", sep="") }'
            % ",".join(f'"{p}"' for p in pkgs))
    try:
        r = subprocess.run([rscript, "--vanilla", "-e", expr], capture_output=True, text=True,
                           timeout=240)
    except Exception:
        return None, pkgs
    found, missing = {}, []
    for line in (r.stdout or "").splitlines():
        if "\t" not in line:
            continue
        p, v = line.split("\t", 1)
        p, v = p.strip(), v.strip()
        if p not in R_REQUIRED:
            continue
        if v == "MISSING":
            missing.append(p)
        else:
            found[p] = v
    if not found and not missing:
        return None, pkgs                       # the probe itself did not run
    return found, missing


def r_outdated(found):
    return [(p, v, R_REQUIRED[p]) for p, v in (found or {}).items()
            if R_REQUIRED.get(p) and not _cmp(v, R_REQUIRED[p])]


def gpu_status():
    """One line on whether the resolver will use a GPU. Reported because the commonest cause of a
    silent CPU fallback in a container is forgetting `--gpus all`, not a missing driver."""
    try:
        import torch
    except Exception:
        return "torch not installed"
    if torch.cuda.is_available():
        try:
            n = torch.cuda.get_device_name(0)
            gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{n} ({gb:.1f} GB), CUDA {torch.version.cuda}"
        except Exception:
            return f"available, CUDA {torch.version.cuda}"
    built = getattr(torch.version, "cuda", None)
    if built:
        return (f"none visible (torch built for CUDA {built}); the resolver will use the CPU. "
                f"In Docker, pass --gpus all")
    return "none (CPU-only torch build); the resolver will use the CPU"


def describe(found_py, r_path, found_r):
    out = [f"python  {sys.version.split()[0]}  ({sys.executable})"]
    req = "  ".join(f"{k} {found_py[k]}" for k in PY_REQUIRED if k in found_py)
    out.append(f"  required : {req}")
    opt = "  ".join(f"{k} {v}" if v else f"{k} -" for k, v in found_py.items()
                    if k in PY_OPTIONAL)
    out.append(f"  optional : {opt}")
    if r_path:
        out.append(f"R       {r_path}")
        out.append("  packages : " + ("  ".join(f"{k} {v}" for k, v in (found_r or {}).items())
                                      or "none found"))
    out.append(f"GPU     {gpu_status()}")
    return "\n".join(out)
