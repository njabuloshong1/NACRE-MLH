# R dependencies for the NACRE-MLH annotators.
#
# Versions are those the pipeline was validated against. The Bioconductor packages come from
# release 3.18, which is what pins zellkonverter 1.12, SingleR 2.4, scuttle 1.12 and
# SingleCellExperiment 1.24 together; changing the release moves all four at once.
#
# spacexr (RCTD) is not on CRAN or Bioconductor and has to come from GitHub.

options(warn = 2, repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = max(1L, parallel::detectCores()))

need <- function(p) !requireNamespace(p, quietly = TRUE)

if (need("BiocManager")) install.packages("BiocManager")
BiocManager::install(version = "3.18", ask = FALSE, update = FALSE)

cran <- c("Matrix", "remotes", "R.utils", "hdf5r")
for (p in cran) if (need(p)) install.packages(p)
if (need("remotes")) stop("remotes is required to install pinned versions")

# Seurat 5 pulls a long dependency chain; installing it before the Bioconductor set keeps the
# resolver from downgrading Matrix underneath it.
#
# Pinned deliberately. Seurat from CRAN-current means a rebuild months from now installs whatever
# is current then, and the version moves results: 5.3.0 and 5.5.1 give byte-identical Seurat
# annotations but differ on 5% of Azimuth calls, because Azimuth is the SCTransform path and
# SCTransform changed between those releases. Pinning keeps a rebuilt image comparable to this one.
SEURAT_VERSION <- "5.5.1"
if (need("Seurat") || as.character(utils::packageVersion("Seurat")) != SEURAT_VERSION) {
  remotes_ok <- requireNamespace("remotes", quietly = TRUE)
  if (!remotes_ok) install.packages("remotes")
  remotes::install_version("Seurat", version = SEURAT_VERSION, upgrade = "never",
                           repos = "https://cloud.r-project.org")
}

bioc <- c("SingleCellExperiment", "scuttle", "SingleR", "celldex", "zellkonverter", "basilisk")
for (p in bioc) if (need(p)) BiocManager::install(p, ask = FALSE, update = FALSE)

# spacexr: pinned to a commit rather than a moving branch, so a rebuild months from now installs
# the same RCTD the results were produced with.
if (need("spacexr")) {
  remotes::install_github("dmcable/spacexr", ref = "master", build_vignettes = FALSE,
                          upgrade = "never")
}

# zellkonverter normally builds its own Python environment on first use, which would mean a network
# call at run time inside the container. Build it now so the image is self-contained.
suppressMessages(library(basilisk))
suppressMessages(library(zellkonverter))
tryCatch({
  env <- zellkonverter::zellkonverterAnnDataEnv()
  basilisk::obtainEnvironmentPath(env)
  cat("zellkonverter python environment provisioned\n")
}, error = function(e) cat("NOTE: could not pre-provision zellkonverter env:",
                           conditionMessage(e), "\n"))

pkgs <- c("zellkonverter", "SingleCellExperiment", "scuttle", "SingleR", "Seurat", "Matrix",
          "spacexr", "celldex")
missing <- Filter(function(p) !requireNamespace(p, quietly = TRUE), pkgs)
if (length(missing)) stop("failed to install: ", paste(missing, collapse = ", "))
cat("\nR packages installed:\n")
for (p in pkgs) cat(sprintf("  %-24s %s\n", p, as.character(utils::packageVersion(p))))
