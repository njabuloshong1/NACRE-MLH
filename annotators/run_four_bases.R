# The four NACRE base annotators (Seurat, Azimuth, RCTD, SingleR) on one query/reference pair,
# emitting exactly the columns nacre_ml.py expects: cell,Seurat,Azimuth,RCTD,SingleR.
#
# Implementations are copied verbatim from the existing scripts so results stay comparable:
#   Seurat + Azimuth  <- nacre_bases.R  (LogNormalize anchors / SCTransform anchors)
#   SingleR + RCTD    <- run_all_baselines.R (Xenium-appropriate RCTD params)
# scmap is deliberately not run: NACRE's panel is these four.
#
# Usage: Rscript run_four_bases.R <query.h5ad> <ref.h5ad> <pred_out.csv> <runtime_out.csv> <label>
#                                 [min_umi] [chunk]
args <- commandArgs(trailingOnly = TRUE)
QUERY <- args[1]; REF <- args[2]; PRED_OUT <- args[3]; RT_OUT <- args[4]; DS <- args[5]
# min_UMI is supplied by the caller: RCTD applies it over shared panel genes only, so the right
# value depends on panel width. Never leave it at the package default of 100.
MIN_UMI <- if (length(args) >= 6) as.integer(args[6]) else 10L
# Maximum query cells to hold in one pass. The caller sizes this from free memory and the panel
# width; 0 means no limit. A section smaller than the chunk runs in exactly one pass, which is
# byte-identical to the unchunked code, so the common case is unaffected.
CHUNK <- if (length(args) >= 7) as.integer(args[7]) else 0L
suppressMessages({
  library(zellkonverter); library(SingleCellExperiment); library(scuttle)
  library(SingleR); library(Seurat); library(Matrix); library(spacexr)
})
set.seed(42)
options(future.globals.maxSize = 8 * 1024^3)

# Split the query and run `fn` over each piece, halving the chunk and retrying whenever a piece dies.
# An out-of-memory failure in R does not present as one recognisable condition (it can be a malloc
# error, a long-vector limit, or a killed worker), so rather than pattern-match the message we treat
# any chunk failure as a reason to try smaller. Only when a single chunk is at the floor and still
# fails do we give up and let the caller see the tool as failed.
#
# NOTE ON FIDELITY: only SingleR is per-cell independent, so only SingleR is unaffected by where the
# chunk boundaries fall. Seurat and Azimuth find anchors against the query set, and RCTD estimates
# platform effects across the puck, so splitting them changes results slightly. That is why the
# caller sizes chunks to be as large as memory allows instead of chunking by default.
FLOOR <- 2000L
run_chunked <- function(tool, fn, size) {
  if (size <= 0L || size >= ncell) {
    return(fn(seq_len(ncell)))                       # one pass: identical to not chunking at all
  }
  repeat {
    idx <- split(seq_len(ncell), ceiling(seq_len(ncell) / size))
    cat(sprintf("  %s: %d cells in %d chunk(s) of up to %d\n", tool, ncell, length(idx), size))
    flush.console()
    out <- rep(NA_character_, ncell)
    ok <- TRUE
    for (k in seq_along(idx)) {
      res <- tryCatch(fn(idx[[k]]), error = function(e) {
        cat(sprintf("    chunk %d/%d failed: %s\n", k, length(idx), conditionMessage(e)))
        NULL
      })
      if (is.null(res)) { ok <- FALSE; break }
      out[idx[[k]]] <- res
      gc()
    }
    if (ok) return(out)
    if (size <= FLOOR) stop(sprintf("%s still failing at the %d-cell floor", tool, FLOOR))
    size <- max(FLOOR, as.integer(size / 2))
    cat(sprintf("  %s: retrying with %d-cell chunks\n", tool, size)); flush.console()
  }
}

runtimes <- data.frame(dataset = character(), tool = character(),
                       n_cells = integer(), minutes = numeric())
log_rt <- function(tool, n, mins) {
  runtimes <<- rbind(runtimes, data.frame(dataset = DS, tool = tool,
                                          n_cells = n, minutes = round(mins, 3)))
  write.csv(runtimes, RT_OUT, row.names = FALSE)
  cat(sprintf("  [%s] %s: %.2f min on %d cells\n", DS, tool, mins, n)); flush.console()
}

q <- readH5AD(QUERY); r <- readH5AD(REF)
if ("high_quality" %in% colnames(colData(q)))
  q <- q[, as.integer(as.character(colData(q)$high_quality)) == 1]
qc <- as(assay(q, "X"), "CsparseMatrix"); rc <- as(assay(r, "X"), "CsparseMatrix")
ref_labels <- as.character(colData(r)$major_annotation)
keep <- !is.na(ref_labels) & ref_labels != "nan"
rc <- rc[, keep]; ref_labels <- ref_labels[keep]
cells <- colnames(qc); ncell <- length(cells)
shared <- intersect(rownames(rc), rownames(qc))
rc <- rc[shared, ]; qc <- qc[shared, ]
cat(sprintf("%s: query %d cells x %d genes | ref %d cells | shared genes %d\n",
            DS, ncell, nrow(qc), ncol(rc), length(shared))); flush.console()
preds <- data.frame(cell = cells, stringsAsFactors = FALSE)

## ---------- SingleR ----------
tryCatch({
  t <- Sys.time()
  rs <- logNormCounts(SingleCellExperiment(list(counts = rc)))
  singler_on <- function(idx) {
    qs <- logNormCounts(SingleCellExperiment(list(counts = qc[, idx, drop = FALSE])))
    sr <- tryCatch(SingleR(test = qs, ref = rs, labels = ref_labels, num.threads = 4),
                   error = function(e) SingleR(test = qs, ref = rs, labels = ref_labels))
    as.character(sr$labels)
  }
  preds$SingleR <- run_chunked("SingleR", singler_on, CHUNK)
  log_rt("SingleR", ncell, as.numeric(difftime(Sys.time(), t, units = "mins")))
  rm(rs); gc()
}, error = function(e) { cat("  SingleR FAILED:", conditionMessage(e), "\n"); preds$SingleR <<- NA })

## ---------- Seurat (LogNormalize anchor transfer) ----------
tryCatch({
  t <- Sys.time()
  rseu <- CreateSeuratObject(counts = rc); rseu$ct <- ref_labels
  rseu <- NormalizeData(rseu, verbose = FALSE); rseu <- FindVariableFeatures(rseu, verbose = FALSE)
  rseu <- ScaleData(rseu, features = shared, verbose = FALSE)
  rseu <- RunPCA(rseu, features = shared, npcs = 30, verbose = FALSE)
  seurat_on <- function(idx) {
    qseu <- CreateSeuratObject(counts = qc[, idx, drop = FALSE])
    qseu <- NormalizeData(qseu, verbose = FALSE)
    anc <- FindTransferAnchors(reference = rseu, query = qseu, features = shared,
                               dims = 1:30, reduction = "pcaproject", verbose = FALSE)
    td <- TransferData(anchorset = anc, refdata = rseu$ct, dims = 1:30, verbose = FALSE)
    as.character(td$predicted.id)
  }
  preds$Seurat <- run_chunked("Seurat", seurat_on, CHUNK)
  log_rt("Seurat", ncell, as.numeric(difftime(Sys.time(), t, units = "mins")))
  rm(rseu); gc()
}, error = function(e) { cat("  Seurat FAILED:", conditionMessage(e), "\n"); preds$Seurat <<- NA })

## ---------- Azimuth (SCTransform anchor transfer) ----------
tryCatch({
  t <- Sys.time()
  rs <- CreateSeuratObject(counts = rc); rs$ct <- ref_labels
  rs <- SCTransform(rs, verbose = FALSE); rs <- RunPCA(rs, npcs = 30, verbose = FALSE)
  azimuth_on <- function(idx) {
    qs <- CreateSeuratObject(counts = qc[, idx, drop = FALSE])
    qs <- SCTransform(qs, verbose = FALSE)
    anc <- FindTransferAnchors(reference = rs, query = qs, normalization.method = "SCT",
                               reference.reduction = "pca", dims = 1:30, verbose = FALSE)
    td <- TransferData(anchorset = anc, refdata = rs$ct, weight.reduction = "pcaproject",
                       dims = 1:30, verbose = FALSE)
    # SCTransform can drop cells, so align on names rather than assuming the order survived
    as.character(td$predicted.id)[match(cells[idx], colnames(qs))]
  }
  preds$Azimuth <- run_chunked("Azimuth", azimuth_on, CHUNK)
  log_rt("Azimuth", ncell, as.numeric(difftime(Sys.time(), t, units = "mins")))
  rm(rs); gc()
}, error = function(e) { cat("  Azimuth FAILED:", conditionMessage(e), "\n"); preds$Azimuth <<- NA })

## ---------- RCTD (spacexr), Xenium-appropriate params ----------
tryCatch({
  t <- Sys.time()
  # spacexr rejects cell-type names containing "/", and published references use them freely:
  # LuCA has "Transitional Club/AT2". Left alone this aborts RCTD for the whole run while the other
  # three annotators succeed, so the panel silently drops to three tools. Substitute for RCTD only
  # and restore the original names on the way out, so the emitted vocabulary is unchanged.
  safe_labels <- gsub("/", "_", ref_labels, fixed = TRUE)
  unsafe_map <- setNames(unique(ref_labels), unique(safe_labels))
  if (any(safe_labels != ref_labels))
    cat(sprintf("  RCTD: renamed %d label(s) containing '/' for spacexr\n",
                length(unique(ref_labels[safe_labels != ref_labels]))))
  ct <- factor(safe_labels); names(ct) <- colnames(rc)
  nU_r <- colSums(rc); names(nU_r) <- colnames(rc)
  # nUMI is computed over the SHARED PANEL genes only. On a 280-plex panel against a whole
  # transcriptome FFPE reference that leaves most reference cells under spacexr's default
  # min_UMI = 100, which silently dropped the reference from 10,689 to 1,950 cells and wiped out
  # CD4T/CD8T/DC/Neutrophil entirely. Lower the floor so the reference keeps its composition.
  reference <- Reference(rc, ct, nU_r, min_UMI = MIN_UMI)
  rctd_on <- function(idx) {
    sub <- qc[, idx, drop = FALSE]
    coords <- data.frame(x = seq_along(idx), y = rep(1, length(idx)), row.names = colnames(sub))
    nU_q <- colSums(sub); names(nU_q) <- colnames(sub)
    puck <- SpatialRNA(coords, sub, nU_q)
    myRCTD <- create.RCTD(puck, reference, max_cores = 4,
                          gene_cutoff = 0, fc_cutoff = 0, gene_cutoff_reg = 0,
                          fc_cutoff_reg = 0, UMI_min = 0, counts_MIN = 0,
                          UMI_min_sigma = 1, CELL_MIN_INSTANCE = 10)
    myRCTD <- run.RCTD(myRCTD, doublet_mode = "doublet")
    df <- myRCTD@results$results_df
    out <- rep(NA_character_, length(idx)); names(out) <- colnames(sub)
    first <- as.character(df$first_type)
    # undo the substitution so RCTD's labels match the other three annotators' vocabulary
    restored <- unname(unsafe_map[first]); restored[is.na(restored)] <- first[is.na(restored)]
    out[rownames(df)] <- restored
    unname(out)
  }
  preds$RCTD <- run_chunked("RCTD", rctd_on, CHUNK)
  log_rt("RCTD", ncell, as.numeric(difftime(Sys.time(), t, units = "mins")))
  rm(reference); gc()
}, error = function(e) { cat("  RCTD FAILED:", conditionMessage(e), "\n"); preds$RCTD <<- NA })

preds <- preds[, c("cell", "Seurat", "Azimuth", "RCTD", "SingleR")]
write.csv(preds, PRED_OUT, row.names = FALSE)
for (m in c("Seurat", "Azimuth", "RCTD", "SingleR"))
  cat(sprintf("  %s: %d/%d labelled\n", m, sum(!is.na(preds[[m]])), nrow(preds)))
cat("WROTE", PRED_OUT, "rows", nrow(preds), "\n")
