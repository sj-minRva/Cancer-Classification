# 1) Do a basic internet check
if (!requireNamespace("curl", quietly=TRUE)) install.packages("curl")
curl::has_internet()

# 2) Try fetching Bioconductor homepage (shows if SSL/network ok)
url <- "https://bioconductor.org"
tryCatch(readLines(url, n = 1), error = function(e) e)

# 3) See current repos
getOption("repos")

# inside R, set for the session (replace host/port/user/pass if needed)
Sys.setenv(http_proxy  = "http://proxy.myorg.com:8080")
Sys.setenv(https_proxy = "http://proxy.myorg.com:8080")
# If authentication required:
# Sys.setenv(http_proxy  = "http://user:pass@proxy.myorg.com:8080")




# --- Install (one-time) ---
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
BiocManager::install(c("curatedTCGAData", "MultiAssayExperiment", "SummarizedExperiment",
                       "ExperimentHub", "BiocGenerics", "matrixStats", "MOFA2", "basilisk"))

# --- Load libraries ---
library(curatedTCGAData)
library(MultiAssayExperiment)
library(SummarizedExperiment)
library(matrixStats)
library(MOFA2)      # MOFA2 R package

# --- 1) List available curatedTCGAData experiments (example) ---
available_exps <- curatedTCGADataList()
head(available_exps)  # shows available tumor types and assays

# --- 2) Download a TCGA cohort as a MultiAssayExperiment
# Replace "BRCA" with the cohort you want. 
# Use "ExperimentHub" style name: "BRCA" -> "BRCA" is example; curatedTCGAData uses names like "BRCA"
cohort <- "BRCA"
mae <- curatedTCGAData(cohort, assays = c("RNASeq2GeneNorm", "Methylation", "miRNASeqGene",
                                          "CopyNumber", "RPPA"), dry.run = FALSE)

# NOTE: the assays available differ by cohort. Check names:
assayNames(mae)        # lists assays included for this cohort
colData_names <- colnames(colData(mae))  # metadata available

# --- 3) Inspect & extract assays (SummarizedExperiment objects)
# Example extraction (use exact assay names from assayNames(mae))
# Find closest matches to RNA / miRNA / methylation
assayNames(mae)

# Common example: "RNASeq2GeneNorm" or "RNASeq2GeneNorm_log2"
rna_se <- tryCatch(experiments(mae)[["RNASeq2GeneNorm"]], error = function(e) NULL)
mirna_se <- tryCatch(experiments(mae)[["miRNASeqGene"]], error = function(e) NULL)
meth_se <- tryCatch(experiments(mae)[["Methylation"]], error = function(e) NULL)
cnv_se  <- tryCatch(experiments(mae)[["CopyNumber"]], error = function(e) NULL)
rppa_se <- tryCatch(experiments(mae)[["RPPA"]], error = function(e) NULL)

# Convert to matrices (features x samples). Experiment objects may have different assay names; inspect
get_matrix <- function(se) {
  if (is.null(se)) return(NULL)
  # pick first assay
  a <- assay(se)
  m <- as.matrix(a)
  # ensure rownames/colnames
  rownames(m) <- rowData(se)$feature_name %||% rownames(m)
  colnames(m) <- colnames(se)
  return(m)
}

rna_mat  <- get_matrix(rna_se)
mirna_mat <- get_matrix(mirna_se)
meth_mat <- get_matrix(meth_se)
cnv_mat  <- get_matrix(cnv_se)
rppa_mat <- get_matrix(rppa_se)

# --- 4) Harmonize sample IDs to patient-level TCGA barcodes
# TCGA sample barcodes: patient = first 12 chars, sample type = chars 14-15 ("01" primary tumor)
patient_id <- function(barcodes) substr(as.character(barcodes), 1, 12)
colnames_to_patients <- function(mat) {
  if (is.null(mat)) return(NULL)
  colnames(mat) <- patient_id(colnames(mat))
  return(mat)
}

rna_mat <- colnames_to_patients(rna_mat)
mirna_mat <- colnames_to_patients(mirna_mat)
meth_mat <- colnames_to_patients(meth_mat)
cnv_mat  <- colnames_to_patients(cnv_mat)
rppa_mat <- colnames_to_patients(rppa_mat)

# --- 5) Subset to primary tumor patients only (optional)
# If your matrices include normal or multiple aliquots, ensure patient-level uniqueness.
unique_patients <- function(mat) {
  if (is.null(mat)) return(NULL)
  # If duplicate patient columns (multiple aliquots), keep first or aggregate (mean)
  if (any(duplicated(colnames(mat)))) {
    warning("Duplicate patient columns detected; aggregating by mean across duplicates.")
    # aggregate: compute column means by patient
    patients <- unique(colnames(mat))
    mat2 <- sapply(patients, function(p) {
      cols <- which(colnames(mat) == p)
      if (length(cols) == 1) return(mat[, cols])
      rowMeans(mat[, cols, drop=FALSE], na.rm = TRUE)
    })
    colnames(mat2) <- unique(colnames(mat))
    return(as.matrix(mat2))
  } else {
    return(mat)
  }
}

rna_mat <- unique_patients(rna_mat)
mirna_mat <- unique_patients(mirna_mat)
meth_mat <- unique_patients(meth_mat)
cnv_mat  <- unique_patients(cnv_mat)
rppa_mat <- unique_patients(rppa_mat)

# --- 6) Intersect samples across views (common patients)
views_present <- list(RNA = rna_mat, miRNA = mirna_mat, METH = meth_mat, CNV = cnv_mat, RPPA = rppa_mat)
# remove NULLs
views_present <- views_present[!sapply(views_present, is.null)]

common_samples <- Reduce(intersect, lapply(views_present, colnames))
length(common_samples)
if (length(common_samples) < 10) stop("Too few common samples across requested views; consider fewer views or another cohort.")

# Subset each view to common samples and ensure numeric matrix
for (nm in names(views_present)) {
  views_present[[nm]] <- as.matrix(views_present[[nm]][, common_samples, drop = FALSE])
}

# --- 7) Filter features (optional): keep highly variable features to reduce size
keep_top_var <- function(mat, top = 5000) {
  if (is.null(mat)) return(NULL)
  v <- rowVars(mat, na.rm=TRUE)
  topn <- min(sum(!is.na(v)), top)
  idx <- order(v, decreasing = TRUE)[1:topn]
  return(mat[idx, , drop = FALSE])
}

views_present$RNA  <- keep_top_var(views_present$RNA, top = 5000)
if (!is.null(views_present$METH)) views_present$METH <- keep_top_var(views_present$METH, top = 10000)
if (!is.null(views_present$miRNA)) views_present$miRNA <- keep_top_var(views_present$miRNA, top = 500)

# --- 8) Build MOFA2 object and run (example)
mofa_input <- create_mofa(views_present)   # views: named list of matrices features x samples

data_opts  <- get_default_data_options(mofa_input)
model_opts <- get_default_model_options(mofa_input)
train_opts <- get_default_training_options(mofa_input)

model_opts$num_factors <- 10
# Set Gaussian likelihood for continuous views (adjust if you have counts)
model_opts$list_likelihoods <- setNames(
  rep("gaussian", length(views_present)), names(views_present)
)

mofa_input <- prepare_mofa(mofa_input, data_opts = data_opts, model_opts = model_opts, training_opts = train_opts)
trained_mofa <- run_mofa(mofa_input)

# Save results
saveRDS(trained_mofa, file = "trained_mofa_BRCA.rds")

# --- End ---

