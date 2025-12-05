# R: Convert .rds to CSV
# Run in RStudio. Install packages if needed:
if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
  BiocManager::install("SummarizedExperiment")
}
if (!requireNamespace("Biobase", quietly = TRUE)) {
  BiocManager::install("Biobase")
}

library(SummarizedExperiment)  # for SummarizedExperiment
library(Biobase)               # for ExpressionSet

# Path to your rds file (use double backslashes or forward slashes on Windows)
rds_path <- "C:\\RHEA\\S7\\Project\\Code\\MOFA integration\\Zenodo\\BRCA (Breast invasive carcinoma).rds"

# Where to save CSV(s)
out_dir <- "C:\\RHEA\\S7\\Project\\Code\\MOFA integration\\Zenodo"
# create output dir if missing
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

obj <- readRDS(rds_path)
cat("Class of object loaded:", class(obj), "\n\n")

# Helper to write a data.frame or matrix to CSV
write_table <- function(tbl, name) {
  # tbl should be data.frame or matrix
  if (is.matrix(tbl)) tbl <- as.data.frame(tbl)
  # give safe filename
  safe_name <- gsub("[^A-Za-z0-9_\\-\\.]", "_", name)
  out_path <- file.path(out_dir, paste0(safe_name, ".csv"))
  write.csv(tbl, out_path, row.names = TRUE)
  message("Wrote: ", out_path)
}

# Handle commonly seen types
if (is.data.frame(obj)) {
  write_table(obj, "BRCA_dataframe")
} else if (is.matrix(obj)) {
  write_table(obj, "BRCA_matrix")
} else if (inherits(obj, "SummarizedExperiment")) {
  # extract assays (usually one main assay). If there are many, save all.
  assays_list <- SummarizedExperiment::assays(obj)
  if (length(assays_list) == 1) {
    write_table(assays_list[[1]], "BRCA_assay")
  } else {
    for (i in seq_along(assays_list)) {
      write_table(assays_list[[i]], paste0("BRCA_assay_", i))
    }
  }
  # also save colData (sample metadata)
  if (!is.null(colData(obj))) {
    write_table(as.data.frame(colData(obj)), "BRCA_sample_metadata")
  }
} else if (inherits(obj, "ExpressionSet")) {
  # Biobase ExpressionSet
  expr <- Biobase::exprs(obj)
  write_table(expr, "BRCA_exprs")
  pheno <- Biobase::pData(obj)
  if (!is.null(pheno)) write_table(as.data.frame(pheno), "BRCA_pData")
} else if (is.list(obj)) {
  # If it's a list, try to save each element that is a table-like object
  i <- 1
  for (nm in names(obj)) {
    elem <- obj[[nm]]
    if (is.null(nm) || nm == "") nm <- paste0("element_", i)
    if (is.data.frame(elem) || is.matrix(elem)) {
      write_table(elem, paste0("BRCA_list_", nm))
    } else if (inherits(elem, "SummarizedExperiment")) {
      assays_list <- SummarizedExperiment::assays(elem)
      for (j in seq_along(assays_list)) {
        write_table(assays_list[[j]], paste0("BRCA_list_", nm, "_assay_", j))
      }
      if (!is.null(colData(elem))) write_table(as.data.frame(colData(elem)), paste0("BRCA_list_", nm, "_colData"))
    } else {
      message("Skipping list element '", nm, "' (class: ", class(elem), ")")
    }
    i <- i + 1
  }
} else {
  # fallback: try to coerce to data.frame
  tryCatch({
    df_try <- as.data.frame(obj)
    write_table(df_try, "BRCA_coerced")
  }, error = function(e) {
    stop("Object of class ", paste(class(obj), collapse = ","), " is not recognized and could not be coerced to a data.frame. Inspect 'obj' in RStudio to choose what to export.")
  })
}

message("Done. Check the directory: ", out_dir)

