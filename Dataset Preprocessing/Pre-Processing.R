# Make sure BiocManager is installed
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# Update BiocManager to latest
BiocManager::install(version = "3.18")

install.packages("rvest", dependencies = TRUE, type = "binary")

unlink("C:/Users/rheam/AppData/Local/Temp/Rtmp*", recursive = TRUE) # clear temp
BiocManager::install("MOFAdata", ask = FALSE, update = TRUE)

options(BioC_mirror="https://bioconductor.org")
BiocManager::install("MOFAdata")
BiocManager::install("bioFAM/MOFAdata", dependencies = TRUE)


  # Install MOFA2 (if not already)
BiocManager::install("MOFA2")

version


# Load libraries
library(MOFA2)
library(MOFAdata)
