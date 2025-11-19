if (!require("BiocManager"))install.packages("BiocManager")
BiocManager::install("MultiAssayExperiment")

library(MultiAssayExperiment)
library(GenomicRanges)
library(SummarizedExperiment)
library(RaggedExperiment)
