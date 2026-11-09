args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: standard_lisi.R <input-directory> <output.csv>")
}
if (!requireNamespace("lisi", quietly = TRUE)) {
  stop("The R package 'lisi' is not installed.")
}

input_dir <- args[[1L]]
output_path <- args[[2L]]
metadata <- utils::read.csv(
  file.path(input_dir, "metadata.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

read_embedding <- function(stage) {
  frame <- utils::read.csv(
    file.path(input_dir, paste0(stage, ".csv")),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (!identical(as.character(frame[[1L]]), as.character(metadata[[1L]]))) {
    stop(paste("Embedding sample IDs do not match metadata for", stage))
  }
  values <- as.matrix(frame[, -1L, drop = FALSE])
  storage.mode(values) <- "double"
  if (any(!is.finite(values))) stop(paste("Non-finite embedding for", stage))
  values
}

expected_adjust <- function(values, labels) {
  probabilities <- as.numeric(table(labels)) / length(labels)
  expected <- 1 / sum(probabilities ^ 2)
  denominator <- expected - 1
  if (!is.finite(denominator) || denominator <= .Machine$double.eps) {
    return(rep(NA_real_, length(values)))
  }
  pmin(1, pmax(0, (values - 1) / denominator))
}

calculate_lisi <- function(values, labels) {
  labels <- factor(labels)
  if (nlevels(labels) < 2L || min(table(labels)) < 2L || nrow(values) < 7L) {
    return(NA_real_)
  }
  perplexity <- min(30, max(2, floor((nrow(values) - 1) / 3)))
  result <- tryCatch(
    lisi::compute_lisi(
      values,
      data.frame(label = labels),
      label_colnames = "label",
      perplexity = perplexity
    )$label,
    error = function(error) NULL
  )
  if (is.null(result)) return(NA_real_)
  adjusted <- expected_adjust(result, labels)
  adjusted <- adjusted[is.finite(adjusted)]
  if (length(adjusted)) stats::median(adjusted) else NA_real_
}

stage_metrics <- function(stage) {
  values <- read_embedding(stage)
  strata <- as.character(metadata$group)
  levels <- unique(strata)
  batch_values <- vapply(levels, function(level) {
    keep <- strata == level
    calculate_lisi(values[keep, , drop = FALSE], metadata$batch[keep])
  }, numeric(1))
  weights <- vapply(levels, function(level) sum(strata == level), numeric(1))
  keep <- is.finite(batch_values)
  batch_lisi <- if (any(keep)) {
    stats::weighted.mean(batch_values[keep], weights[keep])
  } else {
    NA_real_
  }
  biology_lisi <- calculate_lisi(values, metadata$group)
  data.frame(
    stage = stage,
    batch_ilisi_expected_adjusted_stratified = batch_lisi,
    biological_factor_separation_score = if (is.finite(biology_lisi)) {
      1 - biology_lisi
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
}

stages <- c("observed", "corrected", "batch_neutral_reference")
output <- do.call(rbind, lapply(stages, stage_metrics))
utils::write.csv(output, output_path, row.names = FALSE)
