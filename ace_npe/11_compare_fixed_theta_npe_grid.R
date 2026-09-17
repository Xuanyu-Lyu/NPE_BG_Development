#!/usr/bin/env Rscript

# STEP 11 -- compare the fixed-theta NPE ensemble with a grid posterior.
#
# This script consumes the completed STEP 10b replicate directories. It does
# not train an NPE and does not simulate new test data. For every shared test
# dataset, it evaluates the Gaussian ACE likelihood on a fine (A, C, E)
# simplex grid under the same Dirichlet prior used for NPE training.
#
# Outputs:
#   grid_posterior_results.csv
#   grid_resolution_check.csv
#   npe_grid_paired_results.csv
#   calibration_by_model.csv
#   aggregate_calibration_summary.csv
#   fixed_theta_npe_vs_grid_posterior_mean.png
#   fixed_theta_npe_vs_grid_posterior_sd.png
#   fixed_theta_npe_grid_sd_ratio_boxplot.png
#
# The sufficient covariance summaries are:
#   mz_var = (S11 + S22) / 2, mz_cov = S12
#   dz_var = (S11 + S22) / 2, dz_cov = S12
# and (N - 1)S follows a Wishart distribution. Constants independent of ACE
# cancel when the grid weights are normalized.

options(stringsAsFactors = FALSE)

PARAMETERS <- c("A", "C", "E")
FEATURE_COLUMNS <- c("mz_var", "mz_cov", "dz_var", "dz_cov")
COLORS <- c(A = "#1f77b4", C = "#ff7f0e", E = "#2ca02c")

script_file <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_file) == 1L) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", script_file)))
} else {
  script_dir <- normalizePath(getwd())
}

defaults <- list(
  ensemble_dir = file.path(script_dir, "results", "fixed_theta_npe_ensemble"),
  output_dir = file.path(
    script_dir, "results", "fixed_theta_npe_grid_comparison"
  ),
  n_replicates = 100L,
  n_pairs = 1000L,
  grid_step = 0.001,
  grid_check_multiplier = 2,
  grid_check_tolerance = 0.0005,
  dirichlet_alpha = c(1, 1, 1),
  total_variance = 1
)

print_help <- function() {
  cat(paste0(
    "Usage: Rscript ace_npe/11_compare_fixed_theta_npe_grid.R [options]\n\n",
    "Options:\n",
    "  --ensemble_dir PATH       STEP 10b output directory\n",
    "  --output_dir PATH         Directory for STEP 11 outputs\n",
    "  --n_replicates N          Number of NPE replicates (default: 100)\n",
    "  --n_pairs N               MZ and DZ pairs per dataset (default: 1000)\n",
    "  --grid_step H             Simplex grid spacing (default: 0.001)\n",
    "  --grid_check_multiplier M Compare with a grid M times coarser (default: 2)\n",
    "  --grid_check_tolerance T  Warn if fine/coarse moments differ by > T\n",
    "  --dirichlet_alpha A C E   Dirichlet prior (default: 1 1 1)\n",
    "  --total_variance V        Fixed A+C+E total (default: 1)\n",
    "  --help                    Show this message\n"
  ))
}

parse_arguments <- function(args, values) {
  i <- 1L
  while (i <= length(args)) {
    option <- args[[i]]
    if (option == "--help") {
      print_help()
      quit(status = 0L)
    }
    if (option == "--dirichlet_alpha") {
      if (i + 3L > length(args)) {
        stop("--dirichlet_alpha requires three numbers", call. = FALSE)
      }
      values$dirichlet_alpha <- as.numeric(args[(i + 1L):(i + 3L)])
      i <- i + 4L
      next
    }
    key <- sub("^--", "", option)
    if (!startsWith(option, "--") || !key %in% names(values)) {
      stop("Unknown option: ", option, call. = FALSE)
    }
    if (i == length(args)) stop(option, " requires a value", call. = FALSE)
    raw_value <- args[[i + 1L]]
    if (key %in% c("n_replicates", "n_pairs")) {
      values[[key]] <- as.integer(raw_value)
    } else if (key %in% c(
      "grid_step", "grid_check_multiplier", "grid_check_tolerance",
      "total_variance"
    )) {
      values[[key]] <- as.numeric(raw_value)
    } else {
      values[[key]] <- raw_value
    }
    i <- i + 2L
  }
  values
}

args <- parse_arguments(commandArgs(trailingOnly = TRUE), defaults)

if (!dir.exists(args$ensemble_dir)) {
  stop("Ensemble directory does not exist: ", args$ensemble_dir, call. = FALSE)
}
if (!is.finite(args$n_replicates) || args$n_replicates < 1L) {
  stop("--n_replicates must be positive", call. = FALSE)
}
if (!is.finite(args$n_pairs) || args$n_pairs < 3L) {
  stop("--n_pairs must be at least 3", call. = FALSE)
}
if (!is.finite(args$grid_step) || args$grid_step <= 0 ||
    args$grid_step >= args$total_variance / 3) {
  stop("--grid_step must be positive and smaller than total_variance / 3",
       call. = FALSE)
}
if (!is.finite(args$grid_check_multiplier) ||
    args$grid_check_multiplier <= 1) {
  stop("--grid_check_multiplier must be greater than 1", call. = FALSE)
}
if (!is.finite(args$grid_check_tolerance) ||
    args$grid_check_tolerance <= 0) {
  stop("--grid_check_tolerance must be positive", call. = FALSE)
}
if (length(args$dirichlet_alpha) != 3L ||
    any(!is.finite(args$dirichlet_alpha)) ||
    any(args$dirichlet_alpha <= 0)) {
  stop("--dirichlet_alpha must contain three positive values", call. = FALSE)
}
if (!is.finite(args$total_variance) || args$total_variance <= 0) {
  stop("--total_variance must be positive", call. = FALSE)
}

dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

required_columns <- c(
  "replicate", "test_simulation", FEATURE_COLUMNS,
  paste0("true_", PARAMETERS),
  as.vector(outer(PARAMETERS, c("posterior_mean", "posterior_sd"), paste,
                  sep = "_"))
)

cat(sprintf("Reading %d completed NPE replicates ...\n", args$n_replicates))
npe_frames <- vector("list", args$n_replicates)
reference_test_data <- NULL

for (replicate_index in seq_len(args$n_replicates)) {
  replicate_dir <- file.path(
    args$ensemble_dir, sprintf("replicate_%03d", replicate_index)
  )
  result_path <- file.path(replicate_dir, "fixed_theta_posterior_results.csv")
  complete_path <- file.path(replicate_dir, "COMPLETE")
  if (!file.exists(complete_path) || !file.exists(result_path)) {
    stop("Missing completed replicate ", replicate_index, ": ", result_path,
         call. = FALSE)
  }
  frame <- read.csv(result_path, check.names = FALSE)
  missing_columns <- setdiff(required_columns, names(frame))
  if (length(missing_columns) > 0L) {
    stop("Replicate ", replicate_index, " is missing columns: ",
         paste(missing_columns, collapse = ", "), call. = FALSE)
  }
  if (anyDuplicated(frame$test_simulation)) {
    stop("Duplicate test_simulation values in replicate ", replicate_index,
         call. = FALSE)
  }
  frame <- frame[order(frame$test_simulation), required_columns, drop = FALSE]
  frame$replicate <- replicate_index

  shared_columns <- c(
    "test_simulation", FEATURE_COLUMNS, paste0("true_", PARAMETERS)
  )
  current_test_data <- frame[, shared_columns, drop = FALSE]
  if (is.null(reference_test_data)) {
    reference_test_data <- current_test_data
  } else {
    if (!identical(current_test_data$test_simulation,
                   reference_test_data$test_simulation)) {
      stop("Test simulation IDs differ in replicate ", replicate_index,
           call. = FALSE)
    }
    differences <- abs(
      as.matrix(current_test_data[, -1L]) -
        as.matrix(reference_test_data[, -1L])
    )
    if (any(!is.finite(differences)) || max(differences) > 1e-10) {
      stop(
        "STEP 11 requires the shared STEP 10b test set, but replicate ",
        replicate_index, " contains different test data.", call. = FALSE
      )
    }
  }
  npe_frames[[replicate_index]] <- frame
}

npe_results <- do.call(rbind, npe_frames)
rownames(npe_results) <- NULL
n_test <- nrow(reference_test_data)
if (n_test < 2L) stop("At least two test datasets are required", call. = FALSE)

true_theta <- as.numeric(reference_test_data[1L, paste0("true_", PARAMETERS)])
if (any(abs(rowSums(reference_test_data[, paste0("true_", PARAMETERS)]) -
            args$total_variance) > 1e-8)) {
  stop("True ACE values do not sum to --total_variance", call. = FALSE)
}

make_simplex_grid <- function(total_variance, step) {
  a_values <- seq(step, total_variance - 2 * step, by = step)
  counts <- floor(
    (total_variance - a_values - sqrt(.Machine$double.eps)) / step
  )
  keep <- counts >= 1L
  a_values <- a_values[keep]
  counts <- counts[keep]
  estimated_points <- sum(counts)
  if (estimated_points > 2e7) {
    stop(sprintf(
      "Grid would contain %s points; choose a larger --grid_step.",
      format(estimated_points, big.mark = ",", scientific = FALSE)
    ), call. = FALSE)
  }
  if (estimated_points > 5e6) {
    warning(sprintf(
      "Large grid: %s points", format(estimated_points, big.mark = ",")
    ))
  }
  A <- rep(a_values, counts)
  C <- unlist(
    lapply(counts, function(count) seq_len(count) * step),
    use.names = FALSE
  )
  E <- total_variance - A - C
  valid <- A > 0 & C > 0 & E > 0
  data.frame(A = A[valid], C = C[valid], E = E[valid])
}

evaluate_grid <- function(step, label) {
  cat(sprintf(
    "Building %s simplex grid (step = %.6g; total variance = %.6g) ...\n",
    label, step, args$total_variance
  ))
  grid <- make_simplex_grid(args$total_variance, step)
  cat(sprintf("%s grid points: %s\n", label,
              format(nrow(grid), big.mark = ",")))

  # Precompute the parameter-only portions of the Wishart log likelihood.
  V <- args$total_variance
  df <- args$n_pairs - 1
  mz_covariance <- grid$A + grid$C
  dz_covariance <- 0.5 * grid$A + grid$C
  mz_determinant <- V^2 - mz_covariance^2
  dz_determinant <- V^2 - dz_covariance^2
  if (any(mz_determinant <= 0) || any(dz_determinant <= 0)) {
    stop("The grid contains a non-positive-definite ACE covariance",
         call. = FALSE)
  }

  grid_matrix <- as.matrix(grid)
  grid_matrix_squared <- grid_matrix^2
  composition <- grid_matrix / V
  grid_edge <- apply(grid_matrix, 1L, min) <= 2 * step
  log_prior <- rowSums(
    sweep(log(composition), 2L, args$dirichlet_alpha - 1, `*`)
  )
  log_determinants <- log(mz_determinant) + log(dz_determinant)

  grid_moments <- function(data_row) {
    mz_trace <- (
      2 * V * data_row[["mz_var"]] -
        2 * mz_covariance * data_row[["mz_cov"]]
    ) / mz_determinant
    dz_trace <- (
      2 * V * data_row[["dz_var"]] -
        2 * dz_covariance * data_row[["dz_cov"]]
    ) / dz_determinant
    log_weight <- log_prior - 0.5 * df * (
      log_determinants + mz_trace + dz_trace
    )
    log_weight <- log_weight - max(log_weight)
    normalized_weight <- exp(log_weight)
    normalized_weight <- normalized_weight / sum(normalized_weight)
    means <- colSums(grid_matrix * normalized_weight)
    second_moments <- colSums(grid_matrix_squared * normalized_weight)
    variances <- pmax(second_moments - means^2, 0)
    c(
      setNames(means, paste0(PARAMETERS, "_grid_mean")),
      setNames(sqrt(variances), paste0(PARAMETERS, "_grid_sd")),
      grid_effective_points = 1 / sum(normalized_weight^2),
      grid_edge_mass = sum(normalized_weight[grid_edge])
    )
  }

  cat(sprintf("Evaluating %s grid for %d shared datasets ...\n", label, n_test))
  grid_rows <- vector("list", n_test)
  progress_every <- max(1L, floor(n_test / 10L))
  for (test_index in seq_len(n_test)) {
    grid_rows[[test_index]] <- grid_moments(
      reference_test_data[test_index, ]
    )
    if (test_index %% progress_every == 0L || test_index == n_test) {
      cat(sprintf("  %d/%d datasets\n", test_index, n_test))
    }
  }

  list(
    results = cbind(
      reference_test_data,
      as.data.frame(do.call(rbind, grid_rows), check.names = FALSE)
    ),
    n_points = nrow(grid)
  )
}

fine_grid <- evaluate_grid(args$grid_step, "primary")
grid_results <- fine_grid$results
grid_n_points <- fine_grid$n_points
grid_path <- file.path(args$output_dir, "grid_posterior_results.csv")
write.csv(grid_results, grid_path, row.names = FALSE)

# A second, coarser grid checks that the requested resolution is fine enough
# for the posterior means and SDs used below. This is a numerical diagnostic,
# not a second inferential method.
check_step <- args$grid_step * args$grid_check_multiplier
if (check_step >= args$total_variance / 3) {
  stop("The grid resolution-check step is too large; reduce ",
       "--grid_check_multiplier", call. = FALSE)
}
check_grid <- evaluate_grid(check_step, "resolution-check")
moment_columns <- as.vector(outer(
  PARAMETERS, c("grid_mean", "grid_sd"), paste, sep = "_"
))
resolution_check <- do.call(rbind, lapply(moment_columns, function(column) {
  difference <- abs(grid_results[[column]] - check_grid$results[[column]])
  data.frame(
    quantity = column,
    primary_step = args$grid_step,
    check_step = check_step,
    max_absolute_difference = max(difference),
    mean_absolute_difference = mean(difference),
    tolerance = args$grid_check_tolerance,
    passed = max(difference) <= args$grid_check_tolerance
  )
}))
resolution_path <- file.path(args$output_dir, "grid_resolution_check.csv")
write.csv(resolution_check, resolution_path, row.names = FALSE)
if (!all(resolution_check$passed)) {
  warning(
    "Grid resolution check exceeded the tolerance for: ",
    paste(resolution_check$quantity[!resolution_check$passed],
          collapse = ", "),
    ". Reduce --grid_step and rerun."
  )
}
rm(check_grid)
invisible(gc())

paired_frames <- lapply(PARAMETERS, function(parameter) {
  match_index <- match(npe_results$test_simulation, grid_results$test_simulation)
  data.frame(
    replicate = npe_results$replicate,
    test_simulation = npe_results$test_simulation,
    parameter = parameter,
    true_value = npe_results[[paste0("true_", parameter)]],
    npe_posterior_mean = npe_results[[paste0(parameter, "_posterior_mean")]],
    grid_posterior_mean = grid_results[
      match_index, paste0(parameter, "_grid_mean")
    ],
    npe_posterior_sd = npe_results[[paste0(parameter, "_posterior_sd")]],
    grid_posterior_sd = grid_results[
      match_index, paste0(parameter, "_grid_sd")
    ]
  )
})
paired <- do.call(rbind, paired_frames)
paired$npe_minus_grid_mean <-
  paired$npe_posterior_mean - paired$grid_posterior_mean
paired$npe_sd_over_grid_sd <-
  paired$npe_posterior_sd / paired$grid_posterior_sd
if (any(!is.finite(as.matrix(paired[, 5:ncol(paired)])))) {
  stop("Non-finite value encountered in paired NPE/grid results",
       call. = FALSE)
}
paired_path <- file.path(args$output_dir, "npe_grid_paired_results.csv")
write.csv(paired, paired_path, row.names = FALSE)

calibration_rows <- list()
row_index <- 0L
for (parameter in PARAMETERS) {
  selected <- paired[paired$parameter == parameter, ]
  for (replicate_index in seq_len(args$n_replicates)) {
    one_model <- selected[selected$replicate == replicate_index, ]
    posterior_rms_sd <- sqrt(mean(one_model$npe_posterior_sd^2))
    empirical_se <- sd(one_model$npe_posterior_mean)
    row_index <- row_index + 1L
    calibration_rows[[row_index]] <- data.frame(
      method = "NPE",
      replicate = replicate_index,
      parameter = parameter,
      posterior_rms_sd = posterior_rms_sd,
      empirical_se = empirical_se,
      calibration_ratio = posterior_rms_sd / empirical_se
    )
  }

  grid_one_per_dataset <- selected[
    !duplicated(selected$test_simulation),
    c("test_simulation", "grid_posterior_mean", "grid_posterior_sd")
  ]
  posterior_rms_sd <- sqrt(mean(grid_one_per_dataset$grid_posterior_sd^2))
  empirical_se <- sd(grid_one_per_dataset$grid_posterior_mean)
  row_index <- row_index + 1L
  calibration_rows[[row_index]] <- data.frame(
    method = "Grid",
    replicate = NA_integer_,
    parameter = parameter,
    posterior_rms_sd = posterior_rms_sd,
    empirical_se = empirical_se,
    calibration_ratio = posterior_rms_sd / empirical_se
  )
}
calibration <- do.call(rbind, calibration_rows)
calibration_path <- file.path(args$output_dir, "calibration_by_model.csv")
write.csv(calibration, calibration_path, row.names = FALSE)

summary_rows <- list()
row_index <- 0L
for (parameter in PARAMETERS) {
  npe_calibration <- calibration[
    calibration$method == "NPE" & calibration$parameter == parameter,
  ]
  grid_calibration <- calibration[
    calibration$method == "Grid" & calibration$parameter == parameter,
  ]
  row_index <- row_index + 1L
  summary_rows[[row_index]] <- data.frame(
    method = "NPE",
    parameter = parameter,
    n_estimates = nrow(npe_calibration),
    calibration_ratio = median(npe_calibration$calibration_ratio),
    ratio_q1 = unname(quantile(npe_calibration$calibration_ratio, 0.25)),
    ratio_q3 = unname(quantile(npe_calibration$calibration_ratio, 0.75)),
    aggregation = "median and IQR of model-wise ratios across NPEs"
  )
  row_index <- row_index + 1L
  summary_rows[[row_index]] <- data.frame(
    method = "Grid",
    parameter = parameter,
    n_estimates = 1L,
    calibration_ratio = grid_calibration$calibration_ratio,
    ratio_q1 = NA_real_,
    ratio_q3 = NA_real_,
    aggregation = "one ratio across the shared test datasets"
  )
}
calibration_summary <- do.call(rbind, summary_rows)
summary_path <- file.path(args$output_dir, "aggregate_calibration_summary.csv")
write.csv(calibration_summary, summary_path, row.names = FALSE)

padded_range <- function(x, include = numeric()) {
  limits <- range(c(x, include), finite = TRUE)
  width <- diff(limits)
  if (width == 0) width <- max(abs(limits), 1) * 0.05
  limits + c(-1, 1) * 0.06 * width
}

plot_paired_scatter <- function(x_column, y_column, output_path, title,
                                x_label, y_label) {
  png(output_path, width = 3200, height = 1200, res = 200)
  old_par <- par(no.readonly = TRUE)
  on.exit({par(old_par); dev.off()}, add = TRUE)
  par(mfrow = c(1, 3), mar = c(5.2, 5.2, 4.0, 1.2),
      oma = c(0, 0, 3.0, 0), las = 1)
  for (parameter in PARAMETERS) {
    selected <- paired[paired$parameter == parameter, ]
    limits <- padded_range(c(selected[[x_column]], selected[[y_column]]))
    plot(
      selected[[x_column]], selected[[y_column]],
      xlim = limits, ylim = limits, asp = 1,
      pch = 16, cex = 0.42,
      col = adjustcolor(COLORS[[parameter]], alpha.f = 0.15),
      xlab = x_label, ylab = y_label,
      main = sprintf("%s (true %.1f)", parameter,
                     true_theta[match(parameter, PARAMETERS)])
    )
    abline(a = 0, b = 1, lty = 2, lwd = 1.5, col = "#555555")

    # The open circles summarize the 100 NPEs for each individual dataset;
    # the faint points retain every paired model-dataset comparison.
    average_by_dataset <- aggregate(
      selected[[y_column]],
      by = list(test_simulation = selected$test_simulation),
      FUN = mean
    )
    grid_by_dataset <- selected[
      !duplicated(selected$test_simulation),
      c("test_simulation", x_column)
    ]
    average_by_dataset <- merge(
      average_by_dataset, grid_by_dataset,
      by = "test_simulation", sort = TRUE
    )
    points(
      average_by_dataset[[x_column]], average_by_dataset$x,
      pch = 21, cex = 0.68, lwd = 0.8,
      col = COLORS[[parameter]], bg = adjustcolor("white", alpha.f = 0.75)
    )
    grid(col = adjustcolor("#777777", alpha.f = 0.18))
    box()
    if (parameter == "A") {
      legend(
        "topleft",
        legend = c("All NPE x dataset pairs", "Mean NPE over models", "Identity"),
        pch = c(16, 21, NA), lty = c(NA, NA, 2),
        col = c(adjustcolor(COLORS[[parameter]], alpha.f = 0.35),
                COLORS[[parameter]], "#555555"),
        pt.bg = c(NA, "white", NA), cex = 0.72, bty = "n"
      )
    }
  }
  mtext(title, side = 3, outer = TRUE, line = 1.0, cex = 1.25)
}

mean_plot_path <- file.path(
  args$output_dir, "fixed_theta_npe_vs_grid_posterior_mean.png"
)
plot_paired_scatter(
  "grid_posterior_mean", "npe_posterior_mean", mean_plot_path,
  sprintf(
    "Paired posterior means: %d NPEs x %d shared datasets; N = %d pairs",
    args$n_replicates, n_test, args$n_pairs
  ),
  "Grid posterior mean", "NPE posterior mean"
)

sd_plot_path <- file.path(
  args$output_dir, "fixed_theta_npe_vs_grid_posterior_sd.png"
)
plot_paired_scatter(
  "grid_posterior_sd", "npe_posterior_sd", sd_plot_path,
  sprintf(
    "Paired posterior SDs: %d NPEs x %d shared datasets; N = %d pairs",
    args$n_replicates, n_test, args$n_pairs
  ),
  "Grid posterior SD", "NPE posterior SD"
)

ratio_plot_path <- file.path(
  args$output_dir, "fixed_theta_npe_grid_sd_ratio_boxplot.png"
)
all_ratios <- paired$npe_sd_over_grid_sd
ratio_limits <- range(all_ratios, finite = TRUE)
if (diff(ratio_limits) == 0) {
  ratio_limits <- ratio_limits + c(-1, 1) * max(abs(ratio_limits), 1) * 0.01
}
png(ratio_plot_path, width = 3200, height = 1200, res = 200)
old_par <- par(no.readonly = TRUE)
par(mfrow = c(1, 3), mar = c(4.8, 5.4, 5.3, 1.2),
    oma = c(1.5, 0, 3.2, 0), las = 1)
set.seed(20260917)
for (parameter in PARAMETERS) {
  selected <- paired[paired$parameter == parameter, ]
  npe_calibration <- calibration[
    calibration$method == "NPE" & calibration$parameter == parameter,
  ]
  grid_calibration <- calibration[
    calibration$method == "Grid" & calibration$parameter == parameter,
  ]
  boxplot(
    selected$npe_sd_over_grid_sd,
    outline = TRUE, ylim = ratio_limits, yaxs = "i", xaxt = "n",
    col = adjustcolor(COLORS[[parameter]], alpha.f = 0.28),
    border = COLORS[[parameter]],
    outpch = 16, outcex = 0.42,
    outcol = adjustcolor(COLORS[[parameter]], alpha.f = 0.55),
    ylab = "NPE posterior SD / grid posterior SD",
    main = sprintf(
      "%s\nNPE calibration: %.3f [%.3f, %.3f]\nGrid calibration: %.3f",
      parameter,
      median(npe_calibration$calibration_ratio),
      quantile(npe_calibration$calibration_ratio, 0.25),
      quantile(npe_calibration$calibration_ratio, 0.75),
      grid_calibration$calibration_ratio
    )
  )
  abline(h = 1, lty = 2, lwd = 1.5, col = "#555555")
  model_medians <- aggregate(
    selected$npe_sd_over_grid_sd,
    by = list(replicate = selected$replicate),
    FUN = median
  )
  points(
    jitter(rep(1, nrow(model_medians)), amount = 0.075),
    model_medians$x,
    pch = 21, cex = 0.60,
    col = COLORS[[parameter]], bg = adjustcolor("white", alpha.f = 0.72)
  )
  grid(col = adjustcolor("#777777", alpha.f = 0.18))
  box()
  axis(1, at = 1, labels = sprintf("%s\n%d paired ratios",
                                    parameter, nrow(selected)))
}
mtext(
  sprintf(
    "Posterior-SD agreement: %d NPEs versus the grid reference; N = %d pairs",
    args$n_replicates, args$n_pairs
  ),
  side = 3, outer = TRUE, line = 1.1, cex = 1.25
)
mtext(
  "Box: all model-dataset ratios; circles: each model's median. Calibration = RMS posterior SD / empirical SE.",
  side = 1, outer = TRUE, line = 0.3, cex = 0.82
)
par(old_par)
dev.off()

summary_text <- c(
  "STEP 11 fixed-theta NPE versus grid posterior comparison",
  sprintf("Generated: %s", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  sprintf("Ensemble directory: %s", normalizePath(args$ensemble_dir)),
  sprintf("NPE replicates: %d", args$n_replicates),
  sprintf("Shared test datasets: %d", n_test),
  sprintf("Pairs per zygosity: %d", args$n_pairs),
  sprintf("Grid step: %.8g", args$grid_step),
  sprintf("Primary grid points: %d", grid_n_points),
  sprintf("Resolution-check grid step: %.8g", check_step),
  sprintf("Resolution check passed: %s", all(resolution_check$passed)),
  sprintf("Largest fine/coarse moment difference: %.8g",
          max(resolution_check$max_absolute_difference)),
  sprintf("Dirichlet alpha: %s", paste(args$dirichlet_alpha, collapse = ", ")),
  sprintf("Total variance: %.8g", args$total_variance),
  "",
  "Calibration ratio = sqrt(mean(posterior variance)) /",
  "                    SD(posterior means across shared datasets).",
  "NPE annotations report the median [IQR] of the model-wise ratios.",
  "Grid annotations report its single ratio over the shared datasets.",
  "The ratio plot uses the full observed range across A, C, and E and",
  "shows all standard boxplot outliers."
)
writeLines(summary_text, file.path(args$output_dir, "run_summary.txt"))

cat("\nSTEP 11 complete.\n")
cat("Grid results:       ", grid_path, "\n", sep = "")
cat("Resolution check:    ", resolution_path, "\n", sep = "")
cat("Paired results:     ", paired_path, "\n", sep = "")
cat("Calibration summary:", summary_path, "\n", sep = "")
cat("Mean comparison:    ", mean_plot_path, "\n", sep = "")
cat("SD comparison:      ", sd_plot_path, "\n", sep = "")
cat("SD-ratio boxplot:   ", ratio_plot_path, "\n", sep = "")
