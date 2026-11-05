# STEP 04 backend -- fit fixed-total (A+C+E=1) OpenMx to covariances.
#
# Usage:
#   Rscript 04_fit_openmx_paired_dirichlet.R INPUT.csv OUTPUT.csv
#
# INPUT contains one row per Dirichlet condition and fixed N. The Python STEP 04
# entry point evaluates the NPE on summaries of these exact same covariance
# matrices, making the comparison paired rather than merely distributional.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript 04_fit_openmx_paired_dirichlet.R INPUT.csv OUTPUT.csv")
}

input_file  <- normalizePath(args[1L], mustWork = TRUE)
output_file <- normalizePath(args[2L], mustWork = FALSE)

suppressPackageStartupMessages(library(OpenMx))
mxOption(NULL, "Standard Errors", "Yes")
mxOption(NULL, "Calculate Hessian", "Yes")

E_FLOOR <- 1e-6
OPENMX_MODEL <- "fixed_total_variance_1"

build_ace_model <- function(obs_mz, obs_dz, n_mz, n_dz) {
  selected <- c("T1", "T2")
  c_mz <- obs_mz[1L, 2L]
  c_dz <- obs_dz[1L, 2L]

  # Start with E >= 0.05 so near-singular sample covariances do not place the
  # optimizer at a nearly singular expected covariance on its first step.
  start_a <- min(max(2 * (c_mz - c_dz), 1e-4), 0.95 - 1e-4)
  start_c <- min(max(2 * c_dz - c_mz, 1e-4), 0.95 - start_a)
  start_q <- start_c / (1 - E_FLOOR - start_a)

  # Two bounded free parameters span the ACE simplex without mxConstraint:
  # C = (1 - E_FLOOR - A) * Q and E = E_FLOOR + (1 - E_FLOOR - A) * (1 - Q).
  var_a <- mxMatrix(
    "Full", 1L, 1L, free = TRUE, values = start_a,
    lbound = 0, ubound = 1 - E_FLOOR, label = "VA11", name = "VA"
  )
  fraction_c <- mxMatrix(
    "Full", 1L, 1L, free = TRUE, values = start_q,
    lbound = 0, ubound = 1, label = "Q11", name = "Q"
  )
  e_floor <- mxMatrix(
    "Full", 1L, 1L, free = FALSE, values = E_FLOOR, name = "Emin"
  )
  var_c <- mxAlgebra((1 - Emin - VA) * Q, name = "VC")
  var_e <- mxAlgebra(Emin + (1 - Emin - VA) * (1 - Q), name = "VE")
  total <- mxAlgebra(VA + VC + VE, name = "V")
  parameters <- list(var_a, fraction_c, e_floor, var_c, var_e, total)

  cov_mz <- mxAlgebra(VA + VC, name = "cMZ")
  expected_mz <- mxAlgebra(
    rbind(cbind(V, cMZ), cbind(t(cMZ), V)), name = "expCovMZ"
  )
  model_mz <- mxModel(
    parameters,
    cov_mz,
    expected_mz,
    mxData(observed = obs_mz, type = "cov", numObs = n_mz),
    mxExpectationNormal(covariance = "expCovMZ", dimnames = selected),
    mxFitFunctionML(),
    name = "MZ"
  )

  cov_dz <- mxAlgebra(0.5 %x% VA + VC, name = "cDZ")
  expected_dz <- mxAlgebra(
    rbind(cbind(V, cDZ), cbind(t(cDZ), V)), name = "expCovDZ"
  )
  model_dz <- mxModel(
    parameters,
    cov_dz,
    expected_dz,
    mxData(observed = obs_dz, type = "cov", numObs = n_dz),
    mxExpectationNormal(covariance = "expCovDZ", dimnames = selected),
    mxFitFunctionML(),
    name = "DZ"
  )

  mxModel(
    "ACEvc",
    parameters,
    model_mz,
    model_dz,
    mxFitFunctionMultigroup(c("MZ", "DZ"))
  )
}

fixed_total_se <- function(va, q, vcov_matrix) {
  output <- setNames(rep(NA_real_, 3L), c("A", "C", "E"))
  if (is.null(vcov_matrix) || !is.matrix(vcov_matrix)) return(output)

  names_vcov <- rownames(vcov_matrix)
  indices <- match(c("VA11", "Q11"), names_vcov)
  if (anyNA(indices)) return(output)

  jacobian <- rbind(
    c(1, 0),
    c(-q, 1 - E_FLOOR - va),
    c(-(1 - q), -(1 - E_FLOOR - va))
  )
  covariance <- jacobian %*%
    vcov_matrix[indices, indices, drop = FALSE] %*%
    t(jacobian)
  setNames(sqrt(pmax(diag(covariance), 0)), c("A", "C", "E"))
}

# Wu & Neale (2012), Section 2.3: the Wald-test-adjusted interval for A or C
# is assembled from likelihood-profile limits under the fixed-total model.
# It is an interval adjustment, not a replacement for the Hessian SE above.
# For covariance input, OpenMx's -2 log likelihood (up to a constant) is
# n*log|Sigma| + (n-1)*tr(S %*% solve(Sigma)) in each group. Profiling the
# other ACE component leaves one nuisance parameter to optimize.
make_fixed_total_profile <- function(obs_mz, obs_dz, n_mz, n_dz, parameter) {
  stopifnot(parameter %in% c("A", "C"))
  mz_diagonal <- sum(diag(obs_mz))
  dz_diagonal <- sum(diag(obs_dz))
  mz_off_diagonal <- obs_mz[1L, 2L]
  dz_off_diagonal <- obs_dz[1L, 2L]
  objective <- function(a, c) {
    e <- 1 - a - c
    mz_correlation <- a + c
    dz_correlation <- 0.5 * a + c
    mz_det <- 1 - mz_correlation^2
    dz_det <- 1 - dz_correlation^2
    if (e < E_FLOOR - 1e-12 || mz_det <= 0 || dz_det <= 0) return(Inf)

    trace_sum <- (n_mz - 1) *
      (mz_diagonal - 2 * mz_correlation * mz_off_diagonal) / mz_det +
      (n_dz - 1) *
      (dz_diagonal - 2 * dz_correlation * dz_off_diagonal) / dz_det
    n_mz * log(mz_det) + n_dz * log(dz_det) + trace_sum
  }

  function(value) {
    if (!is.finite(value) || value <= -1 || value > 1 - E_FLOOR) {
      return(Inf)
    }
    upper_nuisance <- 1 - E_FLOOR - value
    if (parameter == "A") {
      # When profiling an unrestricted negative A, keep the DZ covariance PD.
      upper_nuisance <- min(upper_nuisance, 1 - E_FLOOR - 0.5 * value)
      at_nuisance <- function(nuisance) objective(value, nuisance)
    } else {
      at_nuisance <- function(nuisance) objective(nuisance, value)
    }
    if (upper_nuisance <= 0) return(at_nuisance(0))
    optimum <- optimize(at_nuisance, c(0, upper_nuisance), tol = 1e-9)
    min(at_nuisance(0), at_nuisance(upper_nuisance), optimum$objective)
  }
}

fixed_total_fit_objective <- function(
    obs_mz, obs_dz, n_mz, n_dz, a, c) {
  mz_corr <- a + c
  dz_corr <- 0.5 * a + c
  mz_det <- 1 - mz_corr^2
  dz_det <- 1 - dz_corr^2
  trace_sum <- (n_mz - 1) *
    (sum(diag(obs_mz)) - 2 * mz_corr * obs_mz[1L, 2L]) / mz_det +
    (n_dz - 1) *
    (sum(diag(obs_dz)) - 2 * dz_corr * obs_dz[1L, 2L]) / dz_det
  n_mz * log(mz_det) + n_dz * log(dz_det) + trace_sum
}

wald_adjusted_interval <- function(profile, estimate, fitted_objective) {
  if (!is.finite(estimate) || estimate < 0 || estimate > 1 - E_FLOOR) {
    stop("Invalid estimate for boundary-adjusted interval")
  }
  # OpenMx can return a tiny positive value for a component at its zero bound.
  at_boundary <- estimate <= 1e-10
  profile_estimate <- if (at_boundary) 0 else estimate
  at_estimate <- profile(profile_estimate)
  if (!is.finite(at_estimate) ||
      abs(at_estimate - fitted_objective) > 1e-2) {
    stop("Profile likelihood does not agree with the OpenMx optimum")
  }
  baseline <- min(at_estimate, fitted_objective)
  cutoff_95 <- qchisq(0.95, df = 1L)
  cutoff_90 <- qchisq(0.90, df = 1L)
  difference <- function(value) profile(value) - baseline
  upper_cutoff <- cutoff_95
  if (at_boundary) {
    # Equation (11): include the distance between the unrestricted optimum
    # and the constrained boundary fit when the latter is at zero.
    unrestricted <- optimize(profile, c(-1 + 1e-8, 0), tol = 1e-8)
    unrestricted_objective <- min(profile(0), unrestricted$objective)
    gap <- max(0, baseline - unrestricted_objective)
    upper_cutoff <- (sqrt(gap) + sqrt(cutoff_95))^2 - gap
  }
  upper_endpoint <- 1 - E_FLOOR
  if (!is.finite(difference(upper_endpoint))) {
    stop("Could not bracket the upper profile limit")
  }
  upper <- if (difference(upper_endpoint) <= upper_cutoff) {
    upper_endpoint
  } else {
    uniroot(
      function(value) difference(value) - upper_cutoff,
      c(profile_estimate, upper_endpoint), tol = 1e-9
    )$root
  }
  if (at_boundary) return(c(lower = 0, upper = upper))

  boundary_distance <- difference(0)
  lower_limit <- function(cutoff) {
    if (boundary_distance <= cutoff) return(0)
    uniroot(
      function(value) difference(value) - cutoff,
      c(0, estimate), tol = 1e-9
    )$root
  }
  lower_95 <- lower_limit(cutoff_95)
  lower_90 <- lower_limit(cutoff_90)
  midpoint <- lower_limit(boundary_distance / 4)
  lower <- if (midpoint <= lower_95) {
    lower_95
  } else if (midpoint <= lower_90) {
    midpoint
  } else {
    lower_90
  }
  c(lower = lower, upper = upper)
}

status_code_for <- function(fit) {
  code <- tryCatch(fit$output$status$code, error = function(e) NULL)
  if (is.null(code) || length(code) != 1L) {
    code <- tryCatch(fit$output$status[[1L]], error = function(e) NA_integer_)
  }
  as.integer(code)
}

paired <- read.csv(input_file, check.names = FALSE)
required <- c(
  "condition_id", "N_pairs", "A_true", "C_true", "E_true",
  "mz_var1", "mz_cov", "mz_var2", "dz_var1", "dz_cov", "dz_var2"
)
missing <- setdiff(required, names(paired))
if (length(missing) > 0L) {
  stop(sprintf("Input is missing columns: %s", paste(missing, collapse = ", ")))
}

fit_one <- function(i) {
  row <- paired[i, , drop = FALSE]
  n_pairs <- as.integer(row$N_pairs)
  dimnames_2 <- list(c("T1", "T2"), c("T1", "T2"))
  observed_mz <- matrix(
    c(row$mz_var1, row$mz_cov, row$mz_cov, row$mz_var2),
    nrow = 2L, byrow = TRUE, dimnames = dimnames_2
  )
  observed_dz <- matrix(
    c(row$dz_var1, row$dz_cov, row$dz_cov, row$dz_var2),
    nrow = 2L, byrow = TRUE, dimnames = dimnames_2
  )

  tryCatch({
    fit <- suppressMessages(mxRun(
      build_ace_model(observed_mz, observed_dz, n_pairs, n_pairs),
      silent = TRUE
    ))
    va <- as.numeric(mxEval(VA, fit))
    q <- as.numeric(mxEval(Q, fit))
    vc <- as.numeric(mxEval(VC, fit))
    ve <- as.numeric(mxEval(VE, fit))
    estimates <- c(A = va, C = vc, E = ve)
    ses <- fixed_total_se(va, q, fit$output$vcov)
    code <- status_code_for(fit)
    intervals <- matrix(
      NA_real_, nrow = 3L, ncol = 2L,
      dimnames = list(c("A", "C", "E"), c("lower", "upper"))
    )
    interval_errors <- character(0)
    if (isTRUE(code == 0L)) {
      fitted_objective <- fixed_total_fit_objective(
        observed_mz, observed_dz, n_pairs, n_pairs, va, vc
      )
      for (parameter in c("A", "C")) {
        interval <- tryCatch({
          profile <- make_fixed_total_profile(
            observed_mz, observed_dz, n_pairs, n_pairs, parameter
          )
          wald_adjusted_interval(
            profile, estimates[[parameter]], fitted_objective
          )
        }, error = function(error) {
          interval_errors <<- c(
            interval_errors, paste(parameter, conditionMessage(error), sep = ": ")
          )
          c(lower = NA_real_, upper = NA_real_)
        })
        intervals[parameter, ] <- interval
      }
    }
    intervals["E", ] <- c(
      estimates[["E"]] - 1.96 * ses[["E"]],
      estimates[["E"]] + 1.96 * ses[["E"]]
    )

    data.frame(
      condition_id = row$condition_id,
      N_pairs = n_pairs,
      A_openmx_est = estimates[["A"]],
      C_openmx_est = estimates[["C"]],
      E_openmx_est = estimates[["E"]],
      A_openmx_se = ses[["A"]],
      C_openmx_se = ses[["C"]],
      E_openmx_se = ses[["E"]],
      A_openmx_ci_lo = intervals["A", "lower"],
      A_openmx_ci_hi = intervals["A", "upper"],
      C_openmx_ci_lo = intervals["C", "lower"],
      C_openmx_ci_hi = intervals["C", "upper"],
      E_openmx_ci_lo = intervals["E", "lower"],
      E_openmx_ci_hi = intervals["E", "upper"],
      openmx_model = OPENMX_MODEL,
      openmx_ci_error = paste(interval_errors, collapse = "; "),
      openmx_converged = as.integer(isTRUE(code == 0L)),
      openmx_status_code = code,
      openmx_error = "",
      stringsAsFactors = FALSE
    )
  }, error = function(error) {
    data.frame(
      condition_id = row$condition_id,
      N_pairs = n_pairs,
      A_openmx_est = NA_real_, C_openmx_est = NA_real_, E_openmx_est = NA_real_,
      A_openmx_se = NA_real_, C_openmx_se = NA_real_, E_openmx_se = NA_real_,
      A_openmx_ci_lo = NA_real_, A_openmx_ci_hi = NA_real_,
      C_openmx_ci_lo = NA_real_, C_openmx_ci_hi = NA_real_,
      E_openmx_ci_lo = NA_real_, E_openmx_ci_hi = NA_real_,
      openmx_model = OPENMX_MODEL,
      openmx_ci_error = conditionMessage(error),
      openmx_converged = 0L,
      openmx_status_code = NA_integer_,
      openmx_error = conditionMessage(error),
      stringsAsFactors = FALSE
    )
  })
}

results <- vector("list", nrow(paired))
for (i in seq_len(nrow(paired))) {
  results[[i]] <- fit_one(i)
  if (i %% max(1L, floor(nrow(paired) / 20L)) == 0L || i == nrow(paired)) {
    cat(sprintf("OpenMx: %d / %d fits complete\n", i, nrow(paired)))
  }
}

results <- do.call(rbind, results)
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
write.csv(results, output_file, row.names = FALSE, na = "")
cat(sprintf(
  "Saved %d OpenMx fits to %s (%.1f%% converged)\n",
  nrow(results), output_file, 100 * mean(results$openmx_converged == 1L)
))
