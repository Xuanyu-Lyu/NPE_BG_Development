# ============================================================================
# STEP 04 — 04_fit_openmx_reference.R
#
# Produces the classical-SEM reference that STEP 06 compares the NPE against,
# and the shared test conditions that STEP 05 also fits.
#
# Purpose:
#   1. Draw 100 random ACE conditions from a symmetric Dirichlet(1,1,1)
#      distribution so that A + C + E = 1.
#   2. Compute theoretical (noise-free) MZ and DZ covariance matrices and
#      save them in a conditions CSV whose columns match ace_training_data.csv.
#   3. For each condition × sample size (N_MZ = N_DZ = 50/100/200/500/1000/2000/20000):
#        a. Simulate bivariate-normal MZ and DZ twin pair data.
#        b. Compute sample covariance matrices.
#        c. Fit the ACE model in OpenMx using covariance-matrix input.
#        d. Extract unstandardised (VA, VC, VE) and standardised (A, C, E)
#           estimates together with their standard errors.
#   4. Save two output files:
#        - data/ace_test_conditions.csv                  (true ACE + covariances)
#        - results/simulations/ace_simulation_results.csv (one row per
#                                                          condition × N)
#
# ACE model (biometric twin model):
#   MZ: Var = A+C+E,  Cov = A+C       (100 % genetic sharing)
#   DZ: Var = A+C+E,  Cov = 0.5*A+C  (50 % genetic sharing)
#
# Notes
#   - Theoretical covariances are used in the conditions file (N_pairs = NA).
#   - SEs for standardised components are derived via the delta method using
#     the parameter covariance matrix from the Hessian (fit$output$vcov).
#   - On convergence failure all estimates are set to NA; status_code is
#     recorded for inspection.
# ============================================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(OpenMx)
  library(MASS)     # mvrnorm
})

# ---- Global OpenMx options (compute Hessian & SEs for every run) -----------
mxOption(NULL, "Standard Errors",    "Yes")
mxOption(NULL, "Calculate Hessian",  "Yes")

set.seed(2025)

# ============================================================================
# Configuration  -- edit these paths if needed
# ============================================================================
N_CONDITIONS    <- 200L
SAMPLE_SIZES    <- c(50L, 100L, 200L, 500L, 1000L, 2000L, 20000L)  # N_MZ = N_DZ

# Paths are resolved relative to this script's folder, mirroring the Python
# side: inputs/test conditions live in data/, generated results in results/.
SCRIPT_DIR       <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) getwd()
)
DATA_DIR         <- file.path(SCRIPT_DIR, "data")
SIM_DIR          <- file.path(SCRIPT_DIR, "results", "simulations")
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(SIM_DIR,  showWarnings = FALSE, recursive = TRUE)

CONDITIONS_FILE  <- file.path(DATA_DIR, "ace_test_conditions.csv")
RESULTS_FILE     <- file.path(SIM_DIR,  "ace_simulation_results.csv")


# ============================================================================
# Helper functions
# ============================================================================

#' Draw one sample from Dirichlet(alpha, alpha, alpha) distribution.
rdirichlet3 <- function(alpha = 1) {
  x <- rgamma(3L, shape = alpha, rate = 1)
  x / sum(x)
}

#' Build a two-group ACE OpenMx model using covariance-matrix input.
#'
#' Parameters are estimated in the VARIANCE-COMPONENT (VC) parameterisation:
#'   VA, VC, VE >= 0 (VE >= 1e-6) are free.
#'
#' Method-of-moments starting values based on the observed covariances give
#' reliable convergence across the full simplex.
#'
#' @param obs_mz  2x2 named sample covariance matrix for MZ pairs.
#' @param obs_dz  2x2 named sample covariance matrix for DZ pairs.
#' @param N_MZ    Number of MZ twin pairs.
#' @param N_DZ    Number of DZ twin pairs.
#' @return An (unfitted) MxModel.
build_ace_model <- function(obs_mz, obs_dz, N_MZ, N_DZ) {

  sel_vars <- c("T1", "T2")

  # ---- Method-of-moments starting values (analytical inverses) ------------
  cMZ  <- obs_mz[1L, 2L]
  cDZ  <- obs_dz[1L, 2L]
  var_ <- obs_mz[1L, 1L]

  sv_A <- max(1e-4, 2 * (cMZ - cDZ))   # VA  =  2*(cMZ - cDZ)
  sv_C <- max(1e-4, 2 * cDZ - cMZ)     # VC  =  2*cDZ - cMZ
  sv_E <- max(1e-4, var_ - cMZ)        # VE  =  Var - cMZ

  # ---- Free variance components (shared between MZ and DZ groups) ----------
  covA <- mxMatrix("Full", 1L, 1L, free = TRUE,
                   values = sv_A, lbound = 0,    label = "VA11", name = "VA")
  covC <- mxMatrix("Full", 1L, 1L, free = TRUE,
                   values = sv_C, lbound = 0,    label = "VC11", name = "VC")
  covE <- mxMatrix("Full", 1L, 1L, free = TRUE,
                   values = sv_E, lbound = 1e-6, label = "VE11", name = "VE")

  # ---- Shared algebra: total phenotypic variance ---------------------------
  algV <- mxAlgebra(VA + VC + VE, name = "V")

  # ---- MZ-specific algebras and data ----------------------------------------
  algCovMZ    <- mxAlgebra(VA + VC, name = "cMZ")
  algExpCovMZ <- mxAlgebra(
    rbind(cbind(V, cMZ), cbind(t(cMZ), V)),
    name = "expCovMZ"
  )
  dataMZ  <- mxData(observed = obs_mz, type = "cov", numObs = N_MZ)
  expMZ   <- mxExpectationNormal(covariance = "expCovMZ", dimnames = sel_vars)

  # ---- DZ-specific algebras and data ----------------------------------------
  algCovDZ    <- mxAlgebra(0.5 %x% VA + VC, name = "cDZ")
  algExpCovDZ <- mxAlgebra(
    rbind(cbind(V, cDZ), cbind(t(cDZ), V)),
    name = "expCovDZ"
  )
  dataDZ  <- mxData(observed = obs_dz, type = "cov", numObs = N_DZ)
  expDZ   <- mxExpectationNormal(covariance = "expCovDZ", dimnames = sel_vars)

  funML <- mxFitFunctionML()

  # ---- Shared parameters list (duplicated into each sub-model) -------------
  pars <- list(covA, covC, covE, algV)

  modelMZ <- mxModel(pars, algCovMZ, algExpCovMZ, dataMZ, expMZ, funML,
                     name = "MZ")
  modelDZ <- mxModel(pars, algCovDZ, algExpCovDZ, dataDZ, expDZ, funML,
                     name = "DZ")
  multi   <- mxFitFunctionMultigroup(c("MZ", "DZ"))

  # ---- Top-level standardised-component algebras (for delta-method SEs) ----
  algA <- mxAlgebra(VA / V, name = "A_std")
  algC <- mxAlgebra(VC / V, name = "C_std")
  algE <- mxAlgebra(VE / V, name = "E_std")

  mxModel("ACEvc", pars, algA, algC, algE, modelMZ, modelDZ, multi)
}


#' Compute delta-method SEs for standardised ACE using the full parameter
#' covariance matrix (fit$output$vcov).
#'
#' @param VA_est, VC_est, VE_est  Point estimates of variance components.
#' @param vcov_mat  Covariance matrix of all free parameters (row/colnames =
#'                  OpenMx parameter labels, e.g. "VA11").
#' @return Named numeric(3): A_se, C_se, E_se. Returns NA on failure.
delta_se_std <- function(VA_est, VC_est, VE_est, vcov_mat) {
  na_out <- setNames(rep(NA_real_, 3L), c("A_se", "C_se", "E_se"))

  if (is.null(vcov_mat) || !is.matrix(vcov_mat)) return(na_out)

  V <- VA_est + VC_est + VE_est
  if (!is.finite(V) || V <= 0) return(na_out)

  rn    <- rownames(vcov_mat)
  i_VA  <- which(rn == "VA11")
  i_VC  <- which(rn == "VC11")
  i_VE  <- which(rn == "VE11")
  if (!all(lengths(list(i_VA, i_VC, i_VE)) == 1L)) return(na_out)

  # Jacobian of (A_std, C_std, E_std) with respect to (VA, VC, VE)
  # d(VA/V)/d(VA) = (V - VA)/V^2 = (VC + VE)/V^2,  rest = -VA/V^2
  J <- matrix(0, 3L, 3L)
  J[1L, ] <- c( V - VA_est, -VA_est,  -VA_est ) / V^2
  J[2L, ] <- c(-VC_est,  V - VC_est,  -VC_est ) / V^2
  J[3L, ] <- c(-VE_est,    -VE_est, V - VE_est ) / V^2

  idx      <- c(i_VA, i_VC, i_VE)
  cov_sub  <- vcov_mat[idx, idx, drop = FALSE]

  cov_std  <- J %*% cov_sub %*% t(J)
  se_vals  <- sqrt(pmax(diag(cov_std), 0))
  setNames(se_vals, c("A_se", "C_se", "E_se"))
}


# ============================================================================
# STEP 1 — Generate 100 ACE conditions with theoretical covariances
# ============================================================================
cat("Generating", N_CONDITIONS, "ACE conditions...\n")

conditions <- data.frame(
  condition_id = seq_len(N_CONDITIONS),
  A            = numeric(N_CONDITIONS),
  C            = numeric(N_CONDITIONS),
  E            = numeric(N_CONDITIONS),
  mz_var       = numeric(N_CONDITIONS),
  mz_cov       = numeric(N_CONDITIONS),
  dz_var       = numeric(N_CONDITIONS),
  dz_cov       = numeric(N_CONDITIONS),
  N_pairs      = NA_real_          # theoretical; no sampling noise
)

for (i in seq_len(N_CONDITIONS)) {
  ace <- rdirichlet3()
  A   <- ace[1L]; C <- ace[2L]; E <- ace[3L]
  V   <- A + C + E  # exactly 1 by construction

  conditions[i, "A"]       <- A
  conditions[i, "C"]       <- C
  conditions[i, "E"]       <- E
  conditions[i, "mz_var"]  <- V
  conditions[i, "mz_cov"]  <- A + C
  conditions[i, "dz_var"]  <- V
  conditions[i, "dz_cov"]  <- 0.5 * A + C
}

write.csv(conditions, CONDITIONS_FILE, row.names = FALSE)
cat("  -> Saved:", CONDITIONS_FILE,
    sprintf("(%d rows)\n\n", nrow(conditions)))


# ============================================================================
# STEP 2 — Simulate and fit ACE model for every condition × sample size
# ============================================================================
total    <- N_CONDITIONS * length(SAMPLE_SIZES)
results  <- vector("list", total)
counter  <- 0L

cat(sprintf(
  "Fitting ACE models: %d conditions x %d sample sizes = %d fits\n\n",
  N_CONDITIONS, length(SAMPLE_SIZES), total
))

for (i in seq_len(N_CONDITIONS)) {

  A_true <- conditions$A[i]
  C_true <- conditions$C[i]
  E_true <- conditions$E[i]
  V_true <- A_true + C_true + E_true  # = 1

  # Population covariance matrices (dimnames required by OpenMx)
  dn <- list(c("T1", "T2"), c("T1", "T2"))
  Sigma_MZ <- matrix(
    c(V_true, A_true + C_true, A_true + C_true, V_true),
    2L, 2L, dimnames = dn
  )
  Sigma_DZ <- matrix(
    c(V_true, 0.5 * A_true + C_true, 0.5 * A_true + C_true, V_true),
    2L, 2L, dimnames = dn
  )

  for (N in SAMPLE_SIZES) {
    counter <- counter + 1L

    base_row <- list(
      condition_id = i,
      sample_size  = N,
      true_A       = A_true,
      true_C       = C_true,
      true_E       = E_true
    )

    fit_row <- tryCatch({

      # ---- Simulate twin pair data ----------------------------------------
      mz_data <- mvrnorm(N, mu = c(0, 0), Sigma = Sigma_MZ)
      dz_data <- mvrnorm(N, mu = c(0, 0), Sigma = Sigma_DZ)
      colnames(mz_data) <- colnames(dz_data) <- c("T1", "T2")

      obs_cov_mz <- cov(mz_data)
      obs_cov_dz <- cov(dz_data)

      # ---- Build and run OpenMx ACE model ---------------------------------
      model <- build_ace_model(obs_cov_mz, obs_cov_dz, N_MZ = N, N_DZ = N)
      fit   <- suppressMessages(mxRun(model, silent = TRUE))

      # ---- Extract unstandardised variance component estimates ------------
      VA_est <- fit$VA$values[1L, 1L]
      VC_est <- fit$VC$values[1L, 1L]
      VE_est <- fit$VE$values[1L, 1L]
      V_est  <- VA_est + VC_est + VE_est

      # ---- Standardised estimates (proportions) ---------------------------
      A_est  <- VA_est / V_est
      C_est  <- VC_est / V_est
      E_est  <- VE_est / V_est

      # ---- SEs for unstandardised components (from parameter table) -------
      params <- summary(fit)$parameters
      get_se <- function(lbl) {
        se <- params$Std.Error[params$name == lbl]
        if (length(se) == 1L && is.finite(se)) se else NA_real_
      }
      VA_se <- get_se("VA11")
      VC_se <- get_se("VC11")
      VE_se <- get_se("VE11")

      # ---- SEs for standardised components (delta method) -----------------
      vcov_mat <- tryCatch(fit$output$vcov, error = function(e) NULL)
      std_se   <- delta_se_std(VA_est, VC_est, VE_est, vcov_mat)

      # ---- Convergence status ---------------------------------------------
      status_code <- tryCatch(
        as.integer(fit$output$status[[1L]]),
        error = function(e) NA_integer_
      )

      c(base_row, list(
        # Unstandardised variance components
        VA_est      = VA_est,
        VC_est      = VC_est,
        VE_est      = VE_est,
        VA_se       = VA_se,
        VC_se       = VC_se,
        VE_se       = VE_se,
        # Standardised proportions
        A_est       = A_est,
        C_est       = C_est,
        E_est       = E_est,
        A_se        = std_se["A_se"],
        C_se        = std_se["C_se"],
        E_se        = std_se["E_se"],
        # Convergence
        converged   = as.integer(isTRUE(status_code == 0L)),
        status_code = status_code
      ))

    }, error = function(e) {
      message(sprintf(
        "  [ERROR] condition %d, N = %d: %s", i, N, conditionMessage(e)
      ))
      c(base_row, list(
        VA_est = NA_real_, VC_est = NA_real_, VE_est = NA_real_,
        VA_se  = NA_real_, VC_se  = NA_real_, VE_se  = NA_real_,
        A_est  = NA_real_, C_est  = NA_real_, E_est  = NA_real_,
        A_se   = NA_real_, C_se   = NA_real_, E_se   = NA_real_,
        converged   = 0L,
        status_code = NA_integer_
      ))
    })

    results[[counter]] <- fit_row

    if (counter %% 60L == 0L || counter == total) {
      cat(sprintf("  [%4d / %d]  condition %3d  N = %4d\n",
                  counter, total, i, N))
    }
  }  # end sample-size loop
}  # end condition loop


# ============================================================================
# STEP 3 — Save results
# ============================================================================
results_df <- do.call(
  rbind,
  lapply(results, function(r) as.data.frame(r, stringsAsFactors = FALSE))
)

write.csv(results_df, RESULTS_FILE, row.names = FALSE)

cat("\n========================================\n")
cat("Done.\n")
cat("  Conditions file : ", CONDITIONS_FILE,
    sprintf("  (%d rows)\n", nrow(conditions)))
cat("  Results file    : ", RESULTS_FILE,
    sprintf("  (%d rows)\n", nrow(results_df)))
cat("  Convergence rate: ",
    sprintf("%.1f%%\n",
            100 * mean(results_df$converged == 1L, na.rm = TRUE)))
cat("========================================\n")
