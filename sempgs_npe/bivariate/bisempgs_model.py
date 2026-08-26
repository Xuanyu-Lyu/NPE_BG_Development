"""BiSEMPGS (Bivariate SEM-PGS) core module.

Computes the theoretical 14×14 covariance matrix (CMatrix) for the bivariate
SEM-PGS model (model 2, fixed mate-correlation scheme) and provides utilities
to simulate sample covariances from it.

Variable order in CMatrix (14 variables, 2×2 blocks):
  Yp1, Yp2   – paternal-side parent phenotypes (traits 1 & 2)
  Ym1, Ym2   – maternal-side mate phenotypes
  Yo1, Yo2   – offspring phenotypes
  Tp1, Tp2   – paternal transmitted PGS
  NTp1, NTp2 – paternal non-transmitted PGS
  Tm1, Tm2   – maternal transmitted PGS
  NTm1, NTm2 – maternal non-transmitted PGS

This mirrors the algebra in the R script 11-MVN_rerun_ss.R and the Python
iterative_math() function in generate_sempgs_data.py.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CMATRIX_VARS = [
    "Yp1", "Yp2",
    "Ym1", "Ym2",
    "Yo1", "Yo2",
    "Tp1", "Tp2",
    "NTp1", "NTp2",
    "Tm1", "Tm2",
    "NTm1", "NTm2",
]

N_VARS   = len(CMATRIX_VARS)  # 14

# ---------------------------------------------------------------------------
# Unique-block specification
# ---------------------------------------------------------------------------
# The 14×14 CMatrix is composed of 2×2 blocks, many of which are identical
# across multiple positions.  Only 12 blocks carry unique information:
#   3 symmetric blocks  → 3 elements each  =  9
#   9 general blocks    → 4 elements each  = 36
#                                      total = 45
#
# Each entry: (name, row_start, col_start, is_symmetric)
# Row/col indices refer to the 14×14 CMatrix (one representative position).
_UNIQUE_BLOCKS = [
    # ---- diagonal representatives (symmetric 2×2) ----
    ("VY",      0,  0,  True),   # Cov(Yp,Yp) = Cov(Ym,Ym) = Cov(Yo,Yo)
    ("kgc",     6,  6,  True),   # k+gc; Cov(Tp,Tp) = Cov(NTp,NTp) = Cov(Tm,Tm) = Cov(NTm,NTm)
    ("gc",      6,  8,  True),   # Cov(Tp,NTp) = Cov(Tm,NTm)
    # ---- off-diagonal representatives (general 2×2) ----
    ("YpYm",    0,  2,  False),  # Cov(Yp,Ym)
    ("YpYo",    0,  4,  False),  # Cov(Yp,Yo)  [Yo_Yp.T block]
    ("Omega",   0,  6,  False),  # Cov(Yp,Tp) = Cov(Yp,NTp) = Cov(Ym,Tm) = Cov(Ym,NTm)
    ("YpPGSm",  0, 10,  False),  # Cov(Yp,Tm) = Cov(Yp,NTm)
    ("YmYo",    2,  4,  False),  # Cov(Ym,Yo)  [Yo_Ym.T block]
    ("YmPGSp",  2,  6,  False),  # Cov(Ym,Tp) = Cov(Ym,NTp)
    ("thetaT",  4,  6,  False),  # Cov(Yo,Tp) = Cov(Yo,Tm)
    ("thetaNT", 4,  8,  False),  # Cov(Yo,NTp) = Cov(Yo,NTm)
    ("gt",      6, 10,  False),  # Cov(Tp,Tm) = Cov(Tp,NTm) = Cov(NTp,Tm) = Cov(NTp,NTm)
]

N_UNIQUE_FEATURES = sum(3 if sym else 4 for _, _, _, sym in _UNIQUE_BLOCKS)  # 45


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

def compute_cmatrix(
    vg1, vg2, rg, re,
    prop_h2_latent1, prop_h2_latent2,
    am11, am12, am21, am22,
    f11, f12, f21, f22,
    gens: int = 15,
):
    """Compute the 14×14 theoretical covariance matrix for the bivariate
    SEM-PGS model (model 2, fixed mate-correlation scheme) after *gens*
    generations of assortative mating (AM) and vertical transmission (VT).

    Parameters
    ----------
    vg1, vg2 : float
        Total SNP heritability for trait 1 and trait 2 (0 < vg < 1).
    rg : float
        Genetic correlation between the two traits (-1 < rg < 1).
    re : float
        Residual / environmental correlation (-1 < re < 1).
    prop_h2_latent1, prop_h2_latent2 : float
        Proportion of h² explained by the *latent* (unobserved) PGS
        for each trait (0 < prop < 1).
    am11, am22 : float
        Within-trait assortative mating correlations for traits 1 and 2.
    am12, am21 : float
        Cross-trait AM correlations  (Yp1–Ym2 and Yp2–Ym1).
    f11, f22 : float
        Direct vertical transmission coefficients (parent trait → same offspring
        trait).
    f12, f21 : float
        Cross vertical transmission coefficients.
    gens : int
        Number of generations to iterate (default 15).

    Returns
    -------
    CMatrix : ndarray(14, 14) or None
        Theoretical covariance matrix.  Returns *None* if the parameters
        yield a non-positive-definite matrix or a numerical error.
    """
    # ------------------------------------------------------------------ #
    # Base matrices at t = 0
    # ------------------------------------------------------------------ #
    k2 = np.array([[1., rg], [rg, 1.]])   # 2 × haplotypic genetic cor. matrix

    # Observed (genome-wide-significant) PGS component — delta matrix
    vg_obs1 = vg1 * (1. - prop_h2_latent1)
    vg_obs2 = vg2 * (1. - prop_h2_latent2)
    delta = np.array([[np.sqrt(max(vg_obs1, 0.)), 0.],
                      [0.,                         np.sqrt(max(vg_obs2, 0.))]])

    # Latent (unobserved) PGS component — a matrix
    vg_lat1 = vg1 * prop_h2_latent1
    vg_lat2 = vg2 * prop_h2_latent2
    a = np.array([[np.sqrt(max(vg_lat1, 0.)), 0.],
                  [0.,                         np.sqrt(max(vg_lat2, 0.))]])

    # Environmental covariance
    ve1, ve2 = 1. - vg1, 1. - vg2
    cove     = re * np.sqrt(max(ve1, 0.) * max(ve2, 0.))
    cove_mat = np.array([[ve1, cove], [cove, ve2]])

    # Phenotypic variance at t = 0
    COVY = delta @ k2 @ delta.T + a @ k2 @ a.T + cove_mat

    rmate = np.array([[am11, am12], [am21, am22]])  # mate correlation matrix
    f     = np.array([[f11,  f12 ], [f21,  f22 ]])  # vertical transmission

    j = k2 * 0.5  # haplotypic k-matrix (latent component)
    k = k2 * 0.5  # haplotypic k-matrix (observed PGS component)

    # ------------------------------------------------------------------ #
    # Helper: compute mu via fixed mate-correlation scheme
    # ------------------------------------------------------------------ #
    def _make_mu(VY_cur):
        """mu = inv(VY) @ mate_cov @ inv(VY^T), with mate_cov derived from
        a fixed mate-correlation matrix rmate and the current VY variance."""
        d  = np.sqrt(np.diag(np.diag(VY_cur)))  # diag(sqrt(var1), sqrt(var2))
        mc = d @ rmate @ d                        # mate covariance matrix
        try:
            return np.linalg.solve(VY_cur, mc) @ np.linalg.inv(VY_cur.T)
        except np.linalg.LinAlgError:
            return None

    # ------------------------------------------------------------------ #
    # Initialise iterative state variables (t = 0 values)
    # ------------------------------------------------------------------ #
    gc = np.zeros((2, 2))
    hc = np.zeros((2, 2))
    ic = np.zeros((2, 2))
    w  = np.zeros((2, 2))
    v  = np.zeros((2, 2))
    VY = COVY.copy()
    VF = 2. * (f @ COVY @ f.T)

    mu = _make_mu(VY)
    if mu is None:
        return None

    # Track these for the CMatrix construction
    gt    = np.zeros((2, 2))
    Omega = np.zeros((2, 2))
    Gamma = np.zeros((2, 2))

    # ------------------------------------------------------------------ #
    # Iterative update  (gens − 1 steps, matching the R loop for it = 2..gens)
    # ------------------------------------------------------------------ #
    for _ in range(gens - 1):
        Omega = 2*delta@gc + delta@k + 0.5*w + 2*a@ic
        Gamma = 2*a@hc + 2*delta@ic.T + a@j + 0.5*v

        VY_new = (2*delta@Omega.T + 2*a@Gamma.T
                  + w@delta.T + v@a.T + VF + cove_mat)

        mu = _make_mu(VY_new)
        if mu is None:
            return None

        gt_new = Omega.T @ mu @ Omega
        gc     = 0.5 * (gt_new + gt_new.T)
        gt     = gt_new                        # kept un-symmetrised for CMatrix

        ht = Gamma.T @ mu @ Gamma
        hc = 0.5 * (ht + ht.T)

        w  = 2*f@Omega + f@VY_new@mu@Omega + f@VY_new@mu.T@Omega
        v  = 2*f@Gamma + f@VY_new@mu@Gamma + f@VY_new@mu.T@Gamma

        VF = (2*f@VY_new@f.T
              + f@VY_new@mu@VY_new@f.T
              + f@VY_new@mu.T@VY_new@f.T)

        itlo = Gamma.T @ mu @ Omega
        itol = Omega.T @ mu @ Gamma
        ic   = 0.5 * (itlo + itol.T)

        VY = VY_new

    # ------------------------------------------------------------------ #
    # Build the 14×14 CMatrix from final-generation quantities
    # (mirrors the CMatrix construction in 11-MVN_rerun_ss.R)
    # ------------------------------------------------------------------ #
    thetaNT = 2*delta@gc + 2*a@ic + 0.5*w
    thetaT  = delta@k + thetaNT

    Yp_PGSm = VY @ mu @ Omega          # Cov(Yp,  T_mother) = Cov(Yp, NTm)
    Ym_PGSp = VY @ mu.T @ Omega        # Cov(Ym,  T_father) = Cov(Ym, NTp)
    Yp_Ym   = VY @ mu @ VY             # Cov(Yp,  Ym)
    Ym_Yp   = VY @ mu.T @ VY           # Cov(Ym,  Yp)

    Yo_Yp = (delta @ Omega.T + a @ Gamma.T
             + delta @ Omega.T @ mu.T @ VY + a @ Gamma.T @ mu.T @ VY
             + f @ VY + f @ VY @ mu.T @ VY)          # Cov(Yo, Yp)

    Yo_Ym = (delta @ Omega.T + a @ Gamma.T
             + delta @ Omega.T @ mu @ VY + a @ Gamma.T @ mu @ VY
             + f @ VY + f @ VY @ mu @ VY)             # Cov(Yo, Ym)

    # fmt: off
    CMatrix = np.block([
        #            Yp         Ym          Yo           Tp        NTp       Tm         NTm
        [VY,         Yp_Ym,     Yo_Yp.T,    Omega,       Omega,    Yp_PGSm,  Yp_PGSm ],  # Yp
        [Ym_Yp,      VY,        Yo_Ym.T,    Ym_PGSp,     Ym_PGSp,  Omega,    Omega   ],  # Ym
        [Yo_Yp,      Yo_Ym,     VY,          thetaT,      thetaNT,  thetaT,   thetaNT ],  # Yo
        [Omega.T,    Ym_PGSp.T, thetaT.T,    k + gc,      gc,       gt,       gt      ],  # Tp
        [Omega.T,    Ym_PGSp.T, thetaNT.T,   gc,          k + gc,   gt,       gt      ],  # NTp
        [Yp_PGSm.T,  Omega.T,   thetaT.T,    gt.T,        gt.T,     k + gc,   gc      ],  # Tm
        [Yp_PGSm.T,  Omega.T,   thetaNT.T,   gt.T,        gt.T,     gc,       k + gc  ],  # NTm
    ])
    # fmt: on

    # ------------------------------------------------------------------ #
    # Validate positive definiteness
    # ------------------------------------------------------------------ #
    try:
        eigvals = np.linalg.eigvalsh(CMatrix)
    except np.linalg.LinAlgError:
        return None

    if np.any(np.isnan(eigvals)) or eigvals.min() < -1e-8:
        return None

    return CMatrix


# ---------------------------------------------------------------------------
# Simulation utilities
# ---------------------------------------------------------------------------

def simulate_sample_cov(cmatrix, n_obs: int):
    """Draw *n_obs* samples from MVN(0, cmatrix) and return the 14×14 sample
    covariance matrix.

    Parameters
    ----------
    cmatrix : ndarray(14, 14)
        Theoretical covariance matrix (positive definite).
    n_obs : int
        Number of observations to simulate.

    Returns
    -------
    ndarray(14, 14)  – sample covariance matrix.
    """
    n       = cmatrix.shape[0]
    samples = np.random.multivariate_normal(np.zeros(n), cmatrix, size=n_obs)
    return np.cov(samples, rowvar=False)


# ---------------------------------------------------------------------------
# Feature extraction — 45 unique elements
# ---------------------------------------------------------------------------

def unique_elements(cmatrix):
    """Extract the 45 truly unique elements from the 14×14 CMatrix.

    Many 2×2 blocks are repeated across multiple positions in the CMatrix
    (e.g. Omega appears 4 times, gt appears 4 times, etc.).  This function
    reads each unique block exactly once:
      - symmetric 2×2 blocks  → 3 elements  [row0col0, row0col1, row1col1]
      - general   2×2 blocks  → 4 elements  [row0col0, row0col1, row1col0, row1col1]

    Returns
    -------
    ndarray(45,)
    """
    parts = []
    for _, r, c, sym in _UNIQUE_BLOCKS:
        block = cmatrix[r:r+2, c:c+2]
        if sym:
            parts.extend([block[0, 0], block[0, 1], block[1, 1]])
        else:
            parts.extend(block.ravel())
    return np.array(parts, dtype=float)


def unique_feature_names():
    """Return the 45 column names that correspond to `unique_elements()`.

    Naming convention:
      symmetric blocks  → "<block>_11", "<block>_12", "<block>_22"
      general blocks    → "<block>_11", "<block>_12", "<block>_21", "<block>_22"
    """
    names = []
    for block_name, _, _, sym in _UNIQUE_BLOCKS:
        if sym:
            names += [f"{block_name}_11", f"{block_name}_12", f"{block_name}_22"]
        else:
            names += [f"{block_name}_11", f"{block_name}_12",
                      f"{block_name}_21", f"{block_name}_22"]
    return names
