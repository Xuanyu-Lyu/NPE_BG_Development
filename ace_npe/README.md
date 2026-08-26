# ACE × Neural Posterior Estimation

A validation study: can **neural posterior estimation (NPE)** recover the
parameters of the classical behaviour-genetic **ACE twin model** as well as a
conventional structural equation model fitted by maximum likelihood in OpenMx?

The ACE model is deliberately the simplest interesting case — three parameters,
a closed-form likelihood, and a mature reference implementation. If NPE cannot
match OpenMx here, it will not be trustworthy on the harder SEM-PGS models in
[`../sempgs_npe/`](../sempgs_npe/). This folder is the controlled experiment
that answers that question.

---

## Quick start

Every path below is relative to this folder. Scripts resolve their own
locations, so they can be run from anywhere.

```bash
# 01 — simulate training data          -> data/ace_training_data.csv
python 01_generate_training_data.py --n_samples 50000

# 02 — train the NPE                   -> results/models/se_proxy/
python 02_train_npe.py --data ace_training_data.csv --include_n_pairs \
                       --epochs 500 --device cpu --output se_proxy

# 03 — out-of-sample calibration check -> results/oos/se_proxy/
python 03_evaluate_oos_predictions.py --model_dir se_proxy

# 04 — OpenMx reference + shared test conditions
Rscript 04_fit_openmx_reference.R

# 05 — NPE on the same test conditions -> results/simulations/
python 05_simulate_posterior_recovery.py --model_dir se_proxy \
       --output npe_simulation_results.csv
python 05_simulate_posterior_recovery.py --model_dir no_n_pairs_gaussian_prior \
       --output npe_simulation_results_no_n.csv

# 06 — comparison figures and tables   -> results/analysis/
jupyter lab 06_analysis.ipynb

# 07 — is the reported SE correct?     -> results/se_calibration/
#      Monte-Carlo: replicate the SAME condition 200x, so the spread of the
#      estimates is the TRUE sampling SE to compare the posterior SD against.
python 05_simulate_posterior_recovery.py --model_dir se_proxy \
       --n_conditions 24 --n_reps 200 --output npe_se_montecarlo.csv
jupyter lab 07_se_calibration.ipynb
```

`demo_single_fit.ipynb` is a standalone illustration of fitting one dataset;
it is not a pipeline step.

---

## Layout

```
ace_npe/
├── ace_model.py                      shared library — imported by everything
├── 01_generate_training_data.py      simulate (θ, x) training pairs
├── 02_train_npe.py                   train the normalizing flow
├── 03_evaluate_oos_predictions.py    calibration / coverage on fresh draws
├── 04_fit_openmx_reference.R         OpenMx MLE reference + test conditions
├── 05_simulate_posterior_recovery.py NPE fits on those same conditions
├── 06_analysis.ipynb                 OpenMx vs NPE comparison
├── 07_se_calibration.ipynb           is the reported SE correct?
├── demo_single_fit.ipynb             worked single-observation example
├── data/                             inputs and shared test conditions
│   ├── ace_training_data*.csv
│   └── ace_test_conditions.csv
└── results/                          everything generated
    ├── models/<run>/                 posterior.pkl, config.json, scaler, metrics
    ├── simulations/                  OpenMx + NPE recovery results
    ├── analysis/                     STEP 06 figures and summary tables
    ├── se_calibration/               STEP 07 SE-calibration figures + tables
    ├── oos/<run>/                    STEP 03 calibration output
    └── demo/                         demo notebook figures
```

### Why `ace_model.py` is not numbered

Python cannot import a module whose name begins with a digit —
`from 02_train_npe import ...` is a syntax error. So **no numbered script is
ever imported by another**. Everything shared lives in `ace_model.py`:
the model math, the folder paths, the embedding-net class, and the posterior
helpers (`load_posterior`, `build_features`, `posterior_stats`,
`map_from_samples`). Numbered scripts are pure entry points.

---

## Part 1 — The ACE model

A phenotype is decomposed into three independent variance components:

| Term | Meaning |
|---|---|
| **A** | additive genetic variance |
| **C** | common (shared) environment, shared by both twins |
| **E** | unique environment + measurement error |

Monozygotic twins share 100 % of their segregating genes; dizygotic twins share
50 % in expectation. Both share the common environment entirely. With a
mean-centred phenotype this gives the two population covariance matrices that
carry the entire model:

$$
\Sigma_{MZ}=\begin{pmatrix} A+C+E & A+C \\ A+C & A+C+E\end{pmatrix},
\qquad
\Sigma_{DZ}=\begin{pmatrix} A+C+E & \tfrac{1}{2}A+C \\ \tfrac{1}{2}A+C & A+C+E\end{pmatrix}
$$

Identification is a three-equation linear system, which is why the model is a
good test case — the answer is known exactly:

$$
A = 2(c_{MZ}-c_{DZ}), \qquad C = 2c_{DZ}-c_{MZ}, \qquad E = v - c_{MZ}
$$

where $v$ is the phenotypic variance and $c_z$ the within-pair covariance.
`04_fit_openmx_reference.R` uses exactly these expressions as its
method-of-moments starting values.

Both estimators are given the same data-generating process, implemented in
`ace_model.simulate_covariances`: draw $N$ MZ and $N$ DZ pairs from
$\mathcal N(0,\Sigma_z)$, then compute the **sample** covariance matrices. The
sampling noise in those matrices is the entire inferential problem.

OpenMx consumes those 2×2 matrices whole. The NPE consumes
`ace_model.summarize_cov(S)` — the mean of the diagonal, plus the off-diagonal
— which §3.4 shows is the *sufficient* reduction, so no information is lost
relative to OpenMx and the two estimators are compared like-for-like.

---

## Part 2 — Why simulation-based inference at all

For the ACE model the likelihood is available in closed form, so classical ML
works fine — that is precisely why it makes a good benchmark. The motivation is
what happens *next*, on models where the likelihood is not tractable.

The two estimators differ structurally:

|  | OpenMx (ML) | NPE |
|---|---|---|
| Per-dataset cost | numerical optimisation, can fail to converge | one forward pass |
| Output | point estimate + delta-method SE | full joint posterior |
| Needs a likelihood | yes | no — only a simulator |
| Up-front cost | none | one training run |

NPE is **amortised**: all the cost is paid once during training, after which
inference on a new dataset is a single evaluation of a neural network. That is
the property worth buying, and this folder measures what it costs in accuracy.

---

## Part 3 — The neural architecture, in detail

### 3.1 What the network is actually trained to do

Write $\theta = (A, C, E)$ for the parameters and $x$ for the summary
statistics computed from one simulated study. Training data are pairs drawn
ancestrally,

$$
\theta_i \sim p(\theta), \qquad x_i \sim p(x \mid \theta_i),
$$

i.e. sample a parameter vector, run the simulator, keep both. The network is a
**conditional density estimator** $q_\phi(\theta \mid x)$ trained by maximising
the conditional log-likelihood of the parameters that actually produced each
dataset:

$$
\mathcal{L}(\phi) \;=\; \sum_i \log q_\phi(\theta_i \mid x_i)
$$

This simple objective recovers the true posterior. Taking the expectation and
decomposing:

$$
\mathbb{E}_{p(\theta,x)}\!\left[-\log q_\phi(\theta\mid x)\right]
= \mathbb{E}_{p(x)}\Big[\underbrace{\mathcal H\big(p(\theta\mid x)\big)}_{\text{constant in }\phi}
+ \operatorname{KL}\!\big(p(\theta\mid x)\,\|\,q_\phi(\theta\mid x)\big)\Big]
$$

The entropy term does not depend on $\phi$, so minimising the loss minimises an
expected KL divergence, which is zero exactly when
$q_\phi(\theta \mid x) = p(\theta \mid x)$ for almost every $x$. With enough
capacity and data the trained flow **is** the posterior — no MCMC, no
variational gap by construction. This is single-round `SNPE-C` (APT,
Greenberg et al. 2019) as implemented in [`sbi`](https://sbi-dev.github.io/sbi/);
with one round and proposal = prior, the APT correction term vanishes and the
objective reduces to the plain conditional maximum likelihood above.

### 3.2 Representing the posterior: a normalizing flow

$q_\phi(\theta \mid x)$ must be a *normalised* density over $\mathbb{R}^3$ that
we can both evaluate and sample. A normalizing flow gets this by pushing a
simple base density through an invertible map. Let
$f_\phi(\cdot\,; x): \mathbb{R}^3 \to \mathbb{R}^3$ be invertible for every $x$,
with base density $p_z = \mathcal N(0, I_3)$. The change-of-variables formula
gives

$$
\log q_\phi(\theta \mid x) \;=\; \log p_z\big(f_\phi(\theta; x)\big)
\;+\; \log\left|\det \frac{\partial f_\phi(\theta; x)}{\partial \theta}\right|
$$

Both terms must be cheap, which is what dictates the architecture: the map is
built from layers whose Jacobian is triangular, so the log-determinant is just
a sum of logs of diagonal entries.

**Neural Spline Flow (`nsf`)** is the default here. Each transform applies a
*monotonic rational-quadratic spline* elementwise. The spline is defined by $K$
bins over an interval $[-B, B]$; a conditioner network — which sees the
observation $x$ and the preceding components of $\theta$ — emits that bin's
widths, heights and boundary derivatives. Monotonicity guarantees
invertibility, and the rational-quadratic form has a closed-form inverse, so
sampling and density evaluation are both exact and fast.

### 3.3 Why a spline flow rather than a Gaussian or a mixture

The ACE posterior is genuinely awkward, and each pathology rules out a simpler
choice:

- **It is strongly correlated.** $A$ and $C$ are identified only through the
  *difference* $c_{MZ} - c_{DZ}$. When that difference is noisy — small $N$ —
  the posterior collapses onto a long diagonal ridge trading $A$ against $C$.
  A diagonal-Gaussian estimator cannot represent that ridge at all; the joint
  plots in `demo_single_fit.ipynb` show it directly.
- **It is skewed and boundary-constrained.** True values live on the simplex
  near $A, C, E \ge 0$. Posteriors for a component near zero pile up against
  the boundary and are sharply asymmetric — badly modelled by anything
  symmetric.
- **It is heteroscedastic in $N$.** Posterior width must shrink like
  $1/\sqrt{N}$. The flow has to modulate its whole shape as a function of the
  conditioning input, not merely shift its location.

A mixture density network (`mdn`) could capture some of this, but needs many
components to represent a smooth curved ridge and tends to produce
mode-collapsed or spiky fits. Splines model curved, bounded, skewed densities
with far fewer parameters. `--flow_type` still accepts `maf`, `maf_rqs` and
`mdn` if you want to compare.

### 3.4 Why there is **no embedding network**

This is the part of the design most worth understanding, because the code looks
like it has one and does not.

`ace_model.ACEEmbeddingNet` is defined and exported, but `02_train_npe.py`
passes `nn.Identity()` to the flow. The standardized summary statistics are the
conditioning vector, fed to the flow directly. **Every trained run in
`results/models/` was produced this way.**

An embedding network exists to *learn summary statistics* when $x$ is
high-dimensional and unstructured — raw images, time series, whole genotype
matrices. It compresses raw data into something a flow can condition on. Here
that job is already done, and done optimally, by the model itself.

Condition on the ACE model's own structure. The data are
$n$ i.i.d. mean-zero bivariate normal pairs per zygosity. Writing the
likelihood for one zygosity group,

$$
p(\text{data} \mid \Sigma) \;\propto\; |\Sigma|^{-n/2}
\exp\!\left(-\tfrac{n}{2}\operatorname{tr}\!\big(\Sigma^{-1} S\big)\right),
\qquad S=\tfrac{1}{n}\sum_{j} y_j y_j^{\top}
$$

The data enter **only** through $S$ and $n$. By the Fisher–Neyman factorization
theorem, $(S_{MZ}, S_{DZ}, n)$ is a *sufficient statistic* for
$(\Sigma_{MZ}, \Sigma_{DZ})$, hence for $(A, C, E)$.

That reduction goes one step further here, and the step matters. Twins are
*exchangeable*, so each population matrix is compound symmetric,
$\Sigma=\begin{pmatrix} v & c\\ c & v\end{pmatrix}$, and

$$
\operatorname{tr}\!\big(\Sigma^{-1}S\big)
= \frac{v\,(S_{11}+S_{22}) - 2c\,S_{12}}{v^{2}-c^{2}}
$$

The likelihood therefore depends on $S$ only through $S_{11}+S_{22}$ and
$S_{12}$ — **not** on the two diagonal entries separately. The sufficient
reduction of each 2×2 matrix is thus the *mean of its diagonal* paired with its
off-diagonal, which is exactly what `ace_model.summarize_cov` computes:

```python
var = 0.5 * (S[0, 0] + S[1, 1])     # both entries estimate the same v
cov = S[0, 1]
```

So the four features handed to the flow are the sufficient statistic, not
merely a convenient summary of it.

From there the argument closes by the data-processing inequality: a learned
embedding $g(x)$ satisfies $I(g(x); \theta) \le I(x; \theta)$. It cannot add
information; it can only lose it. Inserting a 4→64→64→32 MLP with dropout
between a sufficient statistic and the flow is, at best, an identity map
learned the hard way — and at worst a lossy bottleneck that injects noise into
the one input the flow depends on.

So the architecture is: **standardize the sufficient statistics, hand them
straight to the flow.** The flow's own conditioner networks supply all the
nonlinearity the mapping $x \mapsto p(\theta \mid x)$ requires.

Two secondary reasons reinforce it. Dropout inside an embedding is active in
ways that interact badly with per-observation sampling, and any batch-dependent
normalisation is wrong when `posterior.sample()` calls the network with
`batch_size = 1` — the reason the class uses `LayerNorm` rather than
`BatchNorm1d` in the first place. Bypassing it sidesteps both.

**When you would want the embedding back:** if $x$ grows beyond a handful of
near-sufficient numbers — raw twin-level data, many phenotypes at once, or the
45-element covariance vector of the bivariate SEM-PGS model in
[`../sempgs_npe/bivariate/`](../sempgs_npe/bivariate/). To re-enable it, swap
the two marked lines in `02_train_npe.py`. **This invalidates every existing
trained run**, which must then be retrained.

### 3.5 Standardization and the prior

**Features.** A `StandardScaler` is fit on the training split only and stored
as `feature_scaler.pkl` in each run directory. Every downstream script loads
that exact scaler — a mismatch here silently produces confident nonsense, which
is why `ace_model.build_features` reconstructs the feature vector from the
run's own `config.json` rather than from any hard-coded ordering.

**Parameters.** `sbi` z-scores $\theta$ internally
(`z_score_theta='independent'`), so the flow works in standardized parameter
space and its base distribution stays well matched.

**The prior** plays two roles: it is the distribution the training $\theta_i$
are drawn from, and it defines the support `sbi` will sample within. Note a
deliberate mismatch in this pipeline:

- Training $\theta$ are drawn as $A, C, E \sim \mathcal U(0,1)$
  **independently** — no sum-to-one constraint (see
  `ace_model.generate_training_data`). This is the *effective* prior the
  learned posterior is conditioned on.
- The prior handed to `sbi` is a `BoxUniform` derived from the training range
  **inflated by `--prior_buffer`** (default 0.5), giving bounds like
  $[-1, 2]$ per parameter.

The wider declared box exists so `sbi` does not truncate or reject draws near
the edges of the region the flow was actually trained on. The consequence is
that posterior draws can fall slightly outside $[0,1]$, and the demo passes
`reject_outside_prior=False` accordingly. Interpret the posterior as being
under the $\mathcal U(0,1)^3$ prior, not the declared box.

### 3.6 Hyperparameters

CLI defaults in `02_train_npe.py`. The values actually used for a given run are
recorded in that run's `config.json` and can differ — `se_proxy`, for instance,
used `flow_hidden=128, flow_transforms=8`.

| Flag | Default | Role |
|---|---|---|
| `--flow_type` | `nsf` | spline flow; also `maf`, `maf_rqs`, `mdn` |
| `--flow_hidden` | 64 | width of the conditioner networks inside the flow |
| `--flow_transforms` | 5 | number of stacked spline transforms |
| `--prior_type` | `boxuniform` | or `gaussian` |
| `--prior_buffer` | 0.5 | inflation of the prior box beyond the training range |
| `--batch_size` | 1024 | large batches suit the cheap simulator |
| `--lr` | 5e-4 | Adam learning rate |
| `--epochs` | 500 | maximum; early stopping usually triggers first |
| `--stop_after_epochs` | 50 | early-stopping patience on validation loss |
| `--include_n_pairs` | off | add an $N$-derived feature (see below) |

Data are split 70 / 15 / 15. Train and validation are handed to `sbi` together
with `validation_fraction` set so the split is honoured internally; the test
15 % is never seen during training and is what `test_metrics.json` reports.

**Encoding the sample size.** Posterior width depends on $N$, so telling the
network $N$ should sharpen it. Three encodings are supported, auto-detected
downstream from `config.json` by `ace_model.n_pairs_encoding`:

| Encoding | Feature | Rationale |
|---|---|---|
| `se_proxy` | $1/\sqrt{N}$ | **preferred** — linear in the quantity that actually scales posterior SD |
| `log_N_pairs` | $\log N$ | compresses the range, but not linear in the SE |
| `N_pairs` | $N$ | raw; poorly scaled across 50 → 20 000 |

`--include_n_pairs` selects `se_proxy` when available. Because the standard
error of a covariance estimate falls as $1/\sqrt{N}$, this feature enters
roughly linearly in the thing being predicted, which is the easiest possible
job for the network.

---

## Part 4 — The pipeline, step by step

**`01_generate_training_data.py`** — draws $A, C, E \sim \mathcal U(0,1)$
independently, picks $N$, simulates twin pairs, reduces each 2×2 sample
covariance matrix through `summarize_cov`, and records the four resulting
statistics plus all three $N$ encodings. Fixed-$N$ files
(`--n_pairs 20000`) are used to train the "no N" models. → `data/`

**`02_train_npe.py`** — cleans the data (drops rows where $|cov| > |var|$,
which cannot come from a valid covariance matrix), splits, standardizes, builds
the prior from the training range, trains the flow, then evaluates on the
held-out test split with posterior mean, MAP and posterior SD.
→ `results/models/<run>/` containing `posterior.pkl`, `config.json`,
`feature_scaler.pkl`, `density_estimator.pt`, `test_metrics.json`, plots.

**`03_evaluate_oos_predictions.py`** — the calibration check. Draws fresh
samples with a *different seed* from training and reports what training-time
metrics cannot: 95 % credible-interval coverage (should be ≈ 0.95), bias
relative to SD, and MAP against posterior mean. → `results/oos/<run>/`

**`04_fit_openmx_reference.R`** — draws 200 conditions from a symmetric
Dirichlet(1,1,1) so $A+C+E=1$ exactly, then for each condition × sample size
(50 → 20 000) simulates twin data and fits the ACE model in OpenMx using
covariance-matrix input, in the variance-component parameterisation with
$V_A, V_C \ge 0$ and $V_E \ge 10^{-6}$. Standardized SEs come from the delta
method applied to the parameter covariance matrix from the Hessian.
→ `data/ace_test_conditions.csv`, `results/simulations/ace_simulation_results.csv`

**`05_simulate_posterior_recovery.py`** — replays *the same* conditions through
the NPE, applying `summarize_cov` exactly as STEP 01 did so training and
inference features match. Feature construction otherwise adapts automatically
to whichever $N$ encoding the loaded run used, so one script serves both the
"with N" and "no N" models. Reports posterior mean (↔ point estimate),
posterior SD (↔ SE) and MAP.
→ `results/simulations/npe_simulation_results*.csv`

**`06_analysis.ipynb`** — aggregates both estimators by sample size and
compares **RMSE around truth**, mean reported SE, and bias with 95 % CI
ribbons. Every figure and table is written through the `save_fig` / `save_table`
helpers. → `results/analysis/` (5 figures + 11 tables, including a tidy
long-format `summary_long.csv`)

**`07_se_calibration.ipynb`** — checks whether the reported SE is *correct*.
STEP 05 with `--n_reps 200 --n_conditions 24` replicates each condition 200
times at fixed true parameters, so the **spread of the estimates across
replicates is the true sampling SE**. Comparing that against the mean reported
posterior SD gives the calibration ratio. Reports `ratio_se`, `ratio_rmse` and
95 % CI coverage side by side — see the shrinkage caveat below for why the
first of those must not be read alone. → `results/se_calibration/`

**Result** (24 conditions × 200 replicates, `se_proxy`, retrained to include
$N=20\,000$ in its training range — see Part 7 for why that matters). Below
$N=2000$, i.e. everywhere the flow was densely trained:

| Parameter | `ratio_se` | `ratio_rmse` | `coverage95` |
|---|---|---|---|
| A | 1.32 | 1.15 | 0.955 |
| C | 1.26 | 1.07 | 0.946 |
| E | 1.32 | 1.15 | 0.940 |

**The NPE's uncertainty is trustworthy, and mildly conservative, everywhere
except very large $N$.** Coverage is close to nominal (0.94–0.96) and the
reported SD is within roughly 15 % of total error. The larger `ratio_se`
(~1.3) is the shrinkage signature described in Part 5, not a miscalibration:
the posterior mean is pulled toward the prior, so the *spread of the point
estimates* is smaller than the honest uncertainty about them.

At $N=20\,000$ this breaks down badly — `ratio_se` 2.1–3.2, `ratio_rmse`
1.8–2.0 — and, contrary to what an earlier pass through this analysis
concluded, it is **not** a training-range extrapolation artefact: see Part 7
for the retrain that tested and ruled that out.

---

## Part 5 — The math behind the three calibration metrics

STEP 07 fixes a parameter setting $\theta_0=(A,C,E)$ and a sample size $N$,
draws $R$ independent datasets $x_1,\dots,x_R \sim p(x\mid\theta_0)$ (fresh
twin data each time), and fits each with the trained posterior
$q_\phi(\theta\mid x)$. That gives, per replicate $r$: a posterior mean
$\hat\theta_r=\mathbb E_{q_\phi}[\theta\mid x_r]$ and a posterior SD
$s_r=\mathrm{SD}_{q_\phi}[\theta\mid x_r]$ (both read straight off
`posterior.sample`). Three quantities are built from $\{\hat\theta_r, s_r\}$.

### 5.1 `mc_se` — the true sampling SE, and its own uncertainty

$$
\texttt{mc\_se} = \sqrt{\frac{1}{R-1}\sum_{r=1}^R (\hat\theta_r-\bar{\hat\theta})^2}
$$

This is the textbook definition of a standard error: the SD of an estimator's
sampling distribution, here obtained by literally sampling that distribution
$R$ times rather than approximating it. It is what `mean_post_sd` is being
checked against.

Because it is itself computed from a finite sample, `mc_se` carries its own
Monte-Carlo error. For $(R-1)S^2/\sigma^2 \sim \chi^2_{R-1}$,
$\operatorname{Var}(S^2)=2\sigma^4/(R-1)$; the delta method
($g(u)=\sqrt u,\ g'(\sigma^2)=1/2\sigma$) then gives

$$
\frac{\mathrm{SD}(S)}{\sigma} \approx \frac{1}{\sqrt{2(R-1)}}
$$

— the relative precision printed at the top of the notebook (≈7 % at $R=200$,
≈2.2 % at $R=1000$). This bounds how finely `ratio_se` and `ratio_rmse` can be
resolved; a ratio that differs from 1 by less than this margin is noise, not
miscalibration.

### 5.2 `ratio_se` — and why it is *not* expected to equal 1

$$
\overline{s} = \frac1R\sum_{r=1}^R s_r,
\qquad
\texttt{ratio\_se} = \frac{\overline{s}}{\texttt{mc\_se}}
$$

At finite $N$ the posterior is a **shrinkage estimator**: with an informative
prior, $\hat\theta(x)$ is pulled toward the prior mean, which *reduces* the
sampling variance of $\hat\theta$ below what the posterior SD reports. A
minimal model makes this exact. Approximate the sampling distribution of a
summary statistic by $\hat\theta_{\mathrm{lik}}(x)\sim\mathcal N(\theta_0,\sigma_{\mathrm{lik}}^2)$
and the prior by $\theta\sim\mathcal N(\mu_0,\sigma_0^2)$ (a Gaussian stand-in
for the pipeline's effective $\mathcal U(0,1)^3$ prior). Standard
Normal–Normal conjugacy gives

$$
\theta \mid x \;\sim\; \mathcal N\big(w\,\hat\theta_{\mathrm{lik}}(x) + (1-w)\mu_0,\; \sigma_{\mathrm{post}}^2\big),
\qquad
w=\frac{\sigma_0^2}{\sigma_0^2+\sigma_{\mathrm{lik}}^2},
\qquad
\sigma_{\mathrm{post}}^2 = w\,\sigma_{\mathrm{lik}}^2
$$

Since only $\hat\theta_{\mathrm{lik}}(x)$ varies across replicates, the sampling
variance of the posterior mean is
$\operatorname{Var}_x[\hat\theta_{\mathrm{Bayes}}]=w^2\sigma_{\mathrm{lik}}^2$ — i.e.
$\texttt{mc\_se}=w\,\sigma_{\mathrm{lik}}$, while the posterior SD is
$\sigma_{\mathrm{post}}=\sqrt w\,\sigma_{\mathrm{lik}}$. Their ratio:

$$
\texttt{ratio\_se} \;=\; \frac{\sigma_{\mathrm{post}}}{\texttt{mc\_se}} \;=\; \frac{1}{\sqrt w} \;\ge\; 1
$$

with equality only as $w\to1$ — i.e. as $N\to\infty$ and the likelihood
dominates the prior ($\sigma_{\mathrm{lik}}\to0$). This is the toy-model version
of the **Bernstein–von Mises theorem**: as $N$ grows, the posterior becomes
asymptotically Gaussian, centred at the MLE, with covariance equal to the
inverse Fisher information — matching the frequentist sampling covariance of
$\hat\theta$, so prior influence (and `ratio_se`'s excess over 1) vanishes. At
finite $N$, `ratio_se` > 1 is the **expected signature of a correctly
calibrated Bayesian estimator**, not evidence of miscalibration — which is
exactly the measured pattern (§ Part 4): `ratio_se` ≈ 1.3 in-range, closer to 1
would only be expected in the $N\to\infty$ limit.

### 5.3 `ratio_rmse` — the Bayes-risk identity

$$
\texttt{rmse} = \sqrt{\frac1R\sum_r(\hat\theta_r-\theta_0)^2}, \qquad
\texttt{ratio\_rmse} = \frac{\overline s}{\texttt{rmse}}
$$

Unlike `ratio_se`, this ratio has an exact target of 1 — but only once pooled
correctly. For the posterior mean under squared-error loss, decision theory
gives an exact identity (not an approximation): averaged over the **same**
joint distribution $\theta\sim p(\theta),\ x\sim p(x\mid\theta)$ used to train
the flow,

$$
\underbrace{\mathbb E_{\theta,x}\!\big[(\theta-\hat\theta(x))^2\big]}_{\text{Bayes risk}}
\;=\;
\mathbb E_x\!\big[\operatorname{Var}(\theta\mid x)\big]
$$

by the tower rule: conditioning on $x$ first, $\mathbb
E_{\theta\mid x}[(\theta-\hat\theta(x))^2]=\operatorname{Var}(\theta\mid x)$ exactly,
because $\hat\theta(x)$ *is* the posterior mean. So a correctly-specified,
well-trained posterior has $\overline{s^2}\approx\texttt{rmse}^2$ **once
averaged over a representative sample of $\theta$ drawn like the prior** —
which is exactly what pooling `ratio_rmse` over many conditions approximates,
and why it is the fairer target: 1.06–1.21 in-range, versus `ratio_se`'s 1.3.

One caveat this identity makes explicit: the test conditions come from
`04_fit_openmx_reference.R`'s Dirichlet(1,1,1) draws ($A+C+E=1$ exactly), not
the pipeline's actual training prior — independent $A,C,E\sim\mathcal U(0,1)$
(no sum constraint). The identity above is exact only when the two coincide, so
residual drift of `ratio_rmse` away from 1 partly reflects that mismatch, not
purely flow miscalibration.

### 5.4 `coverage95` — marginal vs. pointwise, and why per-condition coverage can miss 0.95

$$
\texttt{coverage95} = \frac1R\sum_{r=1}^R \mathbb 1\!\big[\theta_0 \in [q_{2.5}(\theta\mid x_r),\, q_{97.5}(\theta\mid x_r)]\big]
$$

By construction, a 95 % credible interval satisfies
$P(\theta\in\mathrm{CI}(x)\mid x)=0.95$ for every $x$ — that is simply what
"95 % credible" means. Averaging over the joint $(\theta,x)$ then gives exact
**marginal** coverage:

$$
P\big(\theta\in\mathrm{CI}(x)\big) = \mathbb E_x\big[P(\theta\in\mathrm{CI}(x)\mid x)\big] = 0.95
$$

But `coverage95` is computed at **fixed** $\theta=\theta_0$ — a *conditional*
(frequentist) coverage — and nothing above guarantees that equals 0.95 for
every individual $\theta_0$, even for an exactly-computed posterior. Bayesian
credible intervals guarantee coverage on average over the prior, not
pointwise. Pointwise coverage converges to the nominal level only
asymptotically, again by Bernstein–von Mises, and — critically — only at
**interior** points of the parameter space; the theorem's regularity
conditions fail at a boundary. $A\to0$ or $C\to0$ are literally the boundary
of this model's support, which is a concrete, testable reason to expect the
two low-$A$/low-$C$ conditions flagged in Part 7 to behave differently from
the interior ones — worth checking directly against `coverage95` in
`se_calibration_by_condition.csv` once a larger `--n_reps` run tightens the
per-condition estimate.

---

## Part 6 — Trained runs

The two runs STEP 06 depends on have been **retrained under the sufficient
reduction** and are current. The remaining four still carry the old
`S[0,0]`-only features and warn on load (see Part 7).

| Run | Features | Prior | Test R² | Status |
|---|---|---|---|---|
| `se_proxy` | 5 (`se_proxy`) | boxuniform | **0.882** | ✅ current — **default**, the "with N" arm of STEP 06 and the demo. Trained on $N\in\{50,\dots,5000,20\,000\}$ |
| `no_n_pairs_gaussian_prior` | 4 | gaussian | **0.998** | ✅ current — the "no N" arm of STEP 06 |
| `gaussian_prior` | 5 (`log_N_pairs`) | gaussian | 0.862 | ⚠ stale features — retrain before use |
| `boxuniform_prior` | 5 (`log_N_pairs`) | boxuniform | 0.857 | ⚠ stale features — retrain before use |
| `default_legacy` | 5 (`log_N_pairs`) | — | 0.859 | ⚠ stale + posterior will not load |
| `no_n_pairs_legacy` | 4 | — | 0.972 | ⚠ stale + posterior will not load |

Retraining the two current runs on the sufficient features improved both, as
the $2/(1+\rho^2)$ efficiency argument predicts, and `se_proxy` was retrained a
second time to add $N=20\,000$ to its training range (Part 7):

| Run | R² | RMSE | Change |
|---|---|---|---|
| `se_proxy` — original | 0.8731 | 0.1037 | baseline |
| `se_proxy` — sufficient-variance fix | 0.8813 | 0.0980 | −5.5 % RMSE |
| `se_proxy` — + $N=20\,000$ in training | **0.8816** | **0.0991** | ≈ unchanged |
| `no_n_pairs_gaussian_prior` — original | 0.9943 | 0.0211 | baseline |
| `no_n_pairs_gaussian_prior` — sufficient-variance fix | **0.9976** | **0.0142** | −33 % RMSE |

> **These R² values are not comparable across rows.** The 4-feature runs were
> trained and tested on fixed-$N$ data (`ace_training_data_N20000.csv`, so
> $N = 20\,000$ throughout), where sampling noise is tiny and almost any
> estimator looks excellent. The 5-feature runs used mixed
> $N \in [50, 5000, 20\,000]$, which includes genuinely hard low-$N$ cases.
> Compare estimators on the STEP 06 output, which holds the test conditions
> fixed — not on this column. Note also that a good aggregate R² on the
> held-out test split is fully consistent with the N=20,000 SE-calibration
> problem in Part 7: R² measures the point estimate, and that stays accurate;
> the reported *uncertainty* is what under-shrinks.

---

## Part 7 — Known limitations and gotchas

**⚠ Four of the six trained runs predate the sufficient-variance fix and must
be retrained before use.** The pipeline originally used `S[0,0]` alone as each `*_var`
feature, discarding `S[1,1]` — while OpenMx received the complete 2×2 matrix.
That handicapped the NPE in the STEP 06 comparison. `ace_model.summarize_cov`
now applies the sufficient reduction (§3.4), averaging the two diagonal
entries.

The efficiency recovered is exactly quantifiable. For a compound-symmetric
2×2 Wishart, using one diagonal entry instead of the mean of both inflates the
variance of the phenotypic-variance estimate by

$$
\frac{\operatorname{Var}(S_{11})}{\operatorname{Var}\!\big(\tfrac{1}{2}(S_{11}+S_{22})\big)}
= \frac{2}{1+\rho^{2}},
\qquad \rho = c/v \ \text{(the twin correlation)}
$$

so the old features were up to a factor 2 noisier in variance (a factor
$\sqrt2$ in SD) when $\rho \approx 0$, with the penalty shrinking to nothing as
$\rho \to 1$. MZ pairs, being highly correlated, lost least; DZ pairs and
low-heritability conditions lost most.

Because this changes what the features *mean*, any run trained on the old
definition is inconsistent with freshly simulated data.
`ace_model.load_posterior` detects this via the `var_feature` key in
`config.json`: runs written by the current `02_train_npe.py` are stamped
`"mean_diagonal"`, and anything else raises a `UserWarning` *and* prints a
`!! STALE RUN` banner to stderr. (The banner is not redundant — the pipeline
scripts call `warnings.filterwarnings('ignore')` to mute sbi chatter, which
would otherwise swallow the warning exactly where it matters.) Nothing fails
loudly, because each run's own recorded metrics remain valid, but **do not
trust a warning-flagged run for new inference.**

`se_proxy` and `no_n_pairs_gaussian_prior` — the two runs STEP 06 consumes —
have already been retrained, and `data/` regenerated, so the current STEP 06
output is valid. The other four runs are untouched; regenerate and retrain
them the same way (Part 8) before using them. STEP 04's OpenMx output was
**not** rerun and did not need to be: it consumes the full covariance matrices
and the theoretical test conditions, neither of which changed.

**At N = 20,000 the reported SE is too wide by ~1.8×, and this is *not* an
extrapolation artefact — it persists after retraining in-range.** The first
version of this finding blamed training-range extrapolation:
`01_generate_training_data.py` originally drew $N\in\{50,\dots,5000\}$, so
the `se_proxy`$=1/\sqrt N$ feature at $N=20\,000$ was unseen in training. That
hypothesis was **tested directly** by adding $N=20\,000$ to the default
training range (now $\{50,\dots,5000,20\,000\}$, ~6,200 of 50,000 training
rows at that value) and retraining `se_proxy` from scratch — same
hyperparameters, otherwise identical pipeline. The miscalibration barely
moved (`ratio_se` for A: 2.65 → 2.40; C: 1.96 → 2.09; E: 3.18 → 3.17, on 24
conditions × 200 replicates each).

Splitting the STEP 07 Monte-Carlo output into point-estimate precision versus
reported-uncertainty precision shows exactly where the shrinkage stalls, using
$A$ as an example (mean over the 24 conditions, N=2000 → N=20,000, a 10×
increase in data that should shrink any SE by $\sqrt{10}\approx3.16\times$):

| Quantity | N=2000 | N=20,000 | Ratio | Expected |
|---|---|---|---|---|
| `mc_se` (true spread of point estimates) | 0.0263 | 0.0092 | **2.85×** | 3.16× |
| `mean_post_sd` (what the NPE reports) | 0.0326 | 0.0182 | **1.79×** | 3.16× |

**The point estimates themselves are precise and well-behaved at N=20,000** —
`mc_se` tracks the theoretical $1/\sqrt N$ scaling almost exactly. It is
specifically the *reported* posterior width that fails to sharpen enough,
under-shrinking by nearly 2× relative to what the data actually support. That
localizes the problem to the flow's density estimator, not to the se_proxy
feature or the training range: something about representing an extremely
peaked conditional density — the posterior this well-identified, high-N regime
calls for — is harder for this architecture (NSF, 128 hidden, 8 transforms) or
undertrained relative to the more diffuse posteriors the bulk of training data
produces. Candidates worth testing: more `flow_transforms`, more epochs
specifically weighted toward high-$N$ rows, or an embedding network reinstated
for exactly the reason given in §3.4 — with a coarser sufficient statistic this
was unnecessary, but resolving a razor-thin posterior may need more of the
input signal preserved than raw features conveniently provide. This is now an
open problem, not a data-generation bug.

**Two of the twenty-four Monte-Carlo conditions stop gaining precision with
more data — and this held up under a 2× larger, independently-drawn set of
conditions.** Between $N=200$ and $N=2000$ (a 10× increase, so a 3.16× drop in
SE is expected), condition 5 ($A{=}0.095$, $C{=}0.149$, $E{=}0.756$) improved
by only 1.07× and condition 8 ($A{=}0.191$, $C{=}0.029$, $E{=}0.780$) by
1.57×, while the other 22 conditions — including all 12 added in the larger
rerun — reached 1.9–3.7×. Both stalling conditions have **small $A$ and small
$C$ with large $E$**, and no new condition in the expanded set reproduced that
pattern, which sharpens rather than weakens the hypothesis: this looks like a
real feature of the low-$A$/low-$C$ corner, not a fluke of 12 samples. It could
be prior domination where $A$ and $C$ are weakly identified, or a floor on how
sharp a posterior the flow can represent — and ties to the boundary caveat in
§5.4 (Bernstein–von Mises asymptotics fail near $A\to0$ or $C\to0$). Worth
chasing before trusting the NPE in that corner. Per-condition numbers are in
`results/se_calibration/se_calibration_by_condition.csv`.

**The no-N model's posterior width is nearly constant in $N$** — by
construction, and it is the clearest result in STEP 06. Its mean SE sits at
≈ 0.021 at *every* sample size (ratio to its own $N=50$ value: 0.94, 1.00,
1.00, 0.93, 1.05, 1.00), because nothing in its input tells it how much data
produced the covariances. Against OpenMx that makes it ~7× overconfident at
$N=50$ and ~2.7× underconfident at $N=20\,000$. Point estimates stay roughly
unbiased throughout; it is the *uncertainty* that is uninformative. This is
the empirical case for feeding the network an $N$ encoding.

**STEP 06's precision row measures RMSE, not the SD of estimates — and this
was a real bug.** That panel used to plot `SD(A_est)` taken *across the 200
different test conditions*. Because each condition has a different true value,
that statistic is dominated by the Dirichlet spread of the conditions
themselves and is nearly invariant to sample size — it read 0.2515 at $N=50$
and 0.2281 at $N=20\,000$, against a true across-condition spread of 0.2317.
It looked like "precision barely improves with N", which is false. It now
plots RMSE of each estimate around *its own* truth, which falls properly with
$N$ (OpenMx $\hat A$: 0.148 → 0.011 across the grid) and is directly
comparable with the mean reported SE plotted beneath it. Genuine
fixed-condition sampling SEs come from STEP 07.

**Posterior SD is not expected to equal the Monte-Carlo SD of the point
estimate at small $N$.** The NPE posterior is Bayesian under an effective
$\mathcal U(0,1)^3$ prior, so at small $N$ the posterior mean is shrunk toward
the prior. Shrinkage *reduces* the sampling variance of the estimator while
introducing bias, so `mc_se` can sit below the reported posterior SD even for a
perfectly calibrated posterior. Read STEP 07's three metrics together:
`ratio_se` (vs the spread of point estimates), `ratio_rmse` (vs total error,
which absorbs the bias) and `coverage95` (the direct Bayesian check, target
0.95). Measured on the STEP 07 grid (24 conditions × 200 replicates): `ratio_se`
sits at ≈1.3 across the whole in-distribution range ($N\le2000$) while
`ratio_rmse` stays at 1.07–1.15 and coverage holds at 0.94–0.96. The gap
between the two ratios is real and persistent, and it is the shrinkage term —
treating `ratio_se != 1` as a defect is a misreading. (This pattern breaks
down at $N=20\,000$ for a different, tested reason — see below.)

**Two archived posteriors cannot be loaded.** `default_legacy` and
`no_n_pairs_legacy` were pickled from a `__main__` scope that defined
`ACEEmbeddingNet`, so unpickling raises
`AttributeError: Can't get attribute 'ACEEmbeddingNet' on <module '__main__'>`.
Their `config.json`, `test_metrics.json` and plots remain readable. They are
kept only for provenance — retrain if you need them.

**Three CLI flags do not affect the model.** `--hidden_sizes` and `--dropout`
configure only the bypassed embedding network (they are still recorded into
`config.json`, which is misleading), and `--weight_decay` is parsed but never
passed to `inference.train()`. They are retained so existing invocations keep
working.

**The declared prior is wider than the sampling distribution** — see §3.5.
Posterior draws may fall outside $[0,1]$.

**Posterior sampling runs on CPU by design.** `02_train_npe.py` may train on
MPS or CUDA, but evaluation is forced to CPU: per-kernel launch overhead makes
serial single-observation sampling markedly slower on an accelerator than on
CPU.

---

## Part 8 — Reproducing from scratch

This is also the sequence to run after the sufficient-variance fix (Part 7).
STEP 04 is the slow step and can be **skipped** if
`results/simulations/ace_simulation_results.csv` already exists — the OpenMx
side is unaffected by the fix.

```bash
# seconds, not minutes — the simulator is cheap
python 01_generate_training_data.py --n_samples 50000 --output ace_training_data.csv
python 01_generate_training_data.py --n_pairs 20000 --n_samples 20000 \
                                    --output ace_training_data_N20000.csv

python 02_train_npe.py --data ace_training_data.csv --include_n_pairs \
                       --epochs 500 --device cpu --output se_proxy
python 02_train_npe.py --data ace_training_data_N20000.csv --epochs 500 \
                       --device cpu --prior_type gaussian \
                       --output no_n_pairs_gaussian_prior

python 03_evaluate_oos_predictions.py --model_dir se_proxy

Rscript 04_fit_openmx_reference.R          # slowest step: 1400 OpenMx fits
                                           # skip if its output already exists

python 05_simulate_posterior_recovery.py --model_dir se_proxy \
       --output npe_simulation_results.csv
python 05_simulate_posterior_recovery.py --model_dir no_n_pairs_gaussian_prior \
       --output npe_simulation_results_no_n.csv

jupyter lab 06_analysis.ipynb

# STEP 07: Monte-Carlo SE calibration (~18 min for 16,800 fits)
python 05_simulate_posterior_recovery.py --model_dir se_proxy \
       --n_conditions 24 --n_reps 200 --output npe_se_montecarlo.csv
jupyter lab 07_se_calibration.ipynb
```

**Dependencies:** Python — `sbi`, `torch`, `scikit-learn`, `pandas`, `numpy`,
`matplotlib`, `joblib`, `scipy`. R — `OpenMx`, `MASS`.

`data/` and `results/` are gitignored, so a fresh clone starts empty and the
sequence above rebuilds everything.
