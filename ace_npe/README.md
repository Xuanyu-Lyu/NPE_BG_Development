# ACE × Neural Posterior Estimation

The active pipeline evaluates whether neural posterior estimation (NPE) recovers
ACE twin-model parameters and uncertainty, using OpenMx and a Bayesian grid
posterior as reference analyses. The controlled ACE benchmark supports later
work on the more complicated SEM-PGS models in `../sempgs_npe/`.

All commands below run from the repository root. Python scripts resolve data,
model, and result paths relative to `ace_npe/`. Environment setup is documented
in [`../SETUP.md`](../SETUP.md).

## Model and notation

The current fixed-N experiments use `(A, C, E) ~ Dirichlet(1,1,1)` with
`A+C+E=1`. The MZ population covariance is `A+C`; the DZ covariance is
`0.5*A+C`. Both population variances equal one. Each dataset contains N MZ twin
pairs and N DZ twin pairs drawn from the corresponding bivariate Gaussian.

The four NPE inputs are `mz_var`, `mz_cov`, `dz_var`, and `dz_cov`. Each variance
is the mean of the two sample-covariance diagonal entries; each covariance is
the off-diagonal entry. These are sufficient summaries for the current
compound-symmetric covariance model. Simulation uses either raw twin data or
an equivalent direct Wishart sampler.

| Symbol | Meaning |
|---|---|
| K | Simulation corpus size used to train an NPE, including its validation split |
| N | Twin pairs per zygosity group |
| H | Independently trained models per cell |
| M | Evaluation datasets per N |
| L | Joint posterior draws per dataset |

The flow learns two additive-log-ratio coordinates, `log(A/E)` and `log(C/E)`.
Posterior draws are transformed back to positive ACE components summing to one.
The standardized covariance summaries enter the flow directly through
`nn.Identity()`; the fixed-N models do not receive N as an input feature.

## Active steps

| Step | Script | Purpose |
|---|---|---|
| 01 | `01_generate_training_data.py` | Generate Dirichlet ACE training data |
| 02 | `02_train_npe.py` | Train and save an NPE from a simulation CSV |
| 03 | `03_training_budget_grid.py` | Compare training budgets across fixed N values |
| 04 | `04_compare_fixed_n_dirichlet_openmx.py` | Compare saved fixed-N Dirichlet NPEs with OpenMx on paired data |
| 05 | `05_compare_training_sizes_openmx.py` | Compare OpenMx with the retained Step 03 NPE results |
| 06 | `06_npe_diagnostics.py` | Marginal SBC, predictive-RMSE SBC, recovery, and contraction |
| 07 | `07_evaluate_fixed_theta.py` | Evaluate saved 100k models on repeated datasets at one fixed theta |
| 08 | `08_fixed_theta_npe_ensemble.py` | Evaluate the saved 100-model fixed-theta ensemble |
| 09 | `09_decompose_fixed_theta_variance.py` | Compare total, model, dataset, and posterior uncertainty |
| 10 | `10_compare_fixed_theta_npe_grid.R` | Compare the Step 08 NPE estimates with a Bayesian grid posterior |

Steps 03 and 06 train transient models independently and require no saved
models from Step 02. Step 05 uses Step 03's compact results and regenerates its
exact test covariance matrices; it does not load or retrain NPEs. Steps 04, 07,
and 08 require saved models. Steps 09 and 10 require the completed Step 08
aggregation.

## Step 06: marginal and predictive-RMSE SBC on RC

Submit from the repository root on CU Boulder Alpine:

```bash
diagnostic_job=$(sbatch --parsable ace_npe/06_npe_diagnostics.sh)
sbatch --dependency=afterok:"$diagnostic_job" \
  ace_npe/06_aggregate_npe_diagnostics.sh
```

The array and aggregation scripts share this design:

| Setting | Value |
|---|---|
| K | 100,000; 300,000; 500,000 |
| N | 50; 100; 500; 1,000; 2,000; 5,000; 20,000 |
| H | 1 per K × N cell |
| M | 1,000 fresh prior-predictive datasets per N, shared across K |
| L | 2,000 joint posterior draws per dataset |
| Prior | Dirichlet(1,1,1), total variance fixed at 1 |
| Flow | NSF, 64 hidden features, 5 transforms |
| Training | Up to 500 epochs; early stopping after 50 epochs without improvement |
| Validation fraction | 0.15 |
| Batch size / learning rate | 1,024 / 0.0005 |
| Simulation / training seeds | 2026 / 42 |

There are 21 array tasks, with K varying fastest within each N; up to seven
tasks run concurrently. Each task requests four CPU cores, 16 GB RAM, and
20 hours. The aggregation job requests one core, 8 GB RAM, and one hour.
The scripts use the existing RC project directory
`/pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development` and Conda environment
`npe-bg`.

All existing diagnostics are retained. The additional RMSE quantity evaluates
one parameter vector against the same observed, unstandardized summaries:

```text
observed   = (mz_var, mz_cov, dz_var, dz_cov)
prediction = (1, A+C, 1, 0.5*A+C)
RMSE       = sqrt(mean((prediction - observed)^2))
```

For each dataset, calculate this quantity at the true generating parameters
and at every joint posterior draw. Its SBC rank is the number of posterior
RMSEs below the true-parameter RMSE. Exact ties are randomized uniformly using
a separate seeded RNG. The new check reuses the exact draws used for marginal
SBC; it performs no additional posterior sampling or likelihood evaluation.
The variance terms are constant across draws within a dataset because total
variance is fixed. The ranking therefore depends on the predicted MZ/DZ
covariances.

RMSE is calculated in original covariance units with equal weights. This is
prediction of covariance summaries, rather than prediction of individual twin
phenotypes. It also differs from parameter-recovery RMSE, which compares
posterior means with true ACE components across datasets. Both are retained.

Marginal and RMSE SBC ranks lie in `0,...,L`. Their ECDF-minus-uniform plots use
the same discrete-uniform reference and 95% simulation-calibrated simultaneous
bands. A uniform rank distribution is the target; minimum RMSE is not the SBC
criterion. SBC evaluation draws fresh theta values from the matching prior;
the fixed-theta datasets in Steps 07–10 serve a separate purpose.

The new default output is `ace_npe/results/step06_npe_diagnostics_rmse/`, keeping
earlier results in `step06_npe_diagnostics/` separate:

| Output | Contents |
|---|---|
| `diagnostic_results.csv` | Existing per-dataset, per-parameter summaries and marginal ranks |
| `diagnostic_metrics.csv` | Existing bias, parameter RMSE/NRMSE, R-squared, and uncertainty metrics |
| `predictive_rmse_sbc_results.csv` | One row per dataset and cell: observed summaries, true ACE values, true RMSE, posterior RMSE summaries, rank, and tie counts |
| `cell_runtimes.csv` | Runtime, seeds, and configuration for each cell |
| `config.json` | Experiment, training, RMSE definition, and ECDF settings |
| `figures/calibration_ecdf_N*.png` | Existing marginal SBC plots |
| `figures/predictive_rmse_sbc_ecdf_N*.png` | Additional RMSE SBC plots, one panel per K |
| `figures/recovery_N*.png` | Existing posterior-mean recovery plots |
| `figures/z_score_contraction_N*.png` | Existing z-score versus contraction plots |
| `figures/nrmse_by_n.png`, `figures/r_squared_by_n.png` | Existing recovery metrics over N |
| `COMPLETE` | Written after successful aggregation and figure generation |

Fitted models and full posterior draws remain in memory and are discarded.
Temporary cell CSVs and metadata are removed only after successful aggregation;
use `--keep-cell-files` to retain them. Completed cells are reused only when
both diagnostic tables and matching configuration are present. Change the
output directory or use `--overwrite` when changing run settings.

Old marginal summaries cannot supply RMSE SBC ranks because they do not retain
joint posterior draws. The new RC run therefore retrains the transient models.
Figures from the new run can subsequently be regenerated without training:

```bash
python ace_npe/06_npe_diagnostics.py --replot --device cpu
```

An earlier marginal-only run can still be replotted with
`--output-dir step06_npe_diagnostics`; this does not add RMSE SBC to that run.
A full sequential run is available with `--run-all`. For a small local smoke
check of training, sampling, aggregation, and all plots:

```bash
python ace_npe/06_npe_diagnostics.py --run-all --device cpu \
  --k-values 200 300 400 --n-values 100 \
  --n-test-datasets 8 --n-posterior-draws 20 \
  --epochs 1 --stop-after-epochs 1 --batch-size 64 \
  --flow-hidden 8 --flow-transforms 2 \
  --output-dir /tmp/ace_rmse_sbc_smoke
```

These small settings verify execution; they are not a calibration study.

## Steps 01–05: training budgets and OpenMx comparisons

The CSV-based generation and saved-model training entry points are:

```bash
python ace_npe/01_generate_training_data.py --n_samples 50000
python ace_npe/02_train_npe.py --data ace_training_data.csv \
  --include_n_pairs --epochs 500 --device cpu --output dirichlet_with_n
```

This Step 02 example trains across N and includes its sample-size encoding.
Steps 03 and 06 instead train separate fixed-N models with four inputs.

Step 03 uses `K=10k,20k,50k,100k,200k,300k,500k` across the seven N values listed
above, with `H=1`, `M=1000`, and `L=2000`. It retains compact parameter summaries,
metrics, and distribution figures in `results/training_budget_grid/`.
Submit the array, aggregation, and dependent Step 05 comparison:

```bash
grid_job=$(sbatch --parsable ace_npe/03_training_budget_grid.sh)
grid_aggregate_job=$(sbatch --parsable --dependency=afterok:"$grid_job" \
  ace_npe/03_aggregate_training_budget_grid.sh)
sbatch --dependency=afterok:"$grid_aggregate_job" \
  ace_npe/05_compare_training_sizes_openmx.sh
```

Step 05 compares OpenMx with NPE-100k/200k/300k/500k on the same datasets.
Its paired bias, MAE, RMSE, uncertainty, and coverage outputs are saved to
`results/training_set_size_openmx_comparison/`. The native coverage intervals
are OpenMx estimate ± 1.96 SE and NPE posterior 2.5%–97.5% quantiles; an
additional coverage plot uses estimate ± 1.96 uncertainty for every method.

Step 04 separately compares saved fixed-N Dirichlet models with OpenMx on
200 shared parameter conditions at each N. It uses
`04_fit_openmx_paired_dirichlet.R` and reports both methods on the same rows
where OpenMx converged. Its defaults load models from
`results/models/prior_comparison/dirichlet/N*/` and write results to
`results/dirichlet_openmx_comparison/`:

```bash
python ace_npe/04_compare_fixed_n_dirichlet_openmx.py
```

When saved fixed-N models are needed, the generation and training utilities in
`prior_compare/` remain part of this workflow through their Dirichlet arm:

```bash
python ace_npe/prior_compare/01_generate_prior_data.py --schemes dirichlet \
  --n_samples 100000 --n_pairs 50 100 500 1000 2000 5000 20000 \
  --output_dir fixed_n_dirichlet_100k
python ace_npe/prior_compare/03_train_prior_models.py --schemes dirichlet \
  --data_dir fixed_n_dirichlet_100k --output_dir fixed_n_dirichlet_100k \
  --n_pairs 50 100 500 1000 2000 5000 20000 --epochs 500 --device cpu
python ace_npe/04_compare_fixed_n_dirichlet_openmx.py \
  --models_dir fixed_n_dirichlet_100k
```

## Steps 07–10: fixed-theta uncertainty and grid reference

Step 07 evaluates saved 100k models at `theta=(0.4,0.3,0.3)` for
`N=50,100,500,1000`, using 500 repeated datasets per N. It compares the sampling
SD of posterior means with `sqrt(mean(posterior variance))`, overall and in
five blocks of 100. Its current default model directory is
`results/models/prior_comparison_100k_N20000/dirichlet/`:

```bash
python ace_npe/07_evaluate_fixed_theta.py
```

To use the fixed-N models generated by the commands above, pass
`--models_dir fixed_n_dirichlet_100k/dirichlet`.

Step 08 reloads the existing 100 independently trained NPEs in
`results/fixed_theta_npe_ensemble/`. Each model used 100k simulations at N=1000.
They are evaluated on 500 shared fixed-theta datasets with 2000 posterior draws
per dataset. Submit Steps 08–10 with dependencies:

```bash
eval_job=$(sbatch --parsable ace_npe/08_fixed_theta_npe_ensemble.sh)
aggregate_job=$(sbatch --parsable --dependency=afterok:"$eval_job" \
  ace_npe/08_aggregate_fixed_theta_npe_ensemble.sh)
sbatch --dependency=afterok:"$aggregate_job" \
  ace_npe/09_decompose_fixed_theta_variance.sh
sbatch --dependency=afterok:"$aggregate_job" \
  ace_npe/10_compare_fixed_theta_npe_grid.sh
```

Step 08 writes `results/fixed_theta_npe_ensemble_evaluation/`.
Step 09 writes variance-scale and SE-scale comparisons of total, model,
dataset, and posterior uncertainty to
`results/fixed_theta_variance_decomposition/`. These quantities are compared;
they are not an additive decomposition. Step 10 uses the matching Dirichlet
prior and the analytic ACE covariance likelihood to form a numerical grid
posterior, writing `results/fixed_theta_npe_grid_comparison/`.

## Shared utilities and verification

`ace_model.py` supplies covariance math, feature definitions, and model loading.
`training_budget_utils.py` supplies the common Step 03/06 simulation, training,
and posterior-summary routines. `rmse_sbc.py` supplies the additional RMSE
quantity and rank calculation. `prior_compare/prior_compare_utils.py` supplies
the fixed-N simplex prior and posterior wrapper used by those routines.

Run the RMSE and marginal-SBC regression checks in the `npe-bg` environment:

```bash
python -m unittest discover -s ace_npe/tests -v
bash -n ace_npe/06_npe_diagnostics.sh ace_npe/06_aggregate_npe_diagnostics.sh
```
