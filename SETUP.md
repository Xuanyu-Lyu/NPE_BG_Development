# Development setup

This repository has two related projects:

- `ace_npe/` is the complete ACE twin-model validation pipeline. It simulates
  training data, trains an `sbi` neural posterior estimator, evaluates it, and
  compares it with an OpenMx reference analysis.
- `sempgs_npe/bivariate/` simulates and trains a larger 14-parameter bivariate
  SEM-PGS estimator.
- `sempgs_npe/univariate/` is cluster-specific and is not currently standalone.
  It imports `core_simulation.AssortativeMatingSimulation`, but
  `core_simulation.py` is not in this repository. It also contains hard-coded
  `/projects/xuly4739/...` paths and a Slurm configuration. Obtain that missing
  module and update the paths before trying to run this part locally.

## Recommended Python environment

Use the included Mamba/Conda environment. Mamba is preferred here because it
uses the same environment format as Conda but usually resolves it faster.

From the repository root:

```bash
mamba env create -f environment.yml
```

After the environment has been created, complete these two steps:

1. Activate the environment in your terminal:

   ```bash
   conda activate npe-bg
   ```

2. Select the environment in VS Code. Open the Command Palette with
   `Cmd+Shift+P`, choose **Python: Select Interpreter**, and select `npe-bg`.
   On this Mac, the interpreter's full path is:

   ```text
   /opt/homebrew/Caskroom/miniforge/base/envs/npe-bg/bin/python
   ```

Then verify the installation from the repository root:

```bash
python scripts/check_environment.py
```

To update an environment after `environment.yml` changes:

```bash
mamba env update -n npe-bg -f environment.yml --prune
```

The environment uses Python 3.12. The base scientific stack comes from
conda-forge, while PyTorch and `sbi` are installed by pip inside the isolated
environment. `sbi` is pinned to 0.26.1 because the training scripts import the
legacy `SNPE` alias; that alias is being deprecated upstream. A future upgrade
should first change `SNPE` to `NPE` and verify that newly saved and existing
pickled posteriors still load correctly.

If `npe-bg` does not appear as a notebook kernel, register it once:

```bash
python -m ipykernel install --user --name npe-bg --display-name "Python (npe-bg)"
```

## Packages and why they are needed

| Package | Used for |
|---|---|
| NumPy | simulation, covariance matrices, numeric operations |
| pandas | CSV inputs, outputs, and result tables |
| SciPy | kernel-density MAP estimation in `ace_model.py` |
| PyTorch | neural networks, tensors, CPU/CUDA/MPS training |
| sbi | SNPE-C and neural spline-flow posterior estimation |
| scikit-learn | feature scaling and prediction metrics |
| joblib | saving/loading fitted scalers |
| Matplotlib | diagnostic and analysis figures |
| JupyterLab + IPython kernel | the three analysis/demo notebooks |

Python's `argparse`, `json`, `math`, `pathlib`, `pickle`, `sys`, `warnings`,
and similar imports are standard-library modules and do not need installation.

## R/OpenMx dependency

Only `ace_npe/04_fit_openmx_reference.R` needs R. It imports:

- `OpenMx`, which is not part of the Python environment;
- `MASS`, which normally ships with a standard R installation.

Check the R side with:

```bash
Rscript -e 'library(OpenMx); library(MASS); sessionInfo()'
```

If OpenMx is missing, install it using the current instructions from the
[OpenMx project](https://openmx.ssri.psu.edu/installing-openmx). Keeping R
separate from the Python environment is simpler on Apple silicon and matches
the system R already available on this machine.

## Quick smoke test

After activating the environment, run a small ACE simulation:

```bash
python ace_npe/01_generate_training_data.py \
  --n_samples 100 \
  --output ace_training_data_smoke.csv
```

The full ACE workflow is documented in `ace_npe/README.md`. Training is much
more expensive than this smoke test. On an Apple-silicon Mac, the scripts'
`--device auto` setting selects PyTorch MPS when available; use `--device cpu`
if an `sbi` operation is unsupported or unstable on MPS.

## Reproducibility note

The repository saves complete `sbi` posterior objects with Python `pickle`.
Those files can be sensitive to the exact Python, PyTorch, and `sbi` versions.
For a publication or long-lived model archive, export an exact lock after a
successful training run:

```bash
conda list --explicit > conda-explicit-lock.txt
python -m pip freeze > pip-lock.txt
```

The portable `environment.yml` should remain the main development definition;
the lock files are machine/platform-specific snapshots.
