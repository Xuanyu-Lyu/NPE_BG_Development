#!/bin/bash
# Aggregate STEP 06 cells, create figures, and remove temporary cell files.
# Submit with an afterok dependency on the STEP 06 array.

#SBATCH --job-name=ace06-plot
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=ace06-plot-%j.out
#SBATCH --error=ace06-plot-%j.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py
export MPLBACKEND=Agg

python -u ace_npe/06_npe_diagnostics.py \
  --aggregate \
  --k-values 100000 300000 500000 \
  --n-values 50 100 500 1000 2000 5000 20000 \
  --n-test-datasets 1000 \
  --n-posterior-draws 2000 \
  --output-dir step06_npe_diagnostics_rmse \
  --device cpu
