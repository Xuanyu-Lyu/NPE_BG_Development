#!/bin/bash
# Train and diagnose one transient (K,N) NPE per Alpine array task.
# Submit from the repository root with: sbatch ace_npe/06_npe_diagnostics.sh

#SBATCH --job-name=ace06-diag
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --array=1-21%7
#SBATCH --output=ace06-diag-%A_%a.out
#SBATCH --error=ace06-diag-%A_%a.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

python -u ace_npe/06_npe_diagnostics.py \
  --cell-index "${SLURM_ARRAY_TASK_ID}" \
  --device cpu
