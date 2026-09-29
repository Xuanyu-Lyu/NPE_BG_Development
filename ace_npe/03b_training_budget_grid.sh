#!/bin/bash
# Train/evaluate one transient (K,N) model per Alpine array task.
# Submit from the repository root with: sbatch ace_npe/03b_training_budget_grid.sh

#SBATCH --job-name=ace03b-grid
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --array=1-49%10
#SBATCH --output=ace03b-grid-%A_%a.out
#SBATCH --error=ace03b-grid-%A_%a.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -u ace_npe/03b_training_budget_grid.py \
  --cell-index "${SLURM_ARRAY_TASK_ID}" \
  --device cpu
