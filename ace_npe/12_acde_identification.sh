#!/bin/bash
# Submit from the repository root: sbatch ace_npe/12_acde_identification.sh
# One K=200,000 ACDE NPE per array task; seven fixed-N cells.

#SBATCH --job-name=acde12
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --array=1-7%7
#SBATCH --output=acde12-%A_%a.out
#SBATCH --error=acde12-%A_%a.err

set -euo pipefail
cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development
module load anaconda
conda activate npe-bg

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

python -u ace_npe/12_acde_identification.py \
  --cell-index "${SLURM_ARRAY_TASK_ID}" \
  --k 200000 \
  --n-values 50 100 500 1000 2000 5000 20000 \
  --focus-n 1000 \
  --m 1000 \
  --l 2000 \
  --ppc-replicates 500 \
  --device cpu
