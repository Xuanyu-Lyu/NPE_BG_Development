#!/bin/bash
# Train AE and ACE NPEs in separate Alpine array tasks.
# Submit from the repository root, then submit the plotting job afterok.

#SBATCH --job-name=ace11-fit
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --array=1-2%2
#SBATCH --output=ace11-fit-%A_%a.out
#SBATCH --error=ace11-fit-%A_%a.err

set -euo pipefail
cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development
module load anaconda
conda activate npe-bg

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

case "${SLURM_ARRAY_TASK_ID}" in
  1) model=ae ;;
  2) model=ace ;;
  *) echo "Unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

python -u ace_npe/11_compare_ae_ace_ppc.py \
  --model "${model}" --k 200000 --n 1000 --device cpu
