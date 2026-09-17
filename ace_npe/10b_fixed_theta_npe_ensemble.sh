#!/bin/bash
# Train/evaluate 100 independent NPEs on Alpine CPU nodes.
# Submit from the repository root:
#   sbatch ace_npe/10b_fixed_theta_npe_ensemble.sh

#SBATCH --job-name=ace10b
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=1-100%10
#SBATCH --output=ace10b-%A_%a.out
#SBATCH --error=ace10b-%A_%a.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
NPE_DEVICE="${NPE_DEVICE:-cpu}"

python -u ace_npe/10b_fixed_theta_npe_ensemble.py run-one \
    --replicate "$SLURM_ARRAY_TASK_ID" \
    --device "$NPE_DEVICE"
