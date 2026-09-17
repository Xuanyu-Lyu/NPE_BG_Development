#!/bin/bash
# Train/evaluate 100 independent NPEs on Alpine CPU nodes.
# Submit from either the repository root or ace_npe/:
#   sbatch ace_npe/10b_fixed_theta_npe_ensemble.sh
#   sbatch 10b_fixed_theta_npe_ensemble.sh

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

module purge
module load miniforge

# Override at submission if needed, for example:
#   CONDA_ENV=/projects/<project>/<env> sbatch 10b_fixed_theta_npe_ensemble.sh
CONDA_ENV="${CONDA_ENV:-/projects/xuly4739/general_env}"
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
NPE_DEVICE="${NPE_DEVICE:-cpu}"

python -u 10b_fixed_theta_npe_ensemble.py run-one \
    --replicate "$SLURM_ARRAY_TASK_ID" \
    --device "$NPE_DEVICE"
