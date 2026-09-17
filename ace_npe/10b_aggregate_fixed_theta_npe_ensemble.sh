#!/bin/bash
# Submit after the training array, using its job ID:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> \
#       ace_npe/10b_aggregate_fixed_theta_npe_ensemble.sh

#SBATCH --job-name=ace10b-plot
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=ace10b-plot-%j.out
#SBATCH --error=ace10b-plot-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

module purge
module load miniforge
CONDA_ENV="${CONDA_ENV:-/projects/xuly4739/general_env}"
conda activate "$CONDA_ENV"

python -u 10b_fixed_theta_npe_ensemble.py aggregate
