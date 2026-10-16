#!/bin/bash
# One saved N=1000 NPE, 500 independent blocks of 100 fixed-theta datasets.
# Submit from the repository root, then run the aggregation job after success.

#SBATCH --job-name=ace08-single-bw
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=1-500%20
#SBATCH --output=ace08-single-bw-%A_%a.out
#SBATCH --error=ace08-single-bw-%A_%a.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python devtools/check_environment.py
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u ace_npe/08_fixed_theta_npe_ensemble.py evaluate-block \
    --block "$SLURM_ARRAY_TASK_ID" \
    --model_dir fixed_theta_npe_ensemble/replicate_001 \
    --output_dir fixed_theta_single_npe_blocks \
    --n_blocks 500 \
    --datasets_per_block 100 \
    --n_posterior_draws 2000
