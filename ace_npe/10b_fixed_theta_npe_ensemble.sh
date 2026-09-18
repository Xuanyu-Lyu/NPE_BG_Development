#!/bin/bash
# Reload 100 completed NPEs and evaluate 500 shared datasets on Alpine.
# This job does not train or modify any NPE.
# Submit from the repository root:
#   sbatch ace_npe/10b_fixed_theta_npe_ensemble.sh

#SBATCH --job-name=ace10b-eval
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=1-100%10
#SBATCH --output=ace10b-eval-%A_%a.out
#SBATCH --error=ace10b-eval-%A_%a.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u ace_npe/10b_fixed_theta_npe_ensemble.py evaluate-one \
    --replicate "$SLURM_ARRAY_TASK_ID" \
    --models_dir fixed_theta_npe_ensemble \
    --output_dir fixed_theta_npe_ensemble_evaluation \
    --n_test_simulations 500 \
    --n_posterior_draws 2000
