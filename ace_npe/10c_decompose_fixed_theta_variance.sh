#!/bin/bash
# Decompose posterior-mean variance using the completed STEP 10b evaluation.
# Submit from the repository root after STEP 10b aggregation is complete:
#   sbatch ace_npe/10c_decompose_fixed_theta_variance.sh

#SBATCH --job-name=ace10c-var
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=ace10c-var-%j.out
#SBATCH --error=ace10c-var-%j.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py
export MPLBACKEND=Agg

python -u ace_npe/10c_decompose_fixed_theta_variance.py \
    --ensemble_dir fixed_theta_npe_ensemble_evaluation \
    --output_dir fixed_theta_variance_decomposition \
    --n_replicates 100 \
    --n_blocks 5 \
    --n_posterior_draws 2000
