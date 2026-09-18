#!/bin/bash
# Submit after the evaluation array, using its job ID:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> \
#       ace_npe/10b_aggregate_fixed_theta_npe_ensemble.sh

#SBATCH --job-name=ace10b-eval-plot
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=ace10b-eval-plot-%j.out
#SBATCH --error=ace10b-eval-plot-%j.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py

python -u ace_npe/10b_fixed_theta_npe_ensemble.py aggregate \
    --output_dir fixed_theta_npe_ensemble_evaluation
