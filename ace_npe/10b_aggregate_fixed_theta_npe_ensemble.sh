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

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py

python -u ace_npe/10b_fixed_theta_npe_ensemble.py aggregate
