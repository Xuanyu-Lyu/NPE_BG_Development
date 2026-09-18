#!/bin/bash
# Compare the completed 500-dataset STEP 10b evaluation with the grid posterior.
# Submit from the repository root:
#   sbatch ace_npe/11_compare_fixed_theta_npe_grid.sh

#SBATCH --job-name=ace11-grid
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=ace11-grid-%j.out
#SBATCH --error=ace11-grid-%j.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

Rscript --version

Rscript ace_npe/11_compare_fixed_theta_npe_grid.R \
    --ensemble_dir ace_npe/results/fixed_theta_npe_ensemble_evaluation \
    --output_dir ace_npe/results/fixed_theta_npe_grid_comparison
