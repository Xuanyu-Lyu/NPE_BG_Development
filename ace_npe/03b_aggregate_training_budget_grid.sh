#!/bin/bash
# Aggregate STEP 03b cells, make figures, and remove temporary cell files.
# Submit with afterok dependency on the STEP 03b array.

#SBATCH --job-name=ace03b-plot
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=ace03b-plot-%j.out
#SBATCH --error=ace03b-plot-%j.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py
export MPLBACKEND=Agg

python -u ace_npe/03b_training_budget_grid.py --aggregate --device cpu
