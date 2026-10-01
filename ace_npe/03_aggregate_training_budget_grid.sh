#!/bin/bash
# Aggregate STEP 03 cells, make figures, and remove temporary cell files.
# Submit with afterok dependency on the STEP 03 array.

#SBATCH --job-name=ace03-plot
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=ace03-plot-%j.out
#SBATCH --error=ace03-plot-%j.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py
export MPLBACKEND=Agg

python -u ace_npe/03_training_budget_grid.py --aggregate --device cpu
