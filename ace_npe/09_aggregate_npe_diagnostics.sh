#!/bin/bash
# Aggregate STEP 09 cells, create figures, and remove temporary cell files.
# Submit with an afterok dependency on the STEP 09 array.

#SBATCH --job-name=ace09-plot
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=ace09-plot-%j.out
#SBATCH --error=ace09-plot-%j.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py
export MPLBACKEND=Agg

python -u ace_npe/09_npe_diagnostics.py --aggregate --device cpu
