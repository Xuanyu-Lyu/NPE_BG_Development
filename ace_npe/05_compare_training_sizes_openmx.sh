#!/bin/bash
# Compare OpenMx with NPE-100k/200k/300k/500k on STEP 03 test datasets.
# Submit after the STEP 03 aggregation job has completed successfully.

#SBATCH --job-name=ace05-openmx
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=ace05-openmx-%j.out
#SBATCH --error=ace05-openmx-%j.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python --version
python devtools/check_environment.py
Rscript --version
export MPLBACKEND=Agg

python -u ace_npe/05_compare_training_sizes_openmx.py
