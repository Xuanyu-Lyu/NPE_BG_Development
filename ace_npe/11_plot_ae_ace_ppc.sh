#!/bin/bash
# Plot Step 11 after both model tasks finish successfully.

#SBATCH --job-name=ace11-plot
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=ace11-plot-%j.out
#SBATCH --error=ace11-plot-%j.err

set -euo pipefail
cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development
module load anaconda
conda activate npe-bg
export MPLBACKEND=Agg

python -u ace_npe/11_compare_ae_ace_ppc.py \
  --plot --k 200000 --n 1000 --device cpu
