#!/bin/bash
# Submit after the array succeeds:
# job=$(sbatch --parsable ace_npe/12_acde_identification.sh)
# sbatch --dependency=afterok:"$job" ace_npe/12_aggregate_acde_identification.sh

#SBATCH --job-name=acde12-agg
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=acde12-agg-%j.out
#SBATCH --error=acde12-agg-%j.err

set -euo pipefail
cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development
module load anaconda
conda activate npe-bg
export MPLBACKEND=Agg

python -u ace_npe/12_acde_identification.py \
  --aggregate \
  --k 200000 \
  --n-values 50 100 500 1000 2000 5000 20000 \
  --focus-n 1000 \
  --m 1000 \
  --l 2000 \
  --ppc-replicates 500 \
  --device cpu
