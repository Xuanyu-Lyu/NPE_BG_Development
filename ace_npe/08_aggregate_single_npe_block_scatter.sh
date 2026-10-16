#!/bin/bash
# Submit after 08_single_npe_block_scatter.sh completes successfully.

#SBATCH --job-name=ace08-single-bw-plot
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=ace08-single-bw-plot-%j.out
#SBATCH --error=ace08-single-bw-plot-%j.err

set -euo pipefail

cd /pl/active/IBG/collab/NPE_Fei_Xy/NPE_BG_Development

module load anaconda
conda activate npe-bg

python devtools/check_environment.py
export MPLBACKEND=Agg

python -u ace_npe/08_fixed_theta_npe_ensemble.py aggregate-blocks \
    --model_dir fixed_theta_npe_ensemble/replicate_001 \
    --output_dir fixed_theta_single_npe_blocks \
    --n_blocks 500 \
    --datasets_per_block 100 \
    --n_posterior_draws 2000
