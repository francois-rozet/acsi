#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-data        # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-data.log      # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=01:00:00             # Max execution time
#

conda activate amsi
cd ~/amsi

SRC=$SCRATCH/samples

python sample.py -simulator SLCP -seed 0 -samples 1048576 -o $SRC/slcp-train.h5
python sample.py -simulator SLCP -seed 1 -samples 131072 -o $SRC/slcp-test.h5
python sample.py -simulator SLCP -seed 2 -samples 131072 -o $SRC/slcp-valid.h5
python sample.py -simulator SLCP -seed 3 -samples 1048576 -o $SRC/slcp-adv.h5
