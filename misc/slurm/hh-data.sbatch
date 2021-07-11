#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-data          # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-data.log        # Log-file
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=2G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=2-00:00:00           # Max execution time
#

conda activate amsi
cd ~/amsi

SRC=$SCRATCH/samples

python sample.py -simulator HH -seed 0 -moments -events -o $SRC/hh-event.h5
python sample.py -simulator HH -seed 1 -moments -reference $SRC/hh-event.h5 -samples 1048576 -o $SRC/hh-train.h5
python sample.py -simulator HH -seed 2 -moments -reference $SRC/hh-event.h5 -samples 131072 -o $SRC/hh-test.h5
python sample.py -simulator HH -seed 3 -moments -reference $SRC/hh-event.h5 -samples 131072 -o $SRC/hh-valid.h5
python sample.py -simulator HH -seed 4 -moments -reference $SRC/hh-event.h5 -samples 1048576 -o $SRC/hh-adv.h5
