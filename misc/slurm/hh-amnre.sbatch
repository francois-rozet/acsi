#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-amnre         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-amnre_%a.log    # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=03:00:00             # Max execution time
#
#SBATCH --array=1-5                 # Job array

conda activate amsi
cd ~/amsi

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"num_layers": 7, "hidden_size": 256, "activation": "ELU"}'
ADV='{"num_layers": 11, "hidden_size": 512, "activation": "ELU"}'

python train.py -simulator HH -samples $SRC/hh-train.h5 -valid $SRC/hh-valid.h5 -model "$MODEL" -arbitrary -o $DST/models/hh-amnre.pth -device cuda
python train.py -simulator HH -samples $SRC/hh-adv.h5 -valid $SRC/hh-valid.h5 -model "$ADV" -arbitrary -adversary $DST/models/hh-amnre.pth -lr 1e-4 -min-lr 1e-5 -patience 7 -o $DST/models/hh-amnre-adv.pth -device cuda

python eval.py $DST/models/hh-amnre.pth $SRC/hh-event.h5 -masks =1 =2 =8 -o $DST/results/hh-amnre-event.csv
python eval.py $DST/models/hh-amnre.pth $SRC/hh-test.h5 -masks =1 =2 -indices 0 64 -clean -consistency -o $DST/results/hh-amnre_cons.csv
python eval.py $DST/models/hh-amnre.pth $SRC/hh-test.h5 -masks =1 -indices 0 8192 -bins 256 -clean -coverage -o $DST/results/hh-amnre_cov.csv
python eval.py $DST/models/hh-amnre-adv.pth $SRC/hh-test.h5 -masks 11000000 11100100 11110101 -indices 0 0 -classify -o $DST/results/hh-amnre_roc.csv
