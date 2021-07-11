#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-amnre       # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-amnre_%a.log  # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=06:00:00             # Max execution time
#
#SBATCH --array=1-5                 # Job array

conda activate amsi
cd ~/amsi

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"num_layers": 7, "hidden_size": 256, "activation": "ELU"}'
ADV='{"num_layers": 11, "hidden_size": 512, "activation": "ELU"}'

python train.py -simulator SLCP -samples $SRC/slcp-train.h5 -valid $SRC/slcp-valid.h5 -model "$MODEL" -arbitrary -o $DST/models/slcp-amnre.pth -device cuda
python train.py -simulator SLCP -samples $SRC/slcp-adv.h5 -valid $SRC/slcp-valid.h5 -model "$ADV" -arbitrary -adversary $DST/models/slcp-amnre.pth -lr 1e-4 -min-lr 1e-5 -patience 7 -o $DST/models/slcp-amnre-adv.pth -device cuda

python eval.py $DST/models/slcp-amnre.pth $SRC/slcp-test.h5 -masks =1 =2 =5 -indices 0 3 -o $DST/results/slcp-amnre.csv
python eval.py $DST/models/slcp-amnre.pth $SRC/slcp-test.h5 -masks =1 =2 -indices 0 64 -clean -accuracy -consistency -o $DST/results/slcp-amnre_cons.csv
python eval.py $DST/models/slcp-amnre.pth $SRC/slcp-test.h5 -masks =1 -indices 0 8192 -bins 256 -clean -coverage -o $DST/results/slcp-amnre_cov.csv
python eval.py $DST/models/slcp-amnre-adv.pth $SRC/slcp-test.h5 -masks 10000 11100 11111 -indices 0 0 -classify -o $DST/results/slcp-amnre_roc.csv
