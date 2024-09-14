#!/bin/bash
#SBATCH --job-name my_job
#SBATCH --ntasks 3
#SBATCH --nodes 3
#SBATCH --partition gpu
#SBATCH --gres=gpu:3
#conda activate pytorch

# srun -o output/test.out --ntasks=1 -N 1 python3 -u main.py --vki 5 2 0 --lam 0.5 --representation literal --lr 0.1 --batch-size 1000 --n-steps 1000 &

srun -o output/test.out --ntasks=1 -N 1 python3 -u main.py --vki 14 3 2 --lam 0.01 --representation literal --lr 0.1 --batch-size 1000 --n-steps 500 &

wait



