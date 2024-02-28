#!/bin/bash
#SBATCH --job-name my_job
#SBATCH --ntasks 10
#SBATCH --nodes 1
#SBATCH --partition gpu
#SBATCH --output output/result.txt
#SBATCH --error output/result.err
#SBATCH --gres=gpu:1
python3 -u main.py --T 4 --c 9 --representation quantum --batch-size 5000 --n-steps 1000 --make-movie
# python3 -u main.py --T 4 --c 9 --representation softmax --batch-size 5000 --n-steps 1000


