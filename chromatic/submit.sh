#!/bin/bash
#SBATCH --job-name my_job
#SBATCH --ntasks 10
#SBATCH --nodes 1
#SBATCH --partition gpu
#SBATCH --output output/result.txt
#SBATCH --error output/result.err
#SBATCH --gres=gpu:1
python3 main.py --T 4 --c 2 --representation quantum


