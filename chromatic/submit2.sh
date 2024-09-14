#!/bin/bash
#SBATCH --job-name my_job
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --partition medium

module load openmpi-3.0.1/gcc-9.2.0


srun -o test.out --ntasks=1 -N 1 python -u enumerate_cliques.py --n 6 --k 6 --i 3


srun sleep 1
