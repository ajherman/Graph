#!/bin/bash
#SBATCH --job-name my_job
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --partition medium

module load openmpi-3.0.1/gcc-9.2.0

min_k=1
max_k=15
min_n=11
max_n=14
for n in $(seq $min_n $max_n); do
for k in $(seq $min_k $max_k); do
for i in $(seq 0 $((k-1))); do
echo "Running n= $n, k=$k, i=$i..."
srun --ntasks=1 -N 1 python -u clique_generator.py --n $n --k $k --i $i >> clique_log.out 2>> clique_err.out 
done
done 
done

# srun --ntasks=1 -N 1 python -u clique_generator.py --n 13 --k 4 --i 1 >> clique_log.out 2>> clique_err.out 

srun sleep 1
