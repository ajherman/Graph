#!/bin/bash
#SBATCH --job-name my_job
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --partition medium

module load openmpi-3.0.1/gcc-9.2.0

max_k=15
n=6

for k in $(seq 2 $max_k); do
for i in $(seq 1 $((k-1))); do
echo "Running k=$k, i=$i..."
srun -o cliques/$n-test-$k-$i.out --ntasks=1 -N 1 python -u enumerate_cliques.py --n $n --k $k --i $i
done
done 

# srun -o test.out --ntasks=1 -N 1 python -u enumerate_cliques.py -n 6 -k 12 -i 6 &

srun sleep 1
