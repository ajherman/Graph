#!/bin/bash
#SBATCH --job-name my_job
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --mem=0
#SBATCH --partition long

module load openmpi-3.0.1/gcc-9.2.0

# min_k=7
# max_k=7
max_j=3
max_i=40
min_n=12
max_n=16
for n in $(seq $min_n $max_n); do
for j in $(seq 0 $max_j); do
for i in $(seq 0 $max_i); do
k=$((i + j))
if [[ $k -eq 0 ]]; then
  continue
fi
echo "Running n= $n, k=$k, i=$i..."
srun --ntasks=1 -N 1 python -u clique_generator.py --n $n --k $k --i $i --find_bounds --find_min_v >> clique_log.out 2>> clique_err.out 
done
done 
done

# srun --ntasks=1 -N 1 python -u clique_generator.py --n 11 --k 9 --i 8 --find_bounds --find_min_v >> clique_log.out 2>> clique_err.out 



# srun --ntasks=1 -N 1 python -u clique_generator.py --n 5 --k 4 --i 2 --find_min_sum --find_bounds >> test.out 2>> test.err

srun sleep 1
