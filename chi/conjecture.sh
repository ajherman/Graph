#!/bin/bash
#SBATCH --job-name my_job
#SBATCH --ntasks 3
#SBATCH --nodes 3
#SBATCH --partition gpu
#SBATCH --gres=gpu:3
#conda activate pytorch

# srun -o output/test7.out --ntasks=1 -N 1 python3 -u main.py --c 7 --representation quantum --batch-size 5000 --n-steps 5000 &

# srun -o output/test8.out --ntasks=1 -N 1 python3 -u main.py --c 8 --representation quantum --batch-size 5000 --n-steps 5000 &

# srun -o output/test9.out --ntasks=1 -N 1 python3 -u main.py --c 9 --representation quantum --batch-size 5000 --n-steps 5000 &

startv=15
endv=25
# v=13
k=3
i=0
for v in $(seq $startv $endv); do
# for k in $(seq 2 $((v/2))); do
    echo "Running J($v,$k,$i)..."
    srun -o conjecture/K$v-$k.out --ntasks=1 -N 1 python3 -u main.py --vki $v $k $i --c 2 --representation quantum --batch-size 1000 --n-steps 2000 &
# done
done

# v=14
# k=5
# srun -o conjecture/K$v-$k.out --ntasks=1 -N 2 python3 -u main.py --vki $v $k 0 --c 2 --representation literal --batch-size 500 --n-steps 500 &

# srun -o conjecture/counterexample.out --ntasks=1 -N 1 python3 -u main.py --vki 14 3 0 --c 2 --representation quantum --batch-size 1000 --n-steps 200 &


wait



