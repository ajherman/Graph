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

# srun -o output/quantum.out --ntasks=1 -N 1 python3 -u main.py --vki 10 4 3 --c 9 --representation quantum --batch-size 1000 --n-steps 2000 --make-movie &


# srun -o output/literal.out --ntasks=1 -N 1 python3 -u main.py --vki 10 4 3 --c 9 --representation literal --lr 0.1 --batch-size 1000 --n-steps 500 --make-movie &

# srun -o output/test.out --ntasks=1 -N 1 python3 -u main.py --vki 10 4 3 --c 9 --representation experimental --lr 0.1 --batch-size 20000 --n-steps 1500 &

srun -o test.out --ntasks=1 -N 1 python3 -u main.py --vki 11 4 0 --c 2 --representation literal --lr 0.1 --batch-size 500 --n-steps 1000 &

# srun -o output/test.out --ntasks=1 -N 1 python3 -u main.py --vki 9 2 0 --c 2 --representation literal --lr 0.1 --batch-size 5000 --n-steps 2000 &


wait



