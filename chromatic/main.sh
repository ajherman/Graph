
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 1
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch
cores=20

for patch_size in 4 8; do
    for dm in 192 384 768; do
        for h in 8 12; do
            for N in 4 8 12; do
                for lr in 3e-4; do
                    name="results_patch${patch_size}_dm${dm}_h${h}_N${N}_lr${lr}"
                    srun -N 1 -n 1 -c $cores -o $name.out --open-mode=append ./main_wrapper.sh python -u main.py --post-norm 1 --patch-size $patch_size --dm $dm --h $h --N $N --lr $lr &
                done
            done
        done
    done
done


# srun -N 1 -n 1 -c 20 -o $name.out --open-mode=append ./main_wrapper.sh --post-norm 1 --patch-size 4 --dm 256 --h 8 --N 8 --lr 3e-4 &
