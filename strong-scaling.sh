#!/bin/bash
#SBATCH -J mv_strong
#SBATCH -A gen243
#SBATCH -N 8
#SBATCH --ntasks-per-node=32
#SBATCH -t 00:10:00
#SBATCH -o logs/strong_%j.out

module purge
module load PrgEnv-gnu
module load cray-python/3.11.7
module load cray-mpich

N=50000
#N=15360

for P in 32 64 128 256; do
    nodes=$((P/32))
    echo "---- P=$P (NODES=$nodes) ----"
    srun -N $nodes -n $P python mv_strong.py $N
