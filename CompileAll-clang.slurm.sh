#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -c 36
#SBATCH -J compile-polybench-clang
#SBATCH -t 4:00:00
#SBATCH -p pdebug
#SBATCH --mail-type=ALL
#SBATCH -A killian4
#SBATCH -o /p/lscratchrza/killian4/compile-polybench-clang.out

module load clang/4.0.0

THREADS=72

NJOBS=1

REPS=$(yes '-' | head -n ${NJOBS} | tr '\n' ' ')

ls ./gen/ | paste ${REPS} | while read -r benches
do
    echo ${benches}
    for bench in ${benches}
    do
        srun -n 1 -c 1 $PWD/scripts/CompileVersions.sh clang++ ./gen/${bench} $((THREADS/NJOBS)) &
    done
    wait
done
