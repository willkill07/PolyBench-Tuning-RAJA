#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 36
#SBATCH -J run-polybench-clang
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -o /p/lscratchrza/killian4/run-polybench-clang.out

module load clang/4.0.0

ITERATIONS=3
COMPILER_SHORT_NAME=g++
export OMP_NUM_THREADS=18
export OMP_PLACES=cores
export OMP_PROC_BIND=close
CORRECTNESS_BINARY=$HOME/PolyBench-Tuning-RAJA/scripts/Correctness

BIN_DIR=/p/lscratchrza/killian4/polybench/bin/out-${COMPILER_SHORT_NAME}
DEFAULT_DIR=/p/lscratchrza/killian4/polybench/data/default
OUT_DIR=/p/lscratchrza/killian4/polybench/data/out-${COMPILER_SHORT_NAME}

# DO NOT REALLY EDIT PAST HERE

DEFAULT_IN=${DEFAULT_DIR}/in
DEFAULT_OUT=${DEFAULT_DIR}/out

TMP_DIR=$(mktemp -d)

LOCAL_IN=${TMP_DIR}/default/in

mkdir -p ${LOCAL_IN}
echo "cp -r ${DEFAULT_IN} ${TMP_DIR}/default"
cp -r ${DEFAULT_IN} ${TMP_DIR}/default
echo "cp ${CORRECTNESS_BINARY} ${LOCAL_IN}/Correctness"
cp ${CORRECTNESS_BINARY} ${LOCAL_IN}/Correctness

LOCAL_OUT=${TMP_DIR}/data
counts=$(seq ${ITERATIONS})

for i in ${counts}
do
    mkdir -p ${LOCAL_OUT}/out${i}
done

cd ${BIN_DIR}
for bench in $(ls | sort)
do
    kernel_data=$(ls ${DEFAULT_OUT}/${bench}.cpp*.bin | sort)
    find ./${bench} -not -type d -and -executable | awk "NR%${2}==(${1}-1)" | while read version
    do
        mkdir -p $(dirname ${OUT_DIR}/${version})
        if [[ ! -f "${OUT_DIR}/${version}.json" ]] && [[ ! -f "${OUT_DIR}/${version}.txt" ]]
        then
            TIMEARGS=()
            for iter in ${counts}
            do
                DATA_DIR=${LOCAL_IN} OUTPUT_DIR=${LOCAL_OUT}/out${iter} ${version}
                TIMEARGS+=("$(ls ${LOCAL_OUT}/out${iter}/*.txt)")
            done
            ARGS=()
            for base in ${kernel_data}
            do
                arr=$(basename ${base})
                for iter in ${counts}
                do
                    ARGS+=("${base}")
                    ARGS+=("${LOCAL_OUT}/out${iter}/${arr}")
                done
            done
            if [[ ! -f "${OUT_DIR}/${version}.json" ]]
            then
                ${LOCAL_IN}/Correctness ${ARGS[*]} > ${OUT_DIR}/${version}.json
            fi
            if [[ ! -f "${OUT_DIR}/${version}.txt" ]]
            then
                cat ${TIMEARGS[*]} | awk '{sum+=$1}END{print sum/NR}' > ${OUT_DIR}/${version}.txt
            fi
            rm -f ${LOCAL_OUT}/out${iter}/*
        fi
    done
done
