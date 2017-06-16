#!/usr/bin/env bash

if [[ $# -ne 3 ]]
then
    echo "Usage: $0 <C++ Compiler> <benchmark-headerdir> <jobs>"
    exit
fi

CXX=$1
headerdir=$2
compiler=$(basename $CXX)

bench=$(basename ${headerdir})
outdir="out-${compiler}/${bench}"
srcdir="./src/"

rajadir=$HOME/Software/RAJA-${compiler}

NJOBS=$3

WORKING_DIR=$(mktemp -d)/${outdir}
FINAL_DIR=/p/lscratchrza/killian4/polybench/bin/${outdir}

mkdir -p ${FINAL_DIR} ${WORKING_DIR}

REPS=$(yes '-' | head -n ${NJOBS} | tr '\n' ' ')

ls ${headerdir} | paste ${REPS} | while read -r versions
do
    echo "${bench}: ${versions}"
    for version in ${versions}
    do
        outfile=$(basename ${version} .hpp)
        if [ ! -f ${FINAL_DIR}/${outfile} ]
        then
            ${CXX} \
                -O3 -march=native -std=c++11 -fopenmp -DAUTOTUNING=1 \
                -I${headerdir}/${version} -I${PWD}/include -I${rajadir}/include \
                ${srcdir}/${bench}.cpp \
                ${rajadir}/lib/libRAJA.a \
                -o ${WORKING_DIR}/${outfile} &
        fi
    done
    wait
    for version in ${versions}
    do
        outfile=$(basename ${version} .hpp)
        if [ -f ${WORKING_DIR}/${outfile} ]
        then
            mv ${WORKING_DIR}/${outfile} ${FINAL_DIR}/
        fi
    done
done
