#!/usr/bin/env bash

if [[ $# -ne 1 ]]
then
    echo "Usage: $0 <benchmark-headerdir>"
fi

headerdir=$1

bench=$(basename ${headerdir})
outdir="./out/${bench}"
srcdir="./src/"

CXX="g++-5"
CXXFLAGS="-O3 -march=native -std=c++11 -fopenmp -DAUTOTUNING=1"

mkdir -p ${outdir}
for version in $(ls ${headerdir})
do
    outfile=${outdir}/$(basename ${version} .hpp)
    ${CXX} ${CXXFLAGS} -I${headerdir}/${version} -Iinclude ${srcdir}/${bench}.cpp /usr/local/lib/libRAJA.a -o ${outfile}
done
