#!/usr/bin/env bash

dir=./out
if [[ $# -eq 1 ]]
then
    dir=$1
fi

find ${dir} -not -type d -and -executable | while read binary
do
    output=$(echo $binary | sed 's/out/asm/')
    if [[ ! -f ${output} ]]
    then
        mkdir -p $(dirname $output)
        ./scripts/ObjDumpKernels.sh ${binary} > ${output}
    fi
done
