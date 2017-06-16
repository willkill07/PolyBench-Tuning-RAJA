#!/usr/bin/env bash

if [[ $# -ne 1 ]]
then
    echo "Usage: $0 <C++ Compiler>"
    exit
fi

compiler=$1

for bench in gen/*
do
    $PWD/scripts/CompileVersions.sh ${bench} ${compiler}&
done
