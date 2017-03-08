#!/usr/bin/env bash

for bench in gen/*
do
    $PWD/scripts/CompileVersions.sh ${bench} &
done
