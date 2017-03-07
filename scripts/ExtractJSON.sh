#!/usr/bin/env bash

if [[ $# -ne 2 ]]
then
    echo "Usage: $0 <srcdir> <destdir>"
    exit
fi

mkdir -p $2

for srcfile in $1/*.cpp
do
    json=$(basename ${srcfile} .cpp).json;
    #        extract forall policy                           | remove reference to loop and convert Policy to json                  | join and convert to "data": [array]
    < ${srcfile} sed -r '/forall.*Pol_/!d;s/^.*<([^>]+).*/\1/;s/^Pol_Id_[0-9]+/{/;s/_Size_/"size":/;s/_Parent_/,"parent":/;s/$/}/;' | paste -sd, | awk '{ print "{\"data\":[" $0 "]}"}' > $2/$json
done
