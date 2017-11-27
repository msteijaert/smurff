#!/bin/sh

set -e

for d in sparse dense
do
    for m in none global cols rows
    do
        outdir="out_${d}_${m}"
        rm -rf $outdir
        mkdir $outdir
        cd $outdir
        ../../center.py --mode=$m ../input/train_${d}.mtx ../input/test.mtx
        diff -ur . ../ref/$outdir
        cd ..
    done

done

