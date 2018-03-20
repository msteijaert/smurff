#!/bin/bash

#ENVS="smurff-0.10 smurff-local-openblas smurff-local-mkl"
ENVS="smurff-local-intel"

set -e

for E in $ENVS
do
    source activate $E

    set +x

    echo $E macau 1
    /usr/bin/time -f %e smurff --train train.sdm --test test.sdm --prior macau normal \
    --side-info side_c2v.ddm none --aux-data none none --precision 1. --direct \
    --verbose 0 --lambda-beta 5 --num-latent=16 --burnin=1 --nsamples=1 >/dev/null

    echo $E macau 100
    /usr/bin/time -f %e smurff --train train.sdm --test test.sdm --prior macau normal \
    --side-info side_c2v.ddm none --aux-data none none --precision 1. --direct \
    --verbose 0 --lambda-beta 5 --num-latent=16 --burnin=100 --nsamples=100 >/dev/null

    echo $E bpmf 1
    /usr/bin/time -f %e smurff --train train.sdm --test test.sdm --prior normal normal \
    --side-info none none --aux-data none none --precision 1. --direct \
    --verbose 0 --lambda-beta 5 --num-latent=16 --burnin=1 --nsamples=1 >/dev/null

    echo $E bpmf 100
    /usr/bin/time -f %e smurff --train train.sdm --test test.sdm --prior normal normal \
    --side-info none none --aux-data none none --precision 1. --direct \
    --verbose 0 --lambda-beta 5 --num-latent=16 --burnin=100 --nsamples=100 >/dev/null

    set +x
done
