#!/bin/sh

set -e

mkdir -p conda_envs
cd conda_envs

# VERSIONS="0.5.0 0.6.1 0.6.2 0.7.0 0.8.1 0.9.0"
# VERSIONS="0.6.2 0.9.0"
VERSIONS="latest centering"
for V in $VERSIONS
do
    conda create -p smurff-$V -c vanderaa --yes --use-local smurff=$V scipy pandas scikit-learn
done



# conda create -p macau-0.5.0  -c vanderaa --yes macau=0.5.0 black_smurff scipy pandas scikit-learn

cd ..
