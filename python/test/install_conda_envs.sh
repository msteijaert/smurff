#!/bin/sh

set -e

mkdir conda_envs
cd conda_envs

conda create -p smurff-0.5.0 -c vanderaa --yes smurff=0.5.0 scipy pandas scikit-learn
conda create -p smurff-0.6.1 -c vanderaa --yes smurff=0.6.1 scipy pandas scikit-learn
conda create -p smurff-0.7.0 -c vanderaa --yes smurff=0.7.0 scipy pandas scikit-learn
conda create -p smurff-0.8.1 -c vanderaa --yes smurff=0.8.1 scipy pandas scikit-learn
conda create -p macau-0.5.0  -c vanderaa --yes macau=0.5.0 black_smurff scipy pandas scikit-learn

cd ..
