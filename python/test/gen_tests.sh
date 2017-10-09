#!/bin/bash

set -e

DATADIR=$PWD/07_matrices
WORKDIR=$PWD/work
ENVDIR=$PWD/conda_envs

test -d $DATADIR
test -d $ENVDIR

mkdir -p $WORKDIR
cd $WORKDIR

DATE=`date +%Y%m%d_%H%M%S`
RUNDIR=run_${DATE}

mkdir $RUNDIR
rm -f latest
ln -sf $RUNDIR latest
cd $RUNDIR

function run_cmd {
    DIR=$1
    ENV=$2
    CMD=$3

    mkdir $DIR
    cd $DIR
    cat >cmd <<EOF
#!/bin/sh
cd $PWD
source activate $ENV
$CMD >stdout 2>stderr
EOF

    chmod +x cmd
    cd ..
}

function run_version {
    ENV=$1
    DIR=$(basename $ENV)

    mkdir $DIR
    cd $DIR

    TIME="/usr/bin/time --output=time --portability"
    BASE="smurff --burnin 20 --nsamples 200 --num-latent 16 \
        --direct --train $DATADIR/train_sample1_c1.sdm \
        --test $DATADIR/test_sample1_c1.sdm \
        --save-prefix results --save-freq=10 --precision 5.0"

    run_cmd bpmf            $ENV "$TIME $BASE --row-prior normal"
    run_cmd macau_sparsebin $ENV "$TIME $BASE --row-prior macau --row-features $DATADIR/side_sample1_c1_ecfp6_var005.sbm"
    run_cmd macau_dense     $ENV "$TIME $BASE --row-prior macau --row-features $DATADIR/side_sample1_c1_chem2vec.ddm"
    run_cmd macauone_sparsebin $ENV "$TIME $BASE --row-prior macauone --row-features $DATADIR/side_sample1_c1_ecfp6_var005.sbm"
    run_cmd macauone_dense     $ENV "$TIME $BASE --row-prior macauone --row-features $DATADIR/side_sample1_c1_chem2vec.ddm"

    cd ..
}

for env in $ENVDIR/*
do
    run_version $env
done
