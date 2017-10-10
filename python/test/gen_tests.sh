#!/bin/bash

set -e

function create_push { mkdir -p $1; pushd $1 }

DATADIR=$PWD/07_matrices
WORKDIR=$PWD/work
ENVDIR=$PWD/conda_envs

test -d $DATADIR
test -d $ENVDIR

create_push $WORKDIR

DATE=`date +%Y%m%d_%H%M%S`
RUNDIR=run_${DATE}

rm -f latest
ln -sf $RUNDIR latest
create_push $RUNDIR

function run_cmd {
    DIR=$1
    ENV=$2
    CMD=$3

    create_push $DIR

    cat >cmd <<EOF
#!/bin/bash
cd $PWD
source activate $ENV
$CMD >stdout 2>stderr
EOF

    chmod +x cmd
    popd
}

function run_version {
    ENV=$1
    DIR=$(basename $ENV)

    create_push $DIR

    TIME="/usr/bin/time --output=time --portability"
    BASE="smurff --burnin 20 --nsamples 200 --num-latent 16 \
        --direct --train $DATADIR/train_sample1_c1.sdm \
        --test $DATADIR/test_sample1_c1.sdm \
        --save-prefix results --save-freq=10 --precision 5.0"

    run_cmd bpmf               $ENV "$TIME $BASE --row-prior normal"
    run_cmd macau_sparsebin    $ENV "$TIME $BASE --row-prior macau --row-features $DATADIR/side_sample1_c1_ecfp6_var005.sbm"
    run_cmd macau_dense        $ENV "$TIME $BASE --row-prior macau --row-features $DATADIR/side_sample1_c1_chem2vec.ddm"
    run_cmd macauone_sparsebin $ENV "$TIME $BASE --row-prior macauone --row-features $DATADIR/side_sample1_c1_ecfp6_var005.sbm"

    popd
}

for env in $ENVDIR/*
do
    run_version $env
done
