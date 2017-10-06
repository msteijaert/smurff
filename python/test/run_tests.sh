#!/bin/sh

set -e

DATADIR=$PWD/07_matrices
WORKDIR=$PWD/work

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

    echo "env: $ENV"
    echo "cmd: $CMD"
    echo "dir: $DIR"
    source activate $ENV

    mkdir $DIR
    cd $DIR
    echo $CMD >cmd
    set +e
    $CMD >stdout 2>stderr
    LAST_PREDICTIONS=`ls -tl1 results*predictions*csv | head -n 1`
    python3 $WORKDIR/../auc_pred_csv.py --threshold 5.0 $LAST_PREDICTIONS >result
    set -e
    cd ..
}

function run_version {
    ENV=$1
    DIR=$2

    mkdir $DIR
    cd $DIR

    BASE_OPTIONS="time smurff --burnin 20 --nsamples 200 --num-latent 16 \
        --direct --train $DATADIR/train_sample1_c1.sdm \
        --test $DATADIR/test_sample1_c1.sdm \
        --save-prefix results --save-freq=10 --precision 5.0"

    run_cmd bpmf            $ENV "$BASE_OPTIONS --row-prior normal"
    run_cmd macau_sparsebin $ENV "$BASE_OPTIONS --row-prior macau --row-features $DATADIR/side_sample1_c1_ecfp6_var005.sbm"
    run_cmd macau_dense     $ENV "$BASE_OPTIONS --row-prior macau --row-features $DATADIR/side_sample1_c1_chem2vec.ddm"

    cd ..
}

SMURFF_VERSIONS="0.5.0 0.6.1 0.7.0 0.8.0"

for v in $SMURFF_VERSIONS
do
    echo run_version smf-$v smurff-$v
done

run_version mcau-0.5.0 macau-0.5.0


