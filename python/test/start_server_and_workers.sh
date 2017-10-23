#!/bin/bash

ENV=$PWD/hyperloom
SERVER=$HOSTNAME
PORT=9010
WORKERS_FILE=$PBS_NODEFILE
WORKERS=$(cat $WORKERS_FILE)

source activate $ENV
loom-server -p $PORT &
server_pid=$!

for w in $WORKERS
do
    ssh $w -t "source activate $ENV; loom-worker $SERVER $PORT" &
done

wait server_pid
