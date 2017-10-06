#!/bin/bash

DIRS=$*

for d in $DIRS
do
   LAST_PREDICTIONS=`ls -t1 $d/results*predictions*csv 2>/dev/null | head -n 1`
   auc=nan
   [ -n "$LAST_PREDICTIONS" ] && auc=`python3 auc_pred_csv.py --threshold 5.0 $LAST_PREDICTIONS`

   real=`grep ^real $d/time | cut -d ' ' -f 2`
   echo "$d;$auc;$real"
done
