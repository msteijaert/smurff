#!/usr/bin/env python3

import os
import glob
import pandas as pd
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser(description='Compute AUC from CSV file')
parser.add_argument('csv_file', metavar='FILE', nargs=1)
parser.add_argument('--env', metavar='NAME')
args = parser.parse_args()

auc_map = {
        'bpmf': (0.72, 0.03),
        'macau_dense': (0.74, 0.03),
        'macau_sparsebin': (0.63, 0.03),
        'macauone_sparsebin': (0.555, 0.03),
}

def verify_auc(res):
    name = res["test"]
    if name in auc_map:
        (mean, dev) = auc_map[name]
        auc = res["auc"]
        if (auc < mean + dev and auc > mean - dev):
            print("%s / %s : SUCCESS" % (res["env"], res["test"]))
        else:
            print("%s / %s : FAIL (%f < %f < %f)" % (res["env"], res["test"],
                mean-dev, auc, mean+dev))


for f in args.csv_file:
    entries = pd.read_csv(f).to_dict(orient = "index")
    for t in entries.values():
        if args.env and t["env"] not in args.env:
            continue

        verify_auc(t)




