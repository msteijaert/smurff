#!/usr/bin/env python3

import os
import glob
import pandas as pd
from sklearn import metrics
import argparse

def parse_time_file(f):
    with open(f, 'rb') as timefile:
        line = timefile.readline().rstrip()
        real_time = line.split()[1]
        return float(real_time)

def parse_exit_code(fname):
    with open(fname, 'rb') as f:
        line = f.readline().rstrip()
        return int(line)

def parse_predictions_file(f, threshold, val="y", pred="pred_avg"):
    df = pd.read_csv(f)
    df["label"] = df[val] > threshold
        
    fpr, tpr, thresholds = metrics.roc_curve(df["label"], df[pred])
    auc = metrics.auc(fpr, tpr)

    return auc

def find_test_dirs(root="."):
    test_dirs = []
    for filename in glob.iglob('%s/**/exit_code' % root, recursive=True):
        test_dirs.append(os.path.dirname(filename))

    return test_dirs

def process_test_dir(dir):
    try:
        pred_file = max(glob.iglob('%s/*predictions*csv' % dir), key=os.path.getctime)
        auc = parse_predictions_file(pred_file, 5.0)
    except:
        auc = float('nan')

    try:
        real_time = parse_time_file('%s/time' % dir)
    except:
        real_time = float('nan')

    exit_code = parse_exit_code('%s/exit_code' % dir)


    return ( dir, exit_code, auc, real_time )


results = []
for dir in find_test_dirs():
   results.append(process_test_dir(dir))

pd.DataFrame(results, columns = ( "env", "test", "auc", "real_time" ) ).to_csv("results.csv")

