#!/usr/bin/env python3

import os
import glob
import pandas as pd
from sklearn import metrics
import argparse

def oneline(dir, fname):
    try:
        f = open(os.path.join(dir, fname), 'rb')
        line = f.readline().rstrip()
        return line
    except:
        return ''

def parse_predictions_file(f, threshold, val="y", pred="pred_avg"):
    df = pd.read_csv(f)
    df["label"] = df[val] > threshold
        
    fpr, tpr, thresholds = metrics.roc_curve(df["label"], df[pred])
    auc = metrics.auc(fpr, tpr)
    rmse = metrics.mean_squared_error(df[val], df[pred])

    return (auc, rmse)

def find_test_dirs(root="."):
    test_dirs = []
    for filename in glob.iglob('%s/**/exit_code' % root, recursive=True):
        test_dirs.append(os.path.dirname(filename))

    return test_dirs

def process_test_dir(dir):

    pred_file = max(glob.iglob('%s/*predictions*csv' % dir), key=os.path.getctime)
    print("Processing %s" % pred_file)
    (auc, rmse) = parse_predictions_file(pred_file, 5.0)
    real_time = float(oneline(dir, 'time').split()[1])
    exit_code = int(oneline(dir, 'exit_code'))
    env = os.path.basename(oneline(dir, 'env'))
    name = oneline(dir, 'name')

    return ( env, name, exit_code, auc, rmse, real_time )


results = []
for dir in find_test_dirs():
   results.append(process_test_dir(dir))

pd.DataFrame(results, columns = ( "env", "name", "exit_code", "auc", "rmse", "real_time" ) ).to_csv("results.csv")

