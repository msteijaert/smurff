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
    args = open(os.path.join(dir, "args"), 'r').read()
    exit_code = int(oneline(dir, 'exit_code'))

    try:
        pred_file = max(glob.iglob('%s/*predictions*csv' % dir), key=os.path.getctime)
        print("Processing %s" % pred_file)
        (auc, rmse) = parse_predictions_file(pred_file, 5.0)
        real_time = float(oneline(dir, 'time').split()[1])
    except:
        print("Failed %s" % dir)
        auc = rmse = real_time = -1

    result = {
        "auc" : auc,
        "rmse" : rmse,
        "real_time" : real_time,
        "exit_code" : exit_code,
        "time": os.path.getctime(os.path.join(dir, 'exit_code'))
    }

    run = {
        "args" : eval(args),
        "result": result
        
    }

    with open(os.path.join(dir, "result"), "w") as f:
        f.write(str(result))

    return run

def find_test_centring_dirs(data_type, centring, root="."):
    test_dirs = []
    for filename in glob.iglob('%s/*/*_%s/*/*/*/*/*/*/%s/*/*/*/*/exit_code' % (root, data_type, centring), recursive=True):
        test_dirs.append(os.path.dirname(filename))

    return test_dirs

def collect_centring(filename, root0, data_type0, centring0, root1, data_type1, centring1):
    results = []
    for dir in find_test_centring_dirs(data_type0, centring0, root0):
        results.append(process_test_dir(dir))

    for dir in find_test_centring_dirs(data_type1, centring1, root1):
        results.append(process_test_dir(dir))

    pd.DataFrame(results, columns = ( "env", "name", "exit_code", "auc", "rmse", "real_time" ) ).to_csv(filename)

#collect_centring('result0.csv', './smurff-latest', 'none', 'none', './smurff-centering', 'none', 'none')
#collect_centring('result1.csv', './smurff-latest', 'col', 'none', './smurff-centering', 'col', 'none')
#collect_centring('result2.csv', './smurff-latest', 'row', 'none', './smurff-centering', 'row', 'none')
#collect_centring('result3.csv', './smurff-latest', 'global', 'none', './smurff-centering', 'global', 'none')

#collect_centring('result4.csv', './smurff-latest', 'none', 'cols', './smurff-centering', 'col', 'none')
#collect_centring('result5.csv', './smurff-latest', 'none', 'rows', './smurff-centering', 'row', 'none')
#collect_centring('result6.csv', './smurff-latest', 'none', 'global', './smurff-centering', 'global', 'none')

results = []
for dir in find_test_dirs():
    results.append(process_test_dir(dir))

def cat(fname, s):
    with open(fname, "w") as f:
        f.write(str(s))

pd.DataFrame(results).to_csv("results.csv")
cat("results.pydata", results)
