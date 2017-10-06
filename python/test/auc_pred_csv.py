#!/usr/bin/env python3

import pandas as pd
from sklearn import metrics
import argparse


parser = argparse.ArgumentParser(description='Compute AUC from CSV file')

parser.add_argument('csv_file', metavar='FILE', nargs=1)
parser.add_argument('--y', metavar='COL', nargs=1, default = "y")
parser.add_argument('--pred', metavar='COL', nargs=1, default = "pred_avg")
parser.add_argument('--threshold', metavar='FLOAT', type = float, nargs='?', const = 5.0)

args = parser.parse_args()

for f in args.csv_file:
    df = pd.read_csv(f)

    if (args.threshold):
        y_vals = df[args.y] > args.threshold
    else:
        y_vals = df[args.y]
        
    fpr, tpr, thresholds = metrics.roc_curve(y_vals, df[args.pred])
    auc = metrics.auc(fpr, tpr)

    print("%f" % (auc))
