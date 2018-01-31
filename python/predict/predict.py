#!/usr/bin/env python

from math import sqrt
import numpy as np
from scipy import sparse
import scipy.io as sio
import argparse 
import os
import os.path
import matrix_io as mio
from glob import glob
import re
from collections import namedtuple
import csv


parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('--model-prefix',  metavar='PATH', dest='model_prefix', help='prefix of stored model files', required = True)
parser.add_argument('--test',          metavar='FILE', dest='test',         help='Sparse matrix file to use for predictions.')
parser.add_argument('--full', help='Use the full prediction mode', action='store_true')
parser.add_argument('--pred-prefix',   metavar='PATH', dest='pred_prefix',  help='Output prefix for predictions', default='predict')
args = parser.parse_args()
print(args)

ResultItem = namedtuple('ResultItem', ['c0','c1','y','pred_1samp','pred_avg','var','std'])

def init_results(test_file):
    t = mio.read_matrix(test_file)
    (I,J,V) = sparse.find(t)
    results = []
    for r in range(len(I)):
        r = ResultItem(I[r], J[r], V[r], .0, .0, .0, .0)
        results.append(r)

    return results

def predict_some(sample_iter, Us, old_results):
    assert(len(Us) == 2)

    results = []
    for r in old_results:
        pred = np.dot(Us[0][:,r[0]], Us[1][:,r[1]])
        delta = pred - r.pred_avg;
        pred_avg = (r.pred_avg + delta / (sample_iter + 1));
        var = r.var + delta * (pred - pred_avg);
        stds = float("nan") if sample_iter == 0  else sqrt ( var / sample_iter );
        r = ResultItem(r.c0, r.c1, r.y, pred, pred_avg, var, stds)
        results.append(r)

    return results

def write_results(filename, results):
    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerow(('c0','c1','y','pred_1samp','pred_avg','var','std'))
        w.writerows(results)

def predict_all(Us):
    dims = [ U.shape[1] for U in Us ]
    num_latent = Us[0].shape[0]
    return Us[0] * Us[1]


def find_saved_iters(prefix):
    # sorted list of save iterations
    iterations = []

    # find all models and sort them
    for f in glob("%s*U0-latents*" % prefix):
        match=re.search(r"-(\d+)-U0-latents.(\w+)$", f)
        if not match: continue
        iterations.append(int(match[1]))
        ext = match[2]

    iterations.sort()
    return iterations

def get_train_mean(mm_file):
    train = mio.read_matrix(mm_file)
    (I,J,V) = sparse.find(train)
    return V.mean()

def find_Us(prefix, i):
    return [ mio.read_matrix(f) for f in sorted(glob("%s*-%d-U*-latents*" % (prefix, i))) ]

iterations = find_saved_iters(args.model_prefix)
print("Found %d models" % len(iterations))

if not args.full:
    results = init_results(args.test)
    print("Found %d test items"  % len(results))

    for i in iterations:
        Us = find_Us(args.model_prefix, i)
        r = predict_some(i, Us, results)
        write_results("%s-%d-predictions.csv" % ( args.pred_prefix, i ), r)
else:
    models = []
    global_mean = get_train_mean(args.test)
    for i in iterations:
        Us = find_Us(args.model_prefix, i)
        model = np.dot(Us[0].transpose(),Us[1]) + global_mean
        models.append(model)
        mio.write_matrix("%s-%d-full-predictions.ddm" % ( args.pred_prefix, i ), model)

    final_pred = np.average(models, axis=0)
    mio.write_matrix("%s-final-mean.ddm" % ( args.pred_prefix ), final_pred)
    final_std = np.std(models, axis=0)
    mio.write_matrix("%s-final-std.ddm" % ( args.pred_prefix ), final_std)

