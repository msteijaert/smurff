#!/usr/bin/env python

import urllib.request
from zipfile import ZipFile
import pandas
import os
import re
from scipy import sparse
import scipy.io as sio
import matrix_io as mio


url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
zipfile, headers = urllib.request.urlretrieve(url)

def write_matrix(filename, A):
    if sparse.issparse(A):
        mio.write_sparse_float64(filename + ".sdm", A)
    else:
        mio.write_dense_float64(filename + ".ddm", A)

    sio.mmwrite(filename, A)


def to_file(input_file, is_train, no, shape):
    df = pandas.read_csv(input_file, sep='\t', names=["user","item","rating","timestamp"])
    directory = "u%d" % no
    if not os.path.exists(directory): os.makedirs(directory)
    matrix = sparse.coo_matrix((df["rating"], (df["user"]-1, df["item"]-1)), shape = shape)
    fname = os.path.join(directory, "train" if is_train else "test")
    print("Writing %s.*" % fname)
    write_matrix(fname, matrix)

def get_shape(stream):
    for line in stream:
        line = line.decode("utf-8").rstrip().lstrip()
        m = re.match(r"(\d+) users", line)
        if m: users = int(m[1])
        m = re.match(r"(\d+) items", line)
        if m: items = int(m[1])

    return (users, items)

with ZipFile(zipfile, 'r') as z:
    for fname in z.namelist():
        m = re.match(r".+/u.info$", fname)
        if not m: continue
        shape = get_shape(z.open(fname))
        print("%d users, %d items" % shape)

    for fname in z.namelist():
        m = re.match(r".+/u([0-9]).(base|test)$", fname)
        if not m: continue
        is_train = m[2] == "base"
        to_file(fname, is_train, int(m[1]), shape)



