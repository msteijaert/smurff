import numpy as np
from scipy import sparse
import scipy.io as sio
import itertools

def avg_sparse_cols(m):
    m = m.tocsc(copy=False)
    sums = m.sum(axis=0).A1
    counts = np.diff(m.indptr)
    assert all(t > 0 for t in counts)
    mean = sums / counts
    sio.mmwrite("col_avg", np.expand_dims(mean, 0))
    return mean

def center_sparse_cols(m, mean):
    m = m.tocsr(copy=False)
    m.data -= np.take(mean, m.indices)
    return m

def avg_sparse_rows(m):
    m = m.tocsr(copy=False)
    sums = m.sum(axis=1).A1
    counts = np.diff(m.indptr)
    assert all(t > 0 for t in counts)
    mean = sums / counts
    return mean

def center_sparse_rows(m,mean):
    out = m.tocsc().copy()
    out.data -= np.take(mean, out.indices)
    return out

def mean(m, mode):
    """compute mode-mean of matrix"""
    if (sparse.issparse(m)):
        if (mode == "rows"):     mean = avg_sparse_rows(m)
        elif (mode == "cols"):   mean = avg_sparse_cols(m)
        elif (mode == "global"): mean = np.mean(m.data)
        elif (mode == "none"):   mean = None
        else:
            raise ValueError("Unknown centering mode: %s" % ( mode ) )
    else:
        if (mode == "cols"):     mean = np.mean(m, 0)
        elif (mode == "rows"):   mean = np.mean(m, 1)
        elif (mode == "global"): mean = np.mean(m)
        elif (mode == "none"):   mean = None
        elif (mode != "none"):
            raise ValueError("Unknown centering mode: %s" % ( mode ) )

    return mean


def center(m, mode, mean):
    """center matrix m according to mode and computed mean"""
    if (sparse.issparse(m)):
        if (mode == "rows"):     m = center_sparse_rows(m,mean)
        elif (mode == "cols"):   m = center_sparse_cols(m,mean)
        elif (mode == "global"): m.data = m.data - mean
        elif (mode == "none"):   pass
        else:
            raise ValueError("Unknown centering mode: %s" % ( mode ) )
    else:
        if (mode == "cols"):     m = m - np.broadcast_to(np.expand_dims(mean, 0), m.shape)
        elif (mode == "rows"):   m = m - np.broadcast_to(np.expand_dims(mean, 1), m.shape)
        elif (mode == "global"): m = m - mean
        elif (mode == "none"):   pass
        elif (mode != "none"):
            raise ValueError("Unknown centering mode: %s" % ( mode ) )

    return m

def std_sparse_rows(m):
    m = m.tocsr(copy=False)
    sums = m.sum(axis=1).A1
    counts = np.diff(m.indptr)
    assert all(t > 0 for t in counts)
    means = sums / counts
    m1 = np.square(m - np.broadcast_to(np.expand_dims(means, 1), m.shape))
    m2 = m1.sum(axis=1).A1
    stds = np.sqrt(m2 / (m1.shape[1] - 1))
    return stds

def scale_sparse_rows(m, std):
    out = m.tocsc().copy()
    out.data /= np.take(std, out.indices)
    return out

def std_sparse_cols(m):
    m = m.tocsc(copy=False)
    sums = m.sum(axis=0).A1
    counts = np.diff(m.indptr)
    assert all(t > 0 for t in counts)
    means = sums / counts
    m1 = np.square(m - np.broadcast_to(np.expand_dims(means, 0), m.shape))
    m2 = m1.sum(axis=0).A1
    stds = np.sqrt(m2 / (m1.shape[1] - 1))
    return stds

def scale_sparse_cols(m, std):
    m = m.tocsr(copy=False)
    m.data /= np.take(std, m.indices)
    return m

def std(m, mode):
    if (sparse.issparse(m)):
        if (mode == "rows"):     return std_sparse_rows(m)
        elif (mode == "cols"):   return std_sparse_cols(m)
        elif (mode == "global"): return np.std(m.data)
        elif (mode == "none"):   return None
        else:
            raise ValueError("Unknown std mode: %s" % ( mode ) )
    if (mode == "cols"):     return np.std(m, 0)
    elif (mode == "rows"):   return np.std(m, 1)
    elif (mode == "global"): return np.std(m)
    elif (mode == "none"):   return None
    else:
        raise ValueError("Unknown std mode: %s" % ( mode ) )

def scale(m, mode, std):
    """scale matrix m according to mode and computed std"""
    if (sparse.issparse(m)):
        if (mode == "rows"):     m = scale_sparse_rows(m, std)
        elif (mode == "cols"):   m = scale_sparse_cols(m, std)
        elif (mode == "global"): m.data = m.data / std
        elif (mode == "none"):   pass
        else:
            raise ValueError("Unknown std mode: %s" % ( mode ) )
    else:
        if (mode == "cols"):     m = m / np.broadcast_to(np.expand_dims(std, 0), m.shape)
        elif (mode == "rows"):   m = m / np.broadcast_to(np.expand_dims(std, 1), m.shape)
        elif (mode == "global"): m = m / std
        elif (mode == "none"):   pass
        elif (mode != "none"):
            raise ValueError("Unknown std mode: %s" % ( mode ) )
    return m
