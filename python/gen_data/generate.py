
import numpy as np

# product of two gaussian low-rank matrices + noise
def normal_dense(N, D, K):
    X = np.random.normal(N,K)
    W = np.random.normal(D,K)
    return np.cross(X, W) + np.random.normal(N,D)

# product of two low-rank 'ones' matrices
def ones_dense(N, D, K):
    X = np.ones(N,K)
    W = np.ones(D,K)
    return np.cross(X, W)


