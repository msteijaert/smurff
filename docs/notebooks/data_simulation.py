# Synthetic data generation  for evaluating MAUCAU algorithm

## Basic ideas: 
##  1. implant a number of biclusters in a matrix. Each row of a bicluser is sampled from the same distribution
##  2. geneate different set of features for each bicluster.

import numpy as np

def gen_bicluster(nrows, ncols, mu, sigma = 1.0):
    bicluster = sigma * np.random.randn(nrows, ncols) + mu
    return bicluster

def gen_side_feas(nrows, ncols, p):
    fea = np.random.binomial(1, p, size = (nrows * ncols)).reshape(nrows, ncols)
    return (fea)

def gen_matrix(nrows, ncols, nfeatures, nbiclusters, fea_prob = 0.1):
    
    bic_nrows = round(nrows/nbiclusters)
    bic_ncols = round(ncols/nbiclusters)
    bic_nfeas = round(nfeatures/nbiclusters)
    
    np.random.seed(1234)
    bic_mus = 5*np.random.randn(nbiclusters)
    
    bics = [gen_bicluster(bic_nrows, bic_ncols, mu) for mu in bic_mus]
    feas = [gen_side_feas(bic_nrows, bic_nfeas, fea_prob) for mu in bic_mus]
    
    matrix = np.random.randn(nrows, ncols)
    sinfo  = np.random.binomial(1, 0.01, size = (nrows * nfeatures)).reshape(nrows, nfeatures)
    
    c = 0
    r = 0
    s = 0
    
    for i in range(nbiclusters):
        matrix[r : (r + bic_nrows), c : (c + bic_ncols)] = bics[i]
        sinfo[r: (r + bic_nrows), s : (s + bic_nfeas)]   = feas[i]
        r = r + bic_nrows
        c = c + bic_ncols
        s = s + bic_nfeas
    
    return {'matrix':matrix, 'sinfo':sinfo}

def sparsify(matrix, sparsity = 0.2):
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    n = nrows * ncols
    m = round((1-sparsity)*n)
    pos = np.random.choice(n, m, replace = False)
    
    smatrix = matrix.copy()
    
    for p in pos:
        smatrix[(p // ncols), p - (p // ncols)*ncols] = 0
    return (smatrix)
    

# plt.imshow(m1, cmap='jet')
# plt.colorbar()