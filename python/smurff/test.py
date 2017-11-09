from smurff import smurff
import scipy.sparse
import scipy.io

#X = scipy.sparse.rand(15, 10, 0.2)
X = scipy.io.mmread("/home/ipasechnikov/chembl-IC50-346targets.mm")
smurff(X, num_latent = 1, burnin=10, nsamples=15)