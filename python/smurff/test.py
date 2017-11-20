from smurff import smurff
import scipy.sparse
import scipy.io

#train = scipy.sparse.rand(15, 10, 0.2)
train = scipy.io.mmread("/home/ipasechnikov/chembl-IC50-346targets.mm")
test = scipy.io.mmread("/home/ipasechnikov/chembl-IC50-test.mtx")
result = smurff(train, test, num_latent = 1, burnin=1, nsamples=5)
print(result)
