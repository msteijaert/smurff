from smurff import smurff
import scipy.io

train_matrix_path = "/home/ipasechnikov/chembl-IC50-346targets.mm"
test_matrix_path = "/home/ipasechnikov/chembl-IC50-test.mtx"

train = scipy.io.mmread(train_matrix_path)
test = scipy.io.mmread(test_matrix_path)
result_items = smurff(train, test, num_latent = 1, burnin=1, nsamples=5)
print(result_items[0])
