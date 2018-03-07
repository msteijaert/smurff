from smurff import smurff
import scipy.io

train_matrix_path = "chembl-IC50-346targets.mm"
test_matrix_path = "chembl-IC50-test.mm"

train = scipy.io.mmread(train_matrix_path)
test = scipy.io.mmread(test_matrix_path)

result = smurff(
        train,
        Ytest = test,
        priors = ["normal", "normal"],
        side_info = [ None, None ],
        aux_data =  [ [], [] ],
        num_latent = 1,
        burnin = 1,
        nsamples = 5,
        precision = 5.)

print(result.predictions[0])
