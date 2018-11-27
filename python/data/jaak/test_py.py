#!/usr/bin/env python

import smurff
import matrix_io as mio

#load data
ic50 = mio.read_matrix("chembl-IC50-346targets.mm")
ic50_train, ic50_test = smurff.make_train_test(ic50, 0.2)
ic50_threshold = 6.

session = smurff.TrainSession(
                            priors = ['normal', 'normal'],
                            num_latent=32,
                            burnin=10,
                            nsamples=10,
                            # Using threshold of 6. to calculate AUC on test data
                            threshold=ic50_threshold)

## using activity threshold pIC50 > 6. to binarize train data
session.addTrainAndTest(ic50_train, ic50_test, smurff.ProbitNoise(ic50_threshold))
predictions = session.run()
print("RMSE = %.2f" % smurff.calc_rmse(predictions))
print("AUC = %.2f" % smurff.calc_auc(predictions, ic50_threshold))
