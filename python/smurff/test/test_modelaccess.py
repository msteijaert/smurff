import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff
from pprint import pprint

verbose = 0

class TestNoiseModels(unittest.TestCase):
    # Python 2.7 @unittest.skip fix
    __name__ = "TestNoiseModels"

    def train_test(self):
       Y = scipy.sparse.rand(15, 10, 0.2)
       Y, Ytest = smurff.make_train_test(Y, 0.5)
       return Y, Ytest

    def run_train_session(self):
        Ytrain, Ytest = self.train_test()
        nmodes = len(Ytrain.shape)
        priors = ['normal'] * nmodes

        session = smurff.TrainSession(priors = priors, num_latent=10,
                burnin=10, nsamples=15, verbose=verbose,
                save_freq = 1)

        session.addTrainAndTest(Ytrain, Ytest)

        session.init()
        while session.step():
            pass

        return session

    # 5 different noise configs

    def test_simple(self):
        train_session = self.run_train_session()
        predict_session = train_session.makePredictSession()
        print(predict_session)

if __name__ == '__main__':
    unittest.main()
