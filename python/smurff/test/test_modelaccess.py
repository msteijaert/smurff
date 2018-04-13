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

    def run_session(self):
        Ytrain, Ytest = self.train_test()
        nmodes = len(Ytrain.shape)
        priors = ['normal'] * nmodes

        session = smurff.PySession(priors = priors, num_latent=10,
                burnin=10, nsamples=15, verbose=verbose,
                save_freq = 1, save_prefix = 'test-')

        session.addTrainAndTest(Ytrain, Ytest)

        session.init()
        while session.step():
            pass

        results = session.getResult()

        models = session.getModels()

        pprint(models)

        return results

    # 5 different noise configs

    def test_simples(self):
        result = self.run_session()

if __name__ == '__main__':
    unittest.main()
