#!/usr/bin/env python

from make import download
from smurff import smurff
import matrix_io as mio
import unittest


def load_data():
    download()
    train = mio.read_matrix("train.sdm")
    test = mio.read_matrix("test.sdm")

    return (train, test)

class TestExCAPE(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        (cls.train, cls.test) = load_data()

    def test_bpmf(self):
        result = smurff(TestExCAPE.train,
                        Ytest=TestExCAPE.test,
                        priors=['normal', 'normal'],
                        side_info=[None, None],
                        aux_data=[[], []],
                        num_latent = 16,
                        burnin=100,
                        nsamples=200,
                        precision=1.0,
                        verbose=0)

        self.assertAlmostEqual(result.rmse, 1.21, places=2)


if __name__ == "__main__":
    unittest.main()
