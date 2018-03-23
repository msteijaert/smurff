#!/usr/bin/env python

import smurff
import matrix_io as mio
import unittest
import os
import tempfile


class TestSaveRestore(unittest.TestCase):
    def load_data(self):
        files = [ "aux_0_0.mtx", "aux_1_0.mtx", "feat_0_0.mtx", "feat_1_0.mtx", "test.mtx", "train.mtx", ]
        self.data = dict()
        for f in files:
            self.data[f] = mio.read_matrix(f)

    def smurff(self, priors, aux_data = [[],[]], side_info = [None, None]):
        self.load_data()
        print(priors)

        with tempfile.TemporaryDirectory() as tmpdirname:
             result10 = smurff.smurff(self.data["train.mtx"],
                                 Ytest = self.data["test.mtx"],
                                 priors = priors,
                                 aux_data = aux_data,
                                 side_info = side_info,
                                 num_latent = 16,
                                 burnin = 10,
                                 nsamples = 10,
                                 precision = 5.0,
                        );
        
             print(result10.rmse)
           
    def test_normal(self):
        self.smurff(["normal", "normal"])

if __name__ == "__main__":
    unittest.main()
