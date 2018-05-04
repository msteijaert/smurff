#!/usr/bin/env python

import smurff
import matrix_io as mio
import pandas as pd
import unittest
import os
import sys
import tempfile
from subprocess import call
from time import time

global_verbose = 0

class TestExCAPE_py(unittest.TestCase):
    def get_default_opts(self):
        return {
                "priors"          : [ "normal", "normal" ],
                "num_latent"      : 16,
                "burnin"          : 400,
                "nsamples"        : 200,
                "verbose"         : global_verbose, 
                }

    def get_train_noise(self):
        return smurff.FixedNoise(1.0)

    def get_side_noise(self):
        return smurff.FixedNoise(10.)

    @classmethod
    def setUpClass(cls):
        files = [
                "train.sdm",
                "test.sdm",
                "side_c2v.ddm",
                "side_ecfp6_counts_var005.sdm",
                "side_ecfp6_folded_dense.ddm"
                ]

        cls.data = { f : mio.read_matrix(f) for f in files }

    def macau(self, side_info, direct, expected):
        args = self.get_default_opts()

        for d in range(2):
            if side_info[d] != None:
                args["priors"][d] = 'macau'

        session = smurff.TrainSession(**args)
        Ytrain = TestExCAPE_py.data["train.sdm"]
        Ytest = TestExCAPE_py.data["test.sdm"]
        session.addTrainAndTest(Ytrain, Ytest, self.get_train_noise())

        for d in range(2):
            if side_info[d] != None:
                session.addSideInfo(d, TestExCAPE_py.data[side_info[d]], self.get_side_noise(), direct = direct)

        session.init()

        start = time()
        while session.step(): pass
        rmse = session.getRmseAvg()
        stop = time()
        elapsed = stop - start

        self.assertLess(rmse, expected[0])
        self.assertGreater(rmse, expected[1])
        self.assertLess(elapsed, expected[2])

    def test_bpmf(self):
        side_info = [ None, None ]
        self.macau(side_info, True, [ 1.22, 1.10, 120. ])

    def test_macau_c2v(self):
        side_info = [ "side_c2v.ddm", None ]
        self.macau(side_info, True, [1.1, 1.0, 240. ])

    def test_macau_ecfp_sparse_direct(self):
        side_info = [ "side_ecfp6_counts_var005.sdm", None ]
        self.macau(side_info, True, [1.19, 1.0, 1500. ])

    def test_macau_ecfp_sparse_cg(self):
        side_info = [ "side_ecfp6_counts_var005.sdm", None ]
        self.macau(side_info, False, [1.19, 1.0, 480. ])

    def test_macau_ecfp_dense(self):
        side_info = [ "side_ecfp6_folded_dense.ddm", None ]
        self.macau(side_info, True, [ 1.08, 1.0, 240. ])

if __name__ == "__main__":
    for arg in sys.argv:
        if (arg == "-v" or arg == "--verbose"):
            global_verbose = True
    unittest.main()
