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

global_verbose = False

class TestExCAPE_py(unittest.TestCase):
    def get_default_opts(self):
        return {
                "Ytest"           : TestExCAPE_py.data["test.sdm"],
                "priors"          : [ "normal", "normal" ],
                "num_latent"      : 16,
                "burnin"          : 400,
                "nsamples"        : 200,
                "Ynoise"          : ("fixed", 1.0, 0.0, 0.0, 0.0),
                "verbose"         : global_verbose, 
                "aux_data"        : [ [], [] ],
                "side_info"       : [ None, None ],
                "side_info_noises": [ ("fixed", 10.0, 0.0, 0.0, 0.0), ("fixed", 10.0, 0.0, 0.0, 0.0) ],
                }

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
        args["direct"] = direct

        for d in range(2):
            if side_info[d] != None:
                args["side_info"][d] = TestExCAPE_py.data[side_info[d]]
                args["priors"][d] = 'macau'
            else:
                args["side_info"][d] = None


        start = time()
        result = smurff.smurff(TestExCAPE_py.data["train.sdm"], **args)
        stop = time()
        elapsed = stop - start

        self.assertLess(result.rmse, expected[0])
        self.assertGreater(result.rmse, expected[1])
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
