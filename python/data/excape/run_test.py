#!/usr/bin/env python

from make import download
import smurff
import matrix_io as mio
import unittest
import os
from time import time


def load_data():
    download()

    files = [
            "train.sdm",
            "test.sdm",
            "side_c2v.ddm",
            "side_ecfp6_counts_var005.sdm",
            "side_ecfp6_folded_dense.ddm"
            ]

    data = dict()
    for f in files:
        key = os.path.splitext(f)[0]
        data[key] = mio.read_matrix(f)

    return data


def smurff_cmdline(Ytrain, **kwargs):
    pass
    
def time_smurff(Ytrain, **kwargs):
    start = time()
    result = smurff.smurff(Ytrain, **kwargs)
    stop = time()
    result.time = stop - start
    return result

def smurff_bpmf(Ytrain, **kwargs):
    kwargs["priors"] = ["normal", "normal"]
    kwargs["side_info"] = [ None, None ]
    kwargs["aux_data"] =  [ [], [] ]
    return time_smurff(Ytrain, **kwargs)

def smurff_macau(Ytrain, side_info, **kwargs):
    priors = [ 'normal', 'normal' ]
    for d in range(2):
        if not side_info[d] is None:
            priors[d] = 'macau'

    kwargs["priors"] = priors
    kwargs["side_info"] = side_info
    kwargs["aux_data"] =  [ [], [] ]

    return time_smurff(Ytrain, **kwargs)


class TestExCAPE(unittest.TestCase):
    
    default_opts = {
            "num_latent" : 16,
            "burnin"     : 100,
            "nsamples"   : 200,
            "precision"  : 1.0,
            "verbose"    : 0, 
        }

    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_smurff_bpmf(self):
        result = smurff_bpmf(
                        TestExCAPE.data["train"],
                        Ytest      = TestExCAPE.data["test"],
                        **TestExCAPE.default_opts)

        self.assertLess(result.rmse, 1.22)
        self.assertGreater(result.rmse, 1.10)
        self.assertLess(result.time, 60.)

    def test_smurff_macau_c2v(self):
        result = smurff_macau(
                          TestExCAPE.data["train"],
                        [ TestExCAPE.data["side_c2v"], None ],
                        Ytest      = TestExCAPE.data["test"],
                        **TestExCAPE.default_opts)

        self.assertLess(result.rmse, 1.08)
        self.assertGreater(result.rmse, 1.0)
        self.assertLess(result.time, 120.)

    def test_smurff_macau_ecfp_sparse_direct(self):
        opts = TestExCAPE.default_opts
        opts["direct"] = True
        result = smurff_macau(
                          TestExCAPE.data["train"],
                        [ TestExCAPE.data["side_ecfp6_counts_var005"], None ],
                        Ytest      = TestExCAPE.data["test"],
                        **TestExCAPE.default_opts)

        self.assertLess(result.rmse, 1.17)
        self.assertGreater(result.rmse, 1.0)
        self.assertLess(result.time, 60.)

    def test_smurff_macau_ecfp_sparse_cg(self):
        opts = TestExCAPE.default_opts
        opts["direct"] = False
        result = smurff_macau(
                          TestExCAPE.data["train"],
                        [ TestExCAPE.data["side_ecfp6_counts_var005"], None ],
                        Ytest      = TestExCAPE.data["test"],
                        **TestExCAPE.default_opts)

        self.assertLess(result.rmse, 1.17)
        self.assertGreater(result.rmse, 1.0)
        self.assertLess(result.time, 360.)

    def test_smurff_macau_ecfp_dense(self):
        result = smurff_macau(
                          TestExCAPE.data["train"],
                        [ TestExCAPE.data["side_ecfp6_folded_dense"], None ],
                        Ytest      = TestExCAPE.data["test"],
                        **TestExCAPE.default_opts)

        self.assertLess(result.rmse, 1.08)
        self.assertGreater(result.rmse, 1.0)
        self.assertLess(result.time, 120.)


if __name__ == "__main__":
    unittest.main()
