#!/usr/bin/env python

from make import download
import smurff
import matrix_io as mio
import pandas as pd
import unittest
import os
import tempfile
from subprocess import call
from time import time


def smurff_py(args):
    allowed_args = [ "Ytest", "num_latent", "burnin", "nsamples", "precision", "verbose",
            "aux_data", "side_info", "direct", "priors", "lambda_beta"]
    filtered_args = { k: args[k] for k in allowed_args}
    return smurff.smurff(args["train"], **filtered_args)

def smurff_cmd(args):
    args["side_info_string"] = " ".join(args["side_info_files"])
    args["aux_data_string"] = " ".join(args["aux_data_files"])
    args["prior_string"] = " ".join(args["priors"])
    args["direct_string"] = "--direct" if args["direct"] else ""

    cmd = (  "smurff"
             + " --train={datadir}/{train_file}"
             + " --test={datadir}/{test_file}" 
             + " --prior={prior_string}"
             + " --side-info={side_info_string}"
             + " --lambda-beta={lambda_beta}"
             + " --aux-data={aux_data_string}"
             + " {direct_string}"
             + " --num-latent={num_latent} --burnin={burnin} --nsamples={nsamples}"
             + " --precision={precision} --verbose={verbose} --status=stats.csv"
             ).format(**args)

    class Result:
         pass

    result = Result()
    with tempfile.TemporaryDirectory() as tmpdirname:
        for f in [ args["train_file"], args["test_file"] ] + args["side_info_files"] + args["aux_data_files"]:
            if f != "none":
                os.symlink(os.path.join(args["datadir"], f), os.path.join(tmpdirname, f))
    
        result.retcode = call(cmd, shell=True, cwd=tmpdirname)
        result.rmse = float("nan")
        try:
            stats = pd.read_csv(os.path.join(tmpdirname, "stats.csv"), sep=";")
            last_row = stats.tail(1)
            result.rmse = float(last_row['rmse_avg'])
        except:
            pass

    return result
        
def load_data():
    # download()

    files = [
            "train.sdm",
            "test.sdm",
            "side_c2v.ddm",
            "side_ecfp6_counts_var005.sdm",
            "side_ecfp6_folded_dense.ddm"
            ]

    data = dict()
    for f in files:
        data[f] = mio.read_matrix(f)

    return data

class TestExCAPE(unittest.TestCase):

    def get_default_opts(self):
        return {
                "datadir"         : os.getcwd(),
                "test_file"       : "test.sdm",
                "train_file"      : "train.sdm",
                "num_latent"      : 16,
                "burnin"          : 100,
                "nsamples"        : 200,
                "precision"       : 1.0,
                "verbose"         : 0, 
                "aux_data_files"  : [ "none", "none" ],
                "side_info_files" : [ "none", "none" ],
                "direct"          : True,
                "lambda_beta"     : 5.0,
                }

    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        

    def files_to_data(self, args):

        args["train"] = TestExCAPE.data[args["train_file"]]
        args["Ytest"] = TestExCAPE.data[args["test_file"]]

        for t in ["side_info", "aux_data"]:
            args[t] = []
            for f in args[t + "_files"]:
                if f != "none":
                    args[t].append(TestExCAPE.data[f])
                else:
                    args[t].append(None)

        return args

    def time_smurff(self, args, expected):
        args = self.files_to_data(args)

        for smurff in [ smurff_cmd, smurff_py ]:
            start = time()
            result = smurff(args)
            stop = time()
            elapsed = stop - start

            self.assertLess(result.rmse, expected[0])
            self.assertGreater(result.rmse, expected[1])
            self.assertLess(elapsed, expected[2])

    def bpmf(self, args, expected):
        args["priors"] = ["normal", "normal"]
        args["side_info_files"] = [ "none", "none" ]
        self.time_smurff(args, expected)

    def macau(self, side_info_files, args, expected):
        priors = [ 'normal', 'normal' ]
        for d in range(2):
            if side_info_files[d] != "none":
                priors[d] = 'macau'

        args["priors"] = priors
        args["side_info_files"] = side_info_files
        args["aux_data"] =  [ [], [] ]

        self.time_smurff(args, expected)

    def test_bpmf(self):
        params = self.get_default_opts()
        self.bpmf(params, [ 1.22, 1.10, 120. ])

    def test_macau_c2v(self):
        params = self.get_default_opts()
        side_info = [ "side_c2v.ddm", "none" ]
        self.macau(side_info, params, [1.08, 1.0, 240. ])

    def test_macau_ecfp_sparse_direct(self):
        params = self.get_default_opts()
        side_info = [ "side_ecfp6_counts_var005.sdm", "none" ]
        self.macau(side_info, params, [1.17, 1.0, 900. ])

    def test_macau_ecfp_sparse_cg(self):
        params = self.get_default_opts()
        params["direct"] = False
        side_info = [ "side_ecfp6_counts_var005.sdm", "none" ]
        self.macau(side_info, params, [1.17, 1.0, 480. ])

    def test_macau_ecfp_dense(self):
        params = self.get_default_opts()
        side_info = [ "side_ecfp6_folded_dense.ddm", "none" ]
        self.macau(side_info, params, [ 1.08, 1.0, 240. ])

if __name__ == "__main__":
    unittest.main()
