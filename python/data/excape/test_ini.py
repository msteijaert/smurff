#!/usr/bin/env python

import pandas as pd
import unittest
import os
import sys
import tempfile
from subprocess import call
from time import time

global_verbose = False

def extract_rmse(stats_file):
    rmse = float("nan")
    try:
        stats = pd.read_csv(stats_file, sep=";")
        last_row = stats.tail(1)
        rmse = float(last_row['rmse_avg'])
    except Exception as e:
        print(e)

    return rmse

class TestExCAPE_ini(unittest.TestCase):
    def ini(self, ini, expected):
            from_dir = os.getcwd()
            link_files = [
                    "bpmf.ini",
                    "macau-c2v.ini",
                    "macau-ecfp-dense.ini",
                    "macau-ecfp-sparse-cg.ini",
                    "macau-ecfp-sparse-direct.ini",
                    "side_c2v.ddm",
                    "side_ecfp6_counts_var005.sdm",
                    "side_ecfp6_folded_dense.ddm",
                    "test.sdm",
                    "train.sdm",
                ]
            with tempfile.TemporaryDirectory() as tmpdirname:
                for f in link_files:
                    os.symlink(os.path.join(from_dir, f), os.path.join(tmpdirname, f))

                start = time()
                call("smurff --ini " + ini, shell=True, cwd=tmpdirname)
                stop = time()
                elapsed = stop - start
                rmse = extract_rmse(os.path.join(tmpdirname, "save-status.csv"))

            self.assertLess(rmse, expected[0])
            self.assertGreater(rmse, expected[1])
            self.assertLess(elapsed, expected[2])

    def test_bpmf(self):
        self.ini("bpmf.ini", [ 1.22, 1.10, 120. ])
    def test_macau_c2v(self):
        self.ini("macau-c2v.ini", [1.1, 1.0, 240. ])
    def test_macau_ecfp_sparse_direct(self):
        self.ini("macau-ecfp-sparse-direct.ini", [1.19, 1.0, 1500. ])
    def test_macau_ecfp_sparse_cg(self):
        self.ini("macau-ecfp-sparse-cg.ini", [1.19, 1.0, 900. ])
    def test_macau_ecfp_dense(self):
        self.ini("macau-ecfp-dense.ini", [1.08, 1.0, 240. ])
 
if __name__ == "__main__":
    for arg in sys.argv:
        if (arg == "-v" or arg == "--verbose"):
            global_verbose = True
    unittest.main()
