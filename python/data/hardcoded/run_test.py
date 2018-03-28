#!/usr/bin/env python

import pandas as pd
import unittest
import os
import tempfile
from subprocess import call


class TestSaveRestore(unittest.TestCase):
    def defaults(self):
        files = [ "aux_0_0.mtx", "aux_1_0.mtx", "feat_0_0.mtx", "feat_1_0.mtx", "test.mtx", "train.mtx", ]
        return {
                "datadir"         : os.getcwd(),
                "test"            : "test.mtx",
                "train"           : "train.mtx",
                "save_freq"       : 1,
                "save_prefix"       : "save",
                "num_latent"      : 2,
                "burnin"          : 400,
                "verbose"         : 0, 
                "aux_data"        : [ "none", "none" ],
                "side_info"       : [ "none", "none" ],
                }


    def add_datadir(self, datadir, l):
        return [ "none" if v == "none" else os.path.join(datadir, v) for v in l ]

    def make_cmd(self, args):
        args["side_info_string"] = " ".join(self.add_datadir(args["datadir"], args["side_info"]))
        args["aux_data_string"] = " ".join(self.add_datadir(args["datadir"], args["aux_data"]))
        args["prior_string"] = " ".join(args["priors"])

        cmd = (  "OMP_NUM_THREADS=1 smurff"
                + " --train={datadir}/{train}"
                + " --test={datadir}/{test}" 
                + " --prior={prior_string}"
                + " --side-info={side_info_string}"
                + " --aux-data={aux_data_string}"
                + " --seed=1234"
                + " --num-latent={num_latent}"
                + " --burnin={burnin}"
                + " --nsamples={nsamples}"
                + " --verbose={verbose}"
                + " --status=stats.csv"
                + " --save-freq={save_freq}"
                + " --save-prefix={save_prefix}"
                + " --noise_model=\"fixed;5;1;1;1\""
                ).format(**args)

        return cmd

    def run_cmd(self, pfx, cmd):
        popdir = os.getcwd()

        os.chdir(pfx)

        rmse = float("nan")
        retcode = call(cmd, shell=True)
        try:
            stats = pd.read_csv("stats.csv", sep=";")
            last_row = stats.tail(1)
            rmse = float(last_row['rmse_avg'])
        except Exception as e:
            print(e)

        os.chdir(popdir)
        
        return rmse

    def compare(self, priors, aux_data = ["none","none"], side_info = ["none", "none"]):
        args = self.defaults()
        args["priors"] = priors
        args["aux_data"] = aux_data
        args["side_info"] = side_info
        nsamples = 100

        with tempfile.TemporaryDirectory() as tmpdirname:
             pfx_aswhole = os.path.join(tmpdirname, "aswhole")
             pfx_twoparts = os.path.join(tmpdirname, "twoparts")

             os.mkdir(pfx_aswhole)
             args["nsamples"] = int(nsamples)
             cmd = self.make_cmd(args)
             result_aswhole = self.run_cmd(pfx_aswhole, cmd);

             os.mkdir(pfx_twoparts)
             args["nsamples"] = int(nsamples-2)
             cmd = self.make_cmd(args)
             result_twoparts_1 = self.run_cmd(pfx_twoparts, cmd);
             cmd = "sed -i -e 's/nsamples = %d/nsamples = %d/' save-options.ini; OMP_NUM_THREADS=1 smurff --root save-root.ini" % (int(nsamples-2), int(nsamples))
             result_twoparts_2 = self.run_cmd(pfx_twoparts, cmd)
             
             self.assertAlmostEqual(result_aswhole, result_twoparts_2, places = 2)
        
           
    def test_normal(self):
        self.compare(["normal", "normal"])
        self.compare(["normal", "normal"], ["aux_0_0.mtx", "none" ])
        self.compare(["normal", "normal"], ["none", "aux_1_0.mtx" ])

    def test_normalone(self):
        self.compare(["normalone", "normalone"])
        self.compare(["normalone", "normalone"], ["aux_0_0.mtx", "none" ])
        self.compare(["normalone", "normalone"], ["none", "aux_1_0.mtx" ])

    def test_spikeandslab(self):
        self.compare(["normalone", "spikeandslab"], ["aux_0_0.mtx", "none" ])
        self.compare(["spikeandslab", "normalone"], ["none", "aux_1_0.mtx" ])

    def test_macau(self):
        self.compare(["macau", "normal"], ["none", "none" ], ["feat_0_0.mtx", "none"])
        self.compare(["normal", "macau"], ["none", "none" ], ["none", "feat_1_0.mtx"])

    def test_macauone(self):
        self.compare(["macauone", "normalone"], ["none", "none" ], ["feat_0_0.mtx", "none"])
        self.compare(["normalone", "macauone"], ["none", "none" ], ["none", "feat_1_0.mtx"])

if __name__ == "__main__":
    unittest.main()
