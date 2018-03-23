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
                "num_latent"      : 16,
                "burnin"          : 200,
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

        cmd = (  "smurff"
                + " --train={datadir}/{train}"
                + " --test={datadir}/{test}" 
                + " --prior={prior_string}"
                + " --side-info={side_info_string}"
                + " --aux-data={aux_data_string}"
                + " --num-latent={num_latent}"
                + " --burnin={burnin}"
                + " --nsamples={nsamples}"
                + " --verbose={verbose}"
                + " --status=stats.csv"
                + " --save-freq={save_freq}"
                + " --save-prefix={save_prefix}"
                + " --noise_model=\"fixed;10;1;1;1\""
                ).format(**args)

        return cmd

    def run_cmd(self, pfx, cmd):
        print(cmd)
        # return
        os.chdir(pfx)

        rmse = float("nan")
        retcode = call(cmd, shell=True)
        try:
            stats = pd.read_csv("stats.csv", sep=";")
            last_row = stats.tail(1)
            rmse = float(last_row['rmse_avg'])
        except Exception as e:
            print(e)

        print(rmse)
        return rmse

    def compare(self, priors, aux_data = ["none","none"], side_info = ["none", "none"]):
        args = self.defaults()
        args["priors"] = priors
        args["aux_data"] = aux_data
        args["side_info"] = side_info


        with tempfile.TemporaryDirectory() as tmpdirname:
             pfx10 = os.path.join(tmpdirname, "10")
             pfx5 = os.path.join(tmpdirname, "5")

             os.mkdir(pfx10)
             args["nsamples"] = 100
             cmd = self.make_cmd(args)
             result10 = self.run_cmd(pfx10, cmd);

             os.mkdir(pfx5)
             args["nsamples"] = 50
             cmd = self.make_cmd(args)
             result5_1 = self.run_cmd(pfx5, cmd);
             cmd = "sed -i -e 's/nsamples = 50/nsamples = 100/' save-options.ini; smurff --root save-root.ini"
             result5_2 = self.run_cmd(pfx5, cmd)
        
           
    def test_normal(self):
        self.compare(["normal", "normal"])

if __name__ == "__main__":
    unittest.main()
