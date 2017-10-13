#!/usr/bin/env python

import argparse
import os
import glob
import sys
import subprocess
import itertools
import datetime
import hashlib

parser = argparse.ArgumentParser(description='SMURFF tests')

parser.add_argument('--envdir',  metavar='DIR', dest='envdir',  nargs=1, help='Env dir', default='conda_envs')
parser.add_argument('--data', metavar='DIR', dest='datadir', nargs=1, help='Data dir', default='data')
parser.add_argument('--outdir',  metavar='DIR', dest='outdir', nargs=1, help='Output dir',
        default = 'work/' + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"))

args = parser.parse_args()
args.outdir = os.path.abspath(args.outdir)
args.envdir = os.path.abspath(args.envdir)
args.datadir = os.path.abspath(args.datadir)

args.envs = list(map(os.path.basename,glob.glob("%s/*" % args.envdir)))

print("Generating tests in %s" % args.outdir)

if (os.path.islink("latest")):
    os.unlink("latest")
os.symlink(args.outdir, "latest")


try:
    os.makedirs(args.outdir)
except:
    raise Exception("Could not create output dir (" + args.outdir + ")");

def cat(fname, s):
    with open(fname, "w") as f:
        f.write(str(s))

def gen_cmd(outdir, env, test):
    args = test.opts

    fmt = """{bin} --train={datadir}/{train} --burnin={burnin} \
    --test={datadir}/{test}  --nsamples={nsamples} --verbose={verbose} --num-latent={num_latent} \
    --row-prior={row_prior} --col-prior={col_prior} --center={center} --status=stats.csv \
    --save-prefix={save-prefix} --save-freq={save-freq} \
    """

    if (args["row_features"]): fmt = fmt + " --row-features={datadir}/{row_features}"
    if (args["col_features"]): fmt = fmt + " --col-features={datadir}/{col_features}"
    if (args["direct"]): fmt = fmt + " --direct"
    if (args["precision"]): fmt = fmt + " --precision={precision}"
    if (args["adaptive"]): fmt = fmt + " --adaptive={adaptive}"

    cmd = fmt.format(**args)
    name = hashlib.md5(cmd.encode('ascii')).hexdigest()
    fulldir = os.path.join(outdir, name)

    os.makedirs(fulldir)
    os.chdir(fulldir)


    cat("cmd", """
#!/bin/bash
cd %s
source activate %s
/usr/bin/time --output=time --portability \
%s >stdout 2>stderr
""" % (fulldir, env, fmt.format(**args)))


class Test:
    def __init__(self, base, upd = {}):
        self.opts = base.copy()
        self.update(upd)

    def update(self, upd):
        self.opts.update(upd.copy())

    def update_one(self, name, value):
        self.update({name : value})


class TestSuite:
    def __init__(self, base, tests = []):
        self.tests = []
        for t in tests: self.add_test(base, t)

    def add_test(self, base, upd = {}):
        if isinstance(base, Test): base = base.opts
        self.tests.append(Test(base, upd))

    def add_testsuite(self, suite):
        self.tests += suite.tests

    def update(self, upd):
        for t in self.tests:
            t.update(upd)

    def add_options(self, arg_name, arg_options):
        tests = self.tests
        self.tests = []
        for t in tests:
            for c in arg_options:
                self.add_test(t, { arg_name : c })

    def add_centering_options(self):
        return self.add_options("center", ("none", "global", "rows", "cols"))

    def add_adaptive_noise_options(self):
        tests = self.tests
        self.tests = []
        for t in tests:
            self.add_test(t, { "precision" : 5.0,  "adaptive" : None })
            self.add_test(t, { "precision" : None, "adaptive" : "1.0,10.0" })

def chembl_tests(defaults):
    chembl_defaults = defaults
    chembl_defaults.update({
            'train'      : 'chembl_demo/train_sample1_c1.sdm',
            'test'       : 'chembl_demo/test_sample1_c1.sdm',
    })

    chembl_tests_centering = TestSuite(chembl_defaults,
    [
            { 'name': 'bpmf',                   },
            { 'name': 'macau_dense',            "row_prior": "macau",        "row_features": "chembl_demo/side_sample1_c1_chem2vec.ddm"     },
            { 'name': 'cofac_dense',            "row_prior": "normal",       "row_features": "chembl_demo/side_sample1_c1_chem2vec.ddm"     },
            { 'name': 'spikeandslab_dense',     "col_prior": "spikeandslab", "row_features": "chembl_demo/side_sample1_c1_chem2vec.ddm"     },
    ])

    chembl_tests_centering.add_centering_options()

    chembl_tests = TestSuite(chembl_defaults,
        [
            { 'name': 'macau_sparsebin',        "row_prior": "macau",        "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
            { 'name': 'macau_indirect',         "row_prior": "macau", "direct": False, "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
            { 'name': 'macauone_sparsebin',     "row_prior": "macauone",     "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
            { 'name': 'cofac_sparsebin',        "row_prior": "normal", "center": "none", "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
            { 'name': 'spikeandslab_sparsebin', "col_prior": "spikeandslab", "center": "none", "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
        ])

    chembl_tests.add_testsuite(chembl_tests_centering)
    chembl_tests.add_adaptive_noise_options()

    return chembl_tests


def synthetic_tests(defaults):
    synth_defaults = defaults.copy()

    priors = [ "normal", "macau", "spikeandslab" ]
    datadirs = glob.glob("%s/synthetic/*" % defaults.datadir)

    for name, val in synth_tests.items():
        opts = synth_defaults.copy()
        opts.update(val)
        synth_tests[name] = opts

    for test in synth_tests.keys():
        synth_tests.update(add_centering_options(test, synth_tests))

    adaptive_noise_tests = {}
    for test in synth_tests.keys():
        adaptive_noise_tests.update(add_adaptive_noise_option(test, synth_tests))

    synth_tests.update(adaptive_noise_tests)

def all_tests(args):
    defaults = {
            'bin'         : 'smurff',
            'datadir'     : args.datadir, 
            'num_latent'  : 16,
            'row_prior'   : "normal",
            'col_prior'   : "normal",
            'burnin'      : 20,
            'nsamples'    : 200,
            'verbose'     : 1,
            'center'      : "global",
            'row_features': "",
            'col_features': "",
            'direct'      : True,
            'save-prefix' : 'results',
            'save-freq'   : 10,
            'precision'   : 5.0,
            'adaptive'    : None,
    }

    all_tests = chembl_tests(defaults)

    return all_tests


tests = all_tests(args).tests

for opts  in tests:
    for env in args.envs:
        fulldir = os.path.join(args.outdir, env)
        fullenv = os.path.join(args.envdir, env)
        gen_cmd(fulldir, fullenv, opts)

print("Generated %d x %d tests " % (len(tests), len(args.envs)))

