#!/usr/bin/env python

import argparse
import os
from glob import glob
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

args.envs = list(map(os.path.basename,glob("%s/*" % args.envdir)))

defaults = {
        'num_latent'  : 16,
        'row_prior'   : "normal",
        'col_prior'   : "normal",
        'burnin'      : 20,
        'nsamples'    : 200,
        'center'      : "global",
        'row_features': [],
        'col_features': [],
        'direct'      : True,
        'precision'   : 5.0,
        'adaptive'    : None,
}

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

class Test:
    def __init__(self, base, upd = {}):
        self.opts = base.copy()
        self.update(upd)

    def update(self, upd):
        self.opts.update(upd.copy())

    def update_one(self, name, value):
        self.update({name : value})

    def append_one(self, name, value):
        self.opts[name].append(value)

    def gen_cmd(self, outdir, env, datadir):
        args = self.opts

        fmt_cmd = """smurff --train={train} --burnin={burnin} \
        --test={test}  --nsamples={nsamples} --verbose=2 --num-latent={num_latent} \
        --row-prior={row_prior} --col-prior={col_prior} --center={center} --status=stats.csv \
        --save-prefix=results --save-freq=10 \
        """

        if (args["direct"]): fmt_cmd = fmt_cmd + " --direct"
        if (args["precision"]): fmt_cmd = fmt_cmd + " --precision={precision}"
        if (args["adaptive"]): fmt_cmd = fmt_cmd + " --adaptive={adaptive}"
        cmd = fmt_cmd.format(**args)

        for feat in args["row_features"]: 
            cmd += " --row-features=%s" % feat
        for feat in args["col_features"]: 
            cmd += " --col-features=%s" % feat

        fmt_name = "{datasubdir}/{train}/{burnin}/{nsamples}/{num_latent}/{row_prior}/{col_prior}/{center}/"
        fmt_name += "direct/" if (args["direct"]) else "cgsolve/"
        if (args["precision"]): fmt_name +=  "fprec{precision}/"
        if (args["adaptive"]): fmt_name +=  "aprec{adaptive}/"
        fmt_name += "rowf%d/" % len(args["row_features"])
        fmt_name += "colf%d/" % len(args["col_features"])
        name = fmt_name.format(**args)

        fulldir = os.path.join(outdir, name)
        print (fulldir)

        try:
            os.makedirs(fulldir)
        except:
            fulldir += hashlib.md5(cmd.encode('ascii')).hexdigest()[:5] + "/"
            os.makedirs(fulldir)

        os.chdir(fulldir)


        cat("cmd", """
#!/bin/bash
cd %s
/usr/bin/time --output=time --portability \
source activate %s
%s >stdout 2>stderr
echo $? >exit_code
""" % (fulldir, env, cmd))



class TestSuite:
    def __init__(self, name, base = {} , tests = []):
        self.name = name
        self.tests = []
        for t in tests: self.add_test(base, t)

    def __str__(self):
        return "%s: %d tests" % (self.name, len(self.tests))

    def add_test(self, base, upd = {}):
        if isinstance(base, Test): base = base.opts
        self.tests.append(Test(base, upd))
        return self.tests[-1]

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

    def add_noise_options(self):
        tests = self.tests
        self.tests = []
        for t in tests:
            self.add_test(t, { "precision" : 5.0,  "adaptive" : None })
            self.add_test(t, { "precision" : None, "adaptive" : "1.0,10.0" })

def chembl_tests(defaults):
    chembl_defaults = defaults
    chembl_defaults.update({
            'datasubdir' : 'chembl_demo',
            'test'       : 'test_sample1_c1.sdm',
            'train'      : 'train_sample1_c1.sdm',
    })

    chembl_tests_centering = TestSuite("chembl w/ centering", chembl_defaults,
    [
            { },
            { "row_prior": "macau",        "row_features": [ "side_sample1_c1_chem2vec.ddm" ] },
            { "row_prior": "normal",       "row_features": [ "side_sample1_c1_chem2vec.ddm" ] },
            { "col_prior": "spikeandslab", "row_features": [ "side_sample1_c1_chem2vec.ddm" ] },
    ])

    chembl_tests_centering.add_centering_options()

    chembl_tests = TestSuite("chembl", chembl_defaults,
        [
            { "row_prior": "macau",        "row_features": [ "side_sample1_c1_ecfp6_var005.sbm" ] },
            { "row_prior": "macau", "direct": False,  "row_features": [ "side_sample1_c1_ecfp6_var005.sbm" ]},
            { "row_prior": "macauone",     "row_features": [ "side_sample1_c1_ecfp6_var005.sbm" ] },
            { "row_prior": "normal", "center": "none", "row_features": [ "side_sample1_c1_ecfp6_var005.sbm" ] },
            { "col_prior": "spikeandslab", "center": "none", "row_features": [ "side_sample1_c1_ecfp6_var005.sbm" ] },
        ])

    chembl_tests.add_testsuite(chembl_tests_centering)
    chembl_tests.add_noise_options()

    print(chembl_tests)

    return chembl_tests


def synthetic_tests(defaults):
    suite = TestSuite("synthetic")

    priors = [ "normal", "macau", "spikeandslab" ]
    datadirs = glob("%s/synthetic/*" % args.datadir)

    # each datadir == 1 test
    for d in datadirs:
        test = suite.add_test(defaults)
        train_file = list(glob('%s/train.*dm' % d))[0]
        test_file  = os.path.join(d, "test.sdm")
        test.update({ 'train' : train_file, 'test' : test_file, })
        test.update_one("row_features", [])
        test.update_one("col_features", [])
        for f in glob('%s/feat_0_*ddm' % d): test.append_one("row_features", f)
        for f in glob('%s/feat_1_*ddm' % d): test.append_one("col_features", f)

    suite.add_options("row_prior", priors)
    suite.add_options("col_prior", priors)
    suite.add_centering_options()
    suite.add_noise_options()

    print(suite)

    return suite
        
    

def all_tests(args):

    all_tests = chembl_tests(defaults)
    all_tests.add_testsuite(synthetic_tests(defaults))

    return all_tests


tests = all_tests(args).tests

for opts  in tests:
    for env in args.envs:
        fulldir = os.path.join(args.outdir, env)
        fullenv = os.path.join(args.envdir, env)
        opts.gen_cmd(fulldir, fullenv, args.datadir)


