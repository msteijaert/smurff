#!/usr/bin/env python

import argparse
import os
from glob import glob
import sys
import subprocess
import itertools
import datetime

parser = argparse.ArgumentParser(description='SMURFF tests')

parser.add_argument('--time_cmd',  metavar='CMD', dest='time_cmd',  help='Env dir', default='time --output=time --portability')
parser.add_argument('--envdir',  metavar='DIR', dest='envdir',  nargs=1, help='Env dir', default='conda_envs')
parser.add_argument('--data', metavar='DIR', dest='datadir', nargs=1, help='Data dir', default='data')
parser.add_argument('--outdir',  metavar='DIR', dest='outdir', nargs=1, help='Output dir',
        default = 'work/' + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"))

args = parser.parse_args()
args.outdir = os.path.abspath(args.outdir)
args.envdir = os.path.abspath(args.envdir)
args.datadir = os.path.abspath(args.datadir)

center_script = os.path.abspath("./center.py")

args.envs = list(map(os.path.basename,glob("%s/*" % args.envdir)))

defaults = {
        'smurff'      : "smurff",
        'num_latent'  : 16,
        'prior'       : [ "normal", "normal" ],
        'burnin'      : 20,
        'nsamples'    : 50,
        'incenter'    : "global",
        'precenter'   : "none",
        'aux-data'    : [ [], [] ],
        'side-info'   : [ [], [] ],
        'direct'      : True,
        'precision'   : 5.0,
        'adaptive'    : None,
        'test'        : 'test.sdm',
        'train'       : 'train.sdm',
}

print("Generating tests in %s" % args.outdir)

if (os.path.islink("work/latest")):
    os.unlink("work/latest")
os.symlink(args.outdir, "work/latest")


try:
    os.makedirs(args.outdir)
except:
    raise Exception("Could not create output dir (" + args.outdir + ")");

def cat(fname, s):
    with open(fname, "w") as f:
        f.write(str(s))

# joins inner list by ,
# joins outer list by space
# added predix to every inner item
def multi_join(l, prefix = ""):
    print("multi_join: ", l)
    ret = []
    for feat in l:
        print("  feat: ", feat)
        if not feat:
            ret.append("none")
        else: 
            ret.append(",".join([ os.path.join(prefix, f) for f in feat ]))
        print("  ret: ", ret)

    return " ".join(ret)

class Test:
    gen_count = 0

    def __init__(self, base, upd = {}):
        self.opts = base.copy()
        self.update(upd)

    def valid(self):
        print("valid?")
        opts = self.opts
        for i in range(len(opts["prior"])):
            print(i)
            print(opts["prior"][i])
            print(opts["side-info"][i])
            print(opts["aux-data"][i])
            if opts["prior"][i].startswith("macau"): 
                if not opts["side-info"][i]: return False
                if opts["aux-data"][i]: return False
            else:
                if opts["side-info"][i]: return False

        return True
        

    def update(self, upd):
        self.opts.update(upd.copy())

    def update_one(self, name, value):
        self.update({name : value})

    def append_one(self, name, value):
        self.opts[name].append(value)

    def gen_cmd(self, outdir, env, args, makefile = None):
        num_dim = len(self.opts["prior"])
        self.opts["env"] = env

        self.opts["fulldatadir"] = os.path.join(args.datadir, self.opts["datasubdir"])
        self.opts["prior_str"] = " ".join(self.opts["prior"])

        center_cmd = center_script + (" --mode={precenter} --train={fulldatadir}/{train} --test={fulldatadir}/{test}".format(**self.opts))

        smurff_cmd = """{smurff} --train={train} --burnin={burnin} \
        --test={test}  --nsamples={nsamples} --verbose=2 --num-latent={num_latent} \
        --prior={prior_str} --status=stats.csv \
        --aux-data={aux-data_str} --side-info={side-info_str} \
        --save-prefix=results --save-freq=-1 --seed=1234\
        """
    
        for opt in ["aux-data", "side-info" ]:
            self.opts[opt + "_str"] = multi_join(self.opts[opt], self.opts["fulldatadir"])

        if (self.opts["direct"]): smurff_cmd += " --direct"

        # noise arguments should come last, after aux-data!
        if (self.opts["precision"]): smurff_cmd += " --precision={precision}"
        if (self.opts["adaptive"]): smurff_cmd += " --adaptive={adaptive}"

        smurff_cmd = smurff_cmd.format(**self.opts)
        print("cmd: ", smurff_cmd)

        name = "%03d" % Test.gen_count
        Test.gen_count += 1

        fulldir = os.path.join(outdir, name)

        try:
            os.makedirs(fulldir)
        except:
            print("Skipping already existing directory: %s" % fulldir)
            print(self.opts)
            return

        os.chdir(fulldir)

        cat("opts", self.opts)
        cat("name", name)
        cat("env", env)

        cat("cmd", """
#!/bin/bash

set -e

log_exit_code() {
    echo $? >exit_code 
}
trap "log_exit_code" INT TERM EXIT

cd %s
exec >stdout 2>stderr
source activate %s
%s
export OMP_NUM_THREADS=1
%s %s
""" % (fulldir, env, center_cmd, args.time_cmd, smurff_cmd))

        os.chmod("cmd", 0o755)
        if (makefile):
            makefile.write("run: %s/exit_code\n" % fulldir)
            makefile.write("%s/exit_code:\n" % fulldir)
            makefile.write("\tbash -e %s/cmd\n\n" % fulldir)


class TestSuite:
    def __init__(self, name, base = {} , tests = []):
        self.name = name
        self.tests = []
        for t in tests: self.add_test(base, t)

    def filter_tests(self):
        self.tests = [x for x in self.tests if x.valid()]

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
        self.add_options("precenter", ("none", "global", "rows", "cols"))
        self.add_options("incenter", ("none", "global", "rows", "cols"))

    def add_noise_options(self):
        tests = self.tests
        self.tests = []
        for t in tests:
            self.add_test(t, { "precision" : 5.0,  "adaptive" : None })
            self.add_test(t, { "precision" : None, "adaptive" : "1.0,10.0" })

def chembl_tests(defaults):
    chembl_defaults = defaults
    chembl_tests_centering = TestSuite("chembl w/ centering", chembl_defaults,
    [
            { },
            { "row_prior": "macau",        "side-info": [ [ "feat_0_0.ddm" ] , [] ] },
            { "row_prior": "normal",       "aux-data":  [ [ "feat_0_0.ddm" ] , [] ] },
            { "col_prior": "spikeandslab", "aux-data":  [ [ "feat_0_0.ddm" ] , [] ] },
    ])

    # chembl_tests_centering.add_centering_options()

    chembl_tests = TestSuite("chembl", chembl_defaults)
    # chembl_tests = TestSuite("chembl", chembl_defaults,
    #     [
    #         { "row_prior": "macau",        "row_features": [ "feat_0_1.sbm" ] },
    #         { "row_prior": "macau", "direct": False,  "row_features": [ "feat_0_1.sbm" ]},
    #         { "row_prior": "macauone",     "row_features": [ "feat_0_1.sbm" ] },
    #         { "row_prior": "normal", "center": "none", "row_features": [ "feat_0_1.sbm" ] },
    #         { "col_prior": "spikeandslab", "center": "none", "row_features": [ "feat_0_1.sbm" ] },
    #     ])

    chembl_tests.add_testsuite(chembl_tests_centering)
    #chembl_tests.add_noise_options()

    chembl_tests.add_options('datasubdir',
            [
              'chembl_58/sample1/cluster1', # 'chembl_58/sample1/cluster2', 'chembl_58/sample1/cluster3',
     #           'chembl_58/sample2/cluster1', 'chembl_58/sample2/cluster2', 'chembl_58/sample2/cluster3',
     #           'chembl_58/sample3/cluster1', 'chembl_58/sample3/cluster2', 'chembl_58/sample3/cluster3',
            ])

    # chembl_tests.add_options('smurff', ['smurff', 'mpi_smurff'])

    print(chembl_tests)

    return chembl_tests


def synthetic_tests(defaults):
    suite = TestSuite("synthetic")

    priors = [ "normal", "macau", "spikeandslab" ]
    datadirs = glob("%s/synthetic/ones*" % args.datadir)
    datadirs += glob("%s/synthetic/normal*" % args.datadir)

    # each datadir == 1 test
    for d in datadirs:
        test = suite.add_test(defaults)
        test.update_one("datasubdir", os.path.join("synthetic", os.path.basename(d)))
        train_file = os.path.basename(list(glob('%s/train.*dm' % d))[0])
        test_file  = "test.sdm"
        test.update({ 'train' : train_file, 'test' : test_file, })
        test.update_one("row_features", [])
        test.update_one("col_features", [])
        for f in glob('%s/feat_0_*ddm' % d): test.append_one("row_features", os.path.basename(f))
        for f in glob('%s/feat_1_*ddm' % d): test.append_one("col_features", os.path.basename(f))

    suite.add_options("prior", priors)
    suite.add_centering_options()
    suite.add_noise_options()


    print(suite)

    return suite
        
  
def movielens_tests(defaults):
    suite = TestSuite("movielens")
    suite.add_test(defaults)

    datadirs = [ "movielens/u" + i for i in  [1,2,3,4,5]  ]
    suite.add_options('datasubdir', datadirs)

    print(suite)
    return suite

    
  
def jaak_tests(defaults):
    suite = TestSuite("Jaak Simm ChEMBL")

    defaults['datasubdir'] = "jaak"
    defaults['test'] = None

    # chembl-IC50-100compounds-feat-dense.mm
    # chembl-IC50-100compounds-feat.mm
    # chembl-IC50-346targets-01.mm
    # chembl-IC50-346targets-100compounds.mm
    # chembl-IC50-346targets-11.mm
    # chembl-IC50-346targets.mm
    # chembl-IC50-compound-feat.mm

    suite.add_test(defaults, { 'train': 'chembl-IC50-346targets.mm',    'test': 'chembl-IC50-346targets.mm' })
    suite.add_test(defaults, { 'train': 'chembl-IC50-346targets-11.mm', 'test': 'chembl-IC50-346targets-11.mm' })

    print(suite)
    return suite

 
def builtin_tests(defaults):
    suite = TestSuite("Hard-coded values from TestsSmurff.cpp")

    defaults['datasubdir'] = "hardcoded"
    defaults['burnin'] = 50
    defaults['nsamples'] = 50
    defaults['num-latent'] = 4

    suite.add_test(defaults, { 'train': 'train.mtx',    'test': 'test.mtx' })

    priors = [ "normal", "macau", "spikeandslab" ]
    prior_pairs = list( itertools.product(priors, priors) )
    suite.add_options("prior", prior_pairs)
    
    row_features = [ [] , ["feat_0_0.mtx"] ] 
    col_features = [ [] , ["feat_1_0.mtx"] ]
    feature_pairs = list( itertools.product(row_features, col_features))

    row_aux = [ [] , ["aux_0_0.mtx"] ] 
    col_aux = [ [] , ["aux_1_0.mtx"] ]
    aux_pairs = list( itertools.product(row_aux, col_aux))

    suite.add_options("aux-data", aux_pairs)
    suite.add_options("side-info", feature_pairs)

    print(suite)
    return suite
 

def all_tests(args):

    all_tests = builtin_tests(defaults)
    # all_tests = chembl_tests(defaults)
    # all_tests.add_testsuite(jaak_tests(defaults))
    # all_tests.add_testsuite(synthetic_tests(defaults))

    all_tests.filter_tests()

    return all_tests


tests = all_tests(args).tests

print("%d envs" % len(args.envs))

total_num = 0
for env in args.envs:
    fullenv = os.path.join(args.envdir, env)
    makefile = open(os.path.join(args.outdir, "Makefile"), "w")

    for opts in tests:
        total_num += 1
        opts.gen_cmd(args.outdir, fullenv, args, makefile)

    makefile.close()


print("%d total tests" % total_num)
