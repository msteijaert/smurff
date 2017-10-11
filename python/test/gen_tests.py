#!/usr/bin/env python

import argparse
import os
import glob
import sys
import subprocess
import itertools
import datetime

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

def gen_cmd(outdir, env, args):
    os.makedirs(outdir)
    os.chdir(outdir)

    cat("args", args)

    fmt = """{bin} --train={datadir}/{train} --burnin={burnin} \
    --test={datadir}/{test}  --nsamples={nsamples} --verbose={verbose} --num-latent={num_latent} \
    --row-prior={row_prior} --col-prior={col_prior} --center={center} --status=stats.csv \
    --save-prefix={save-prefix} --save-freq={save-freq} \
    """

    if (args["row_features"]): fmt = fmt + " --row-features={datadir}/{row_features}"
    if (args["col_features"]): fmt = fmt + " --col-features={datadir}/{col_features}"
    if (args["direct"]): fmt = fmt + " --direct"
    if (args["precision"]): fmt = fmt + " --precision={precision}"

    cat("cmd", """
#!/bin/bash
cd %s
source activate %s
/usr/bin/time --output=time --portability \
%s >stdout 2>stderr
""" % (outdir, env, fmt.format(**args)))

def all_tests(args):
    all_tests = {}

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
    }

    chembl_defaults = defaults.copy()
    chembl_defaults.update( {
            'train'      : 'chembl_demo/train_sample1_c1.sdm',
            'test'       : 'chembl_demo/test_sample1_c1.sdm',
    })

    chembl_tests = {
        'bpmf'                   : {},
        'macau_sparsebin'        : { "row_prior": "macau",        "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
        'macau_indirect'         : { "row_prior": "macau", "direct": False, "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
        'macau_dense'            : { "row_prior": "macau",        "row_features": "chembl_demo/side_sample1_c1_chem2vec.ddm"     },
        'macauone_sparsebin'     : { "row_prior": "macauone",     "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
        'cofac_sparsebin'        : { "row_prior": "normal", "center": "none", "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
        'cofac_dense'            : { "row_prior": "normal",       "row_features": "chembl_demo/side_sample1_c1_chem2vec.ddm"     },
        'spikeandslab_sparsebin' : { "col_prior": "spikeandslab", "center": "none", "row_features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
        'spikeandslab_dense'     : { "col_prior": "spikeandslab", "row_features": "chembl_demo/side_sample1_c1_chem2vec.ddm"     },
    }

    for name, val in chembl_tests.items():
        opts = chembl_defaults.copy()
        opts.update(val)
        chembl_tests[name] = opts

    


    def add_options(test_name, tests, arg_name, arg_options):
        ret = {}
        for c in arg_options:
            local_opts = tests[test_name].copy()
            local_opts[arg_name] = c;
            ret[test_name + "_" + arg_name + "_" + c] = local_opts

        return ret;

    def add_centering_options(name, tests):
        return add_options(name, tests, "center", ("none", "global", "rows", "cols"))

    def add_adaptive_noise_option(name, tests):
        local_opts = tests[name].copy()
        local_opts["precision"] = None
        local_opts["adaptive"] = "1.0,10.0";
        return { name + "_adaptive" : local_opts }


    for test in ( 'bpmf', 'macau_dense', 'cofac_dense', 'spikeandslab_dense'):
        chembl_tests.update(add_centering_options(test, chembl_tests))

    adaptive_noise_tests = {}
    for test in chembl_tests.keys():
        adaptive_noise_tests.update(add_adaptive_noise_option(test, chembl_tests))

    chembl_tests.update(adaptive_noise_tests)

    all_tests.update(chembl_tests)

    return all_tests


tests = all_tests(args)

for name, opts  in tests.items():
    for env in args.envs:
        fulldir = os.path.join(args.outdir, env, name)
        fullenv = os.path.join(args.envdir, env)
        gen_cmd(fulldir, fullenv, opts)

print("Generated %d x %d tests " % (len(tests), len(args.envs)))

