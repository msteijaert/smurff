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

    fmt = """{bin} --train={train} --burnin={burnin} \
    --test={test}  --nsamples={nsamples} --verbose={verbose} --num-latent={num_latent} \
    --row-prior={row_prior} --col-prior={col_prior} --center={center} --status=stats.csv"""

    if (args["row_features"]): fmt = fmt + " --row-features={dir}/{row_features}"
    if (args["col_features"]): fmt = fmt + " --col-features={dir}/{col_features}"
    if (args["direct"]): fmt = fmt + " --direct"

    header = "#!/bin/bash"
    cd_cmd = "cd %s" % outdir
    env_cmd = "source activate %s" % env
    smurff_cmd = fmt.format(**args)
    cat("cmd", "\n".join((header, cd_cmd, env_cmd, smurff_cmd)))

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
    }

    chembl_defaults = defaults.copy()
    chembl_defaults.update( {
            'train'      : 'chembl_demo/train_sample1_c1.sdm',
            'test'       : 'chembl_demo/test_sample1_c1.sdm',
    })

    chembl_tests = {
        'bpmf'              : {},
        'macau_sparsebin'   : { "row-prior": "macau",    "row-features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
        'macau_dense'       : { "row-prior": "macau",    "row-features": "chembl_demo/side_sample1_c1_chem2vec.ddm"     },
        'macauone_sparsebin': { "row-prior": "macauone", "row-features": "chembl_demo/side_sample1_c1_ecfp6_var005.sbm" },
    }

    for t in chembl_tests.values():
        t.update(chembl_defaults)

    return chembl_tests


tests = all_tests(args)

for name, opts  in tests.items():
    for env in args.envs:
        fulldir = os.path.join(args.outdir, env, name)
        fullenv = os.path.join(args.envdir, env)
        gen_cmd(fulldir, fullenv, opts)

print("Generated %d x %d tests " % (len(tests), len(args.envs)))

