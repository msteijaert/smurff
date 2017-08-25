#!/usr/bin/env python

import argparse
import os
import sys
import subprocess
import itertools
import datetime

parser = argparse.ArgumentParser(description='SMURFF tests')

parser.add_argument('--bin',  metavar='BIN', dest='bin',  nargs='?', help='SMURFF binary', default='smurff')
parser.add_argument('--data', metavar='DIR', dest='datadir', nargs='?', help='Data dir')
parser.add_argument('--out',  metavar='DIR', dest='outdir', nargs='?', help='Log dir', default = datetime.datetime.today().strftime("%Y%m%d-%H%M%S"))
parser.add_argument('--stats',metavar='CSV', dest='statsfile', nargs='?', help='Stats output', default='stats.csv')

args = parser.parse_args()
args.bin = os.path.abspath(args.bin)
args.outdir = os.path.abspath(args.outdir)


try:
    subprocess.call([args.bin, "--help"], stdout=open(os.devnull, 'w'))
except:
    raise Exception("Could not execute SMURFF (" + args.bin + ")");

try:
    os.makedirs(args.outdir)
except:
    raise Exception("Could not create output dir (" + args.outdir + ")");

stats = open(os.path.join(args.outdir, args.statsfile), "w")

def cat(fname, s):
    with open(fname, "w") as f:
        f.write(str(s))

def run_smurf(args):
    cat("args", args)

    args["dir"] = args["datadir"] + "/excape_v3_gfa/compounds_" + str(args["cmpd_count"])

    fmt = """{bin} --train={dir}/fold{fold}_train.sdm --burnin={burnin} \
    --test={dir}/fold{fold}_test.sdm  --nsamples={nsamples} --verbose={verbose} --num-latent={num_latent} \
    --row-prior={row_prior} --col-prior={col_prior} --center={center} --direct --status=stats.csv"""

    if (args["row_features"]): fmt = fmt + " --row-features={dir}/{row_features}"
    if (args["col_features"]): fmt = fmt + " --col-features={dir}/{col_features}"

    cmd = fmt.format(**args)
    cat("cmd", cmd)
    h = "{:021}".format(hash(cmd) + sys.maxsize + 1)

    tgtdir = os.path.join(args["out"], h)
    os.makedirs(tgtdir)
    os.chdir(tgtdir)

    try:
        ps = subprocess.Popen(cmd.split(), stdout=open("stdout", "w"), stderr=open("stderr", "w"))
        ret = ps.wait()
        cat("ret", ret)

        args["rmse"] = float(subprocess.check_output("tail -n 1 stdout | tr -s ' ' | cut -d ' ' -f 4", shell=True))
        args["colmeanrmse"] = float(subprocess.check_output("grep -w colmean stdout |cut -d : -f 2", shell=True))
        args["meanrmse"] = float(subprocess.check_output("grep -w globalmean stdout |cut -d : -f 2", shell=True))
    except KeyboardInterrupt:
        raise
    except:
        args["rmse"] = "failed"
        args["colmeanrmse"] = "failed"
        args["meanrmse"] = "failed"

    stats.write("{cmpd_count};{fold};{center};{row_prior};{col_prior};{row_features};{col_features};{num_latent};{burnin};{nsamples};{meanrmse};{colmeanrmse};{rmse}\n".format(**args))
    stats.flush()

def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def test_gfa_data(args):
    space = {
            'out'         : [ args.outdir ],
            'bin'         : [ args.bin ],
            'datadir'     : [ args.datadir ], 
            'cmpd_count'  : range(500, 3000, 500),
            'fold'        : range(1, 4),
            'num_latent'  : [ 16, 32, 64, 128, 256 ],
            'row_prior'   : [ "normal", "macau" ],
            'col_prior'   : [ "normal", "spikeandslab" ],
            'nsamples'    : [ 200, 800 ],
            'burnin'      : [ 800, 200 ],
            'verbose'     : [ 1 ],
            'center'      : [ "cols", "global", "none" ],
            'row_features': [ "ecfp.sbm", "" ] ,
            'col_features': [ "" ],
    }

    keys = map(str, space.keys())

    runs = list(dict_product(space))
    num = len(runs)
    i = 0
    for e in runs:
        i = i + 1
        print(i, "/", num, "\r", end="")
        run_smurf(e)

    print("\nDone")

test_gfa_data(args)

