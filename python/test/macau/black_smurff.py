#!/usr/bin/env python3

from macau import macau
import scipy.io
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                    help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')


options = [
        { "group": "Priors and side Info"},
        { "name": "row-prior",	      "metavar": "PRIOR", "default": "normal", "help": "One of <normal|spikeandslab|macau|macauone>"},
        { "name": "col-prior",	      "metavar": "PRIOR", "default": "normal", "help": "One of <normal|spikeandslab|macau|macauone>"},
        { "name": "row-features",     "metavar": "FILE",  "help": "side info for rows"},
        { "name": "col-features",     "metavar": "FILE",  "help": "side info for cols"},
        { "name": "row-model",        "metavar": "FILE",  "help": "initialization matrix for row model"},
        { "name": "col-model",        "metavar": "FILE",  "help": "initialization matrix for col model"},
        { "name": "center",           "metavar": "MODE",  "default": "none", "help": "center <global|rows|cols|none>"},
        { "group": "Test and train matrices"},
        { "name": "test",	      "metavar": "FILE", "help":  "test data (for computing RMSE}"},
        { "name": "train",	      "metavar": "FILE", "help":  "train data file", "required": True},
        { "group": "General parameters"},
        { "name": "burnin",	      "metavar": "NUM", "type": float,  "default": 200,  "help":  "200  number of samples to discard"},
        { "name": "nsamples",	      "metavar": "NUM", "type": float,   "default": 800, "help":  "800  number of samples to collect"},
        { "name": "num-latent",	      "metavar": "NUM", "type": float,   "default": 96, "help":  "96  number of latent dimensions"},
        { "name": "restore-prefix",   "metavar": "PATH", "help":  "prefix for file to initialize stae"},
        { "name": "restore-suffix",   "metavar": "EXT",  "help":  "suffix for initialization files (.csv or .ddm}"},
        { "name": "init-model",       "metavar": "NAME", "help":  "One of <random|zero>"},
        { "name": "save-prefix",      "metavar": "PATH", "help":  "prefix for result files"},
        { "name": "save-suffix",      "metavar": "EXT",  "help":  "suffix for result files (.csv or .ddm}"},
        { "name": "save-freq",        "metavar": "NUM", "type": float,  "help":  "save every n iterations (0 == never}"},
        { "name": "threshold",        "metavar": "NUM", "type": float,  "help":  "threshold for binary classification"},
        { "name": "verbose",          "metavar": "NUM", "nargs": "?", "type": float,  "help":  "verbose output (default = 1}"},
        { "name": "quiet",                               "help":  "no output"},
        { "name": "status",           "metavar": "FILE", "help":  "output progress to csv file"},
        { "group": "Noise model"},
        { "name": "precision",	      "metavar": "NUM", "type": float,  "default": 5.0,  "help":  "5.0  precision of observations"},
        { "name": "adaptive",	      "metavar": "NUM,NUM",  "default": "1.0,10.0",  "help":  "1.0,10.0  adavtive precision of observations"},
        { "group": "For the macau prior"},
        { "name": "lambda-beta",      "metavar": "NUM", "type": float,  "default": 10.0, "help":  "10.0  initial value of lambda beta"},
        { "name": "tol",              "metavar": "NUM", "type": float,   "default": 1e-6, "help":  "1e-6  tolerance for CG"},
        { "name": "direct",           "default": False, "help":  "Use Cholesky decomposition i.o. CG Solver"},
]

group = parser
for o in options:
    if ("group" in o):
        group = parser.add_argument_group(o["group"])
    else:
        name = '--' + o["name"]
        del o["name"]
        group.add_argument(name, **o)

args = parser.parse_args()

train_matrix = scipy.io.mmread(args.train)
test_matrix = None
if (args.test):
    test_matrix  = scipy.io.mmread(args.test)
side_rows_matrix = None
side_cols_matrix = None


result = macau(Y          = train_matrix,
               Ytest      = test_matrix,
               num_latent = args.num_latent,
               side = [ side_rows_matrix, side_cols_matrix ],
               verbose    = args.verbose,
               precision  = args.precision,
               burnin     = args.burnin,
               nsamples   = args.nsamples)

