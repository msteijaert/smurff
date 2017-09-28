
#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <cstring>
#include <chrono>
#include <memory>
#include <cmath>
#include <argp.h>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#include <getopt.h>
#include <signal.h>

#include "session.h"
#include "mvnormal.h"
#include "utils.h"
#include "omp_util.h"
#include "linop.h"
#include "gen_random.h"
#include "Data.h"

#include "ILatentPrior.h"
#include "MacauOnePrior.hpp"

using namespace std;
using namespace Eigen;

namespace smurff {

enum OPT_ENUM {
    ROW_PRIOR = 1024, COL_PRIOR, ROW_FEATURES, COL_FEATURES, FNAME_ROW_MODEL, FNAME_COL_MODEL, FNAME_TEST, FNAME_TRAIN,
    BURNIN, NSAMPLES, NUM_LATENT, PRECISION, ADAPTIVE, LAMBDA_BETA, TOL, DIRECT,
    RESTORE_PREFIX, RESTORE_SUFFIX, SAVE_PREFIX, SAVE_SUFFIX, SAVE_FREQ, THRESHOLD, VERBOSE, QUIET,
    INIT_MODEL, CENTER, STATUS_FILE
};

static int parse_opts(int key, char *optarg, struct argp_state *state)
{
    Config &c = *(Config *)(state->input);

    auto set_noise_model = [&c](std::string name, std::string optarg)
    {
        NoiseConfig nc;
        nc.name = name;
        if (name == "adaptive")
        {
            char *token, *str = strdup(optarg.c_str());
            if(str && (token = strsep(&str, ","))) nc.sn_init = strtod(token, NULL);
            if(str && (token = strsep(&str, ","))) nc.sn_max = strtod(token, NULL);
        }
        else if (name == "fixed")
        {
            nc.precision = strtod(optarg.c_str(), NULL);
        }

        // set global noise model
        if (c.train.getNoiseConfig().name == "noiseless")
            c.train.setNoiseConfig(nc);

        //set for row/col feautres
        for(auto &m: c.row_features)
            if (m.getNoiseConfig().name == "noiseless")
               m.setNoiseConfig(nc);

        for(auto &m: c.col_features)
            if (m.getNoiseConfig().name == "noiseless")
               m.setNoiseConfig(nc);
    };

    switch (key)
    {
        case ROW_PRIOR:       c.row_prior          = optarg; break;
        case COL_PRIOR:       c.col_prior          = optarg; break;

        case ROW_FEATURES:    c.row_features.push_back(read_matrix(optarg)); break;
        case COL_FEATURES:    c.col_features.push_back(read_matrix(optarg)); break;
        case CENTER:          c.center_mode        = optarg; break;


        case FNAME_TRAIN:     c.train              = read_sparse(optarg); break;
        case LAMBDA_BETA:     c.lambda_beta        = strtod(optarg, NULL); break;
        case BURNIN:          c.burnin             = strtol(optarg, NULL, 10); break;
        case TOL:             c.tol                = atof(optarg); break;
        case DIRECT:          c.direct             = true; break;
        case FNAME_TEST:      c.test               = read_sparse(optarg); break;
        case NUM_LATENT:      c.num_latent         = strtol(optarg, NULL, 10); break;
        case NSAMPLES:        c.nsamples           = strtol(optarg, NULL, 10); break;

        case RESTORE_PREFIX:  c.restore_prefix      = std::string(optarg); break;
        case RESTORE_SUFFIX:  c.restore_suffix      = std::string(optarg); break;
        case SAVE_PREFIX:     c.save_prefix      = std::string(optarg); break;
        case SAVE_SUFFIX:     c.save_suffix      = std::string(optarg); break;
        case SAVE_FREQ:       c.save_freq        = strtol(optarg, NULL, 10); break;

        case PRECISION:       set_noise_model("fixed", optarg); break;
        case ADAPTIVE:        set_noise_model("adaptive", optarg); break;

        case THRESHOLD:       c.threshold          = strtod(optarg, 0); c.classify = true; break;
        case INIT_MODEL:      c.init_model         = optarg; break;
        case VERBOSE:         c.verbose            = optarg ? strtol(optarg, NULL, 0) : 1; break;
        case QUIET:           c.verbose            = 0; break;
        case STATUS_FILE:     c.csv_status         = optarg; break;
        default:              return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

void CmdSession::setFromArgs(int argc, char** argv) {
    /* Program documentation. */
    char doc[] = "SMURFF: Scalable Matrix Factorization Framework -- http://github.com/ExaScience/smurff";

    struct argp_option options[] = {
        {0,0,0,0,"Priors and side Info:",1},
        {"row-prior",	     ROW_PRIOR     , "PRIOR", 0, "One of <normal|spikeandslab|macau|macauone>"},
        {"col-prior",	     COL_PRIOR	, "PRIOR", 0, "One of <normal|spikeandslab|macau|macauone>"},
        {"row-features",     ROW_FEATURES	, "FILE",  0, "side info for rows"},
        {"col-features",     COL_FEATURES	, "FILE",  0, "side info for cols"},
        {"row-model",        FNAME_ROW_MODEL	, "FILE",  0, "initialization matrix for row model"},
        {"col-model",        FNAME_COL_MODEL	, "FILE",  0, "initialization matrix for col model"},
        {"center",           CENTER	        , "MODE",  0, "center <global|rows|cols|none>"},
        {0,0,0,0,"Test and train matrices:",2},
        {"test",	     FNAME_TEST    , "FILE",  0, "test data (for computing RMSE)"},
        {"train",	     FNAME_TRAIN   , "FILE",  0, "train data file"},
        {0,0,0,0,"General parameters:",3},
        {"burnin",	     BURNIN	, "NUM",   0, "200  number of samples to discard"},
        {"nsamples",	     NSAMPLES	, "NUM",   0, "800  number of samples to collect"},
        {"num-latent",	     NUM_LATENT	, "NUM",   0, "96  number of latent dimensions"},
        {"restore-prefix",   RESTORE_PREFIX	, "PATH",   0, "prefix for file to initialize stae"},
        {"restore-suffix",   RESTORE_SUFFIX	, "EXT",   0, "suffix for initialization files (.csv or .ddm)"},
        {"init-model",       INIT_MODEL	, "NAME",   0, "One of <random|zero>"},
        {"save-prefix",      SAVE_PREFIX	, "PATH",   0, "prefix for result files"},
        {"save-suffix",      SAVE_SUFFIX	, "EXT",   0, "suffix for result files (.csv or .ddm)"},
        {"save-freq",        SAVE_FREQ	, "NUM",   0, "save every n iterations (0 == never)"},
        {"threshold",        THRESHOLD	, "NUM",   0, "threshold for binary classification"},
        {"verbose",          VERBOSE	, "NUM",   OPTION_ARG_OPTIONAL, "verbose output (default = 1)"},
        {"quiet",            QUIET	, 0,   0, "no output"},
        {"status",           STATUS_FILE, "FILE",  0, "output progress to csv file"},
        {0,0,0,0,"Noise model:",4},
        {"precision",	     PRECISION	, "NUM",   0, "5.0  precision of observations"},
        {"adaptive",	     ADAPTIVE	, "NUM,NUM",   0, "1.0,10.0  adavtive precision of observations"},
        {0,0,0,0,"For the macau prior:",5},
        {"lambda-beta",	     LAMBDA_BETA	, "NUM",   0, "10.0  initial value of lambda beta"},
        {"tol",              TOL        , "NUM",   0, "1e-6  tolerance for CG"},
        {"direct",           DIRECT     , 0,   0, "false Use Cholesky decomposition i.o. CG Solver"},
        {0,0,0,0,"General Options:",0},
        {0}
    };

    Config c;
    struct argp argp = { options, parse_opts, 0, doc };
    argp_parse (&argp, argc, argv, 0, 0, &c);

    setFromConfig(c);
}
} // end namespace smurff
