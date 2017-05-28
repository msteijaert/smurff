
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
#include "latentprior.h"
#include "macauoneprior.h"
#include "omp_util.h"
#include "linop.h"
#include "gen_random.h"

using namespace std; 
using namespace Eigen;

namespace Macau {

enum OPT_ENUM {
    ROW_PRIOR = 1024, COL_PRIOR, ROW_FEATURES, COL_FEATURES, FNAME_ROW_MODEL, FNAME_COL_MODEL, FNAME_TEST, FNAME_TRAIN,
    BURNIN, NSAMPLES, NUM_LATENT, PRECISION, ADAPTIVE, LAMBDA_BETA, TOL, DIRECT,
    RESTORE_PREFIX, RESTORE_SUFFIX,
    SAVE_PREFIX, SAVE_SUFFIX, SAVE_FREQ, THRESHOLD, VERBOSE, CENTER
};

static int parse_opts(int key, char *optarg, struct argp_state *state)
{
    Config &config = *(Config *)(state->input);

    auto set_noise_model = [&config](std::string name, std::string optarg) {
        config.noise_model = name;
        if (name == "adaptive") {
            char *token, *str = strdup(optarg.c_str());
            if(str && (token = strsep(&str, ","))) config.sn_init = strtod(token, NULL); 
            if(str && (token = strsep(&str, ","))) config.sn_max = strtod(token, NULL); 
        } else if (name == "fixed") {
            config.precision = strtod(optarg.c_str(), NULL);
        }
    };

    switch (key) {
        case ROW_PRIOR:       config.row_prior          = optarg; break;
        case COL_PRIOR:       config.col_prior          = optarg; break;
        case ROW_FEATURES:    config.fname_row_features.push_back(optarg); break;
        case COL_FEATURES:    config.fname_col_features.push_back(optarg); break;
        case CENTER:          config.center             = optarg; break;

        case FNAME_TRAIN:     config.fname_train        = optarg; break;
        case LAMBDA_BETA:     config.lambda_beta        = strtod(optarg, NULL); break;
        case BURNIN:          config.burnin             = strtol(optarg, NULL, 10); break;
        case TOL:             config.tol                = atof(optarg); break;
        case DIRECT:          config.direct            = true; break;
        case FNAME_TEST:      config.fname_test         = optarg; break;
        case NUM_LATENT:      config.num_latent         = strtol(optarg, NULL, 10); break;
        case NSAMPLES:        config.nsamples           = strtol(optarg, NULL, 10); break;

        case RESTORE_PREFIX:  config.restore_prefix      = std::string(optarg); break;
        case RESTORE_SUFFIX:  config.restore_suffix      = std::string(optarg); break;
        case SAVE_PREFIX:     config.save_prefix      = std::string(optarg); break;
        case SAVE_SUFFIX:     config.save_suffix      = std::string(optarg); break;
        case SAVE_FREQ:       config.save_freq        = strtol(optarg, NULL, 10); break;

        case PRECISION:       set_noise_model("fixed", optarg); break;
        case ADAPTIVE:        set_noise_model("adaptive", optarg); break;

        case THRESHOLD:       config.threshold          = strtod(optarg, 0); config.classify = true; break;
        case VERBOSE:         config.verbose            = true; break;
        default:              return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

void CmdSession::setFromArgs(int argc, char** argv) {
    /* Program documentation. */
    char doc[] = "ExCAPE Matrix Factorization Framework";

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
        {"test",	     FNAME_TEST    , "NUM",   0, "fraction of train matrix to extract for computing RMSE (e.g. 0.2)"},
        {"train",	     FNAME_TRAIN   , "FILE",  0, "train data file"},
        {0,0,0,0,"General parameters:",3},
        {"burnin",	     BURNIN	, "NUM",   0, "200  number of samples to discard"},
        {"nsamples",	     NSAMPLES	, "NUM",   0, "800  number of samples to collect"},
        {"num-latent",	     NUM_LATENT	, "NUM",   0, "96  number of latent dimensions"},
        {"restore-prefix",   RESTORE_PREFIX	, "PATH",   0, "prefix for file to initialize stae"},
        {"restore-suffix",   RESTORE_SUFFIX	, "EXT",   0, "suffix for initialization files (.csv or .ddm)"},
        {"save-prefix",      SAVE_PREFIX	, "PATH",   0, "prefix for result files"},
        {"save-suffix",      SAVE_SUFFIX	, "EXT",   0, "suffix for result files (.csv or .ddm)"},
        {"save-freq",        SAVE_FREQ	, "NUM",   0, "save every n iterations (0 == never)"},
        {"threshold",        THRESHOLD	, "NUM",   0, "threshold for binary classification"},
        {"verbose",          VERBOSE	, 0,       0, "verbose output"},
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

    Config config;
    struct argp argp = { options, parse_opts, 0, doc };
    argp_parse (&argp, argc, argv, 0, 0, &config);

    //-- check if fname_test is actually a number
    if ((config.test_split = atof(config.fname_test.c_str())) > .0) {
        config.fname_test.clear();
    }

    setFromConfig(config);
}
} // end namespace Macau
