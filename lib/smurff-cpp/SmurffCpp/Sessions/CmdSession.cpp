#include <string>
#include <iostream>
#include <sstream>

#include <boost/program_options.hpp>

#include "CmdSession.h"

#include <SmurffCpp/Sessions/SessionFactory.h>
#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Version.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>

#define HELP_NAME "help"
#define ROW_PRIOR_NAME "row-prior"
#define COL_PRIOR_NAME "col-prior"
#define ROW_FEATURES_NAME "row-features"
#define COL_FEATURES_NAME "col-features"
//#define ROW_MODEL_NAME "row-model"
//#define COL_MODEL_NAME "col-model"
//#define CENTER_NAME "center"
#define TEST_NAME "test"
#define TRAIN_NAME "train"
#define BURNIN_NAME "burnin"
#define NSAMPLES_NAME "nsamples"
#define NUM_LATENT_NAME "num-latent"
#define RESTORE_PREFIX_NAME "restore-prefix"
#define RESTORE_SUFFIX_NAME "restore-suffix"
#define INIT_MODEL_NAME "init-model"
#define SAVE_PREFIX_NAME "save-prefix"
#define SAVE_SUFFIX_NAME "save-suffix"
#define SAVE_FREQ_NAME "save-freq"
#define THRESHOLD_NAME "threshold"
#define VERBOSE_NAME "verbose"
#define QUIET_NAME "quiet"
#define VERSION_NAME "version"
#define STATUS_NAME "status"
#define SEED_NAME "seed"
#define PRECISION_NAME "precision"
#define ADAPTIVE_NAME "adaptive"
#define LAMBDA_BETA_NAME "lambda-beta"
#define TOL_NAME "tol"
#define DIRECT_NAME "direct"

using namespace smurff;

boost::program_options::options_description get_desc()
{
   boost::program_options::options_description basic_desc("Basic options");
   basic_desc.add_options()
     (HELP_NAME, "show help information");

   boost::program_options::options_description priors_desc("Priors and side Info");
   priors_desc.add_options()
     (ROW_PRIOR_NAME, boost::program_options::value<std::string>()->default_value(PRIOR_NAME_DEFAULT), "One of <normal|spikeandslab|macau|macauone>")
     (COL_PRIOR_NAME, boost::program_options::value<std::string>()->default_value(PRIOR_NAME_DEFAULT), "One of <normal|spikeandslab|macau|macauone>")
     (ROW_FEATURES_NAME, boost::program_options::value<std::string>(), "side info for rows")
     (COL_FEATURES_NAME, boost::program_options::value<std::string>(), "side info for cols");
     //(ROW_MODEL_NAME, boost::program_options::value<std::string>(), "initialization matrix for row model")
     //(COL_MODEL_NAME, boost::program_options::value<std::string>(), "initialization matrix for col model")
     //(CENTER_NAME, boost::program_options::value<std::string>(), "center <global|rows|cols|none>");

   boost::program_options::options_description train_test_desc("Test and train matrices");
   train_test_desc.add_options()
     (TEST_NAME, boost::program_options::value<std::string>(), "test data (for computing RMSE)")
     (TRAIN_NAME, boost::program_options::value<std::string>(), "train data file");

   boost::program_options::options_description general_desc("General parameters");
   general_desc.add_options()
      (BURNIN_NAME, boost::program_options::value<int>()->default_value(200), "number of samples to discard")
      (NSAMPLES_NAME, boost::program_options::value<int>()->default_value(800), "number of samples to collect")
      (NUM_LATENT_NAME, boost::program_options::value<int>()->default_value(96), "number of latent dimensions")
      (RESTORE_PREFIX_NAME, boost::program_options::value<std::string>()->default_value(std::string()), "prefix for file to initialize state")
      (RESTORE_SUFFIX_NAME, boost::program_options::value<std::string>()->default_value(".csv"), "suffix for initialization files (.csv or .ddm)")
      (INIT_MODEL_NAME, boost::program_options::value<std::string>()->default_value(MODEL_INIT_NAME_ZERO), "One of <random|zero>")
      (SAVE_PREFIX_NAME, boost::program_options::value<std::string>()->default_value("save"), "prefix for result files")
      (SAVE_SUFFIX_NAME, boost::program_options::value<std::string>()->default_value(".csv"), "suffix for result files (.csv or .ddm)")
      (SAVE_FREQ_NAME, boost::program_options::value<int>()->default_value(0), "save every n iterations (0 == never, -1 == final model)")
      (THRESHOLD_NAME, boost::program_options::value<double>(), "threshold for binary classification")
      (VERBOSE_NAME, boost::program_options::value<int>()->default_value(1), "verbose output")
      (QUIET_NAME, "no output")
      (VERSION_NAME, "print version info")
      (STATUS_NAME, boost::program_options::value<std::string>()->default_value(std::string()), "output progress to csv file")
      (SEED_NAME, boost::program_options::value<int>(), "random number generator seed");

   boost::program_options::options_description noise_desc("Noise model");
   noise_desc.add_options()
      (PRECISION_NAME, boost::program_options::value<std::string>()->default_value("5.0"), "precision of observations")
      (ADAPTIVE_NAME, boost::program_options::value<std::string>()->default_value("1.0,10.0"), "adaptive precision of observations");

   boost::program_options::options_description macau_prior_desc("For the macau prior");
   macau_prior_desc.add_options()
      (LAMBDA_BETA_NAME, boost::program_options::value<double>()->default_value(10.0), "initial value of lambda beta")
      (TOL_NAME, boost::program_options::value<double>()->default_value(1e-6), "tolerance for CG")
      (DIRECT_NAME, "Use Cholesky decomposition i.o. CG Solver");

   boost::program_options::options_description desc("SMURFF: Scalable Matrix Factorization Framework\n\thttp://github.com/ExaScience/smurff");
   desc.add(basic_desc);
   desc.add(priors_desc);
   desc.add(train_test_desc);
   desc.add(general_desc);
   desc.add(noise_desc);
   desc.add(macau_prior_desc);

   return desc;
}

void set_noise_model_win(Config& config, std::string noiseName, std::string optarg)
{
   NoiseConfig nc;
   nc.setNoiseType(smurff::stringToNoiseType(noiseName));
   if (nc.getNoiseType() == NoiseTypes::adaptive)
   {
      std::stringstream lineStream(optarg);
      std::string token;
      std::vector<std::string> tokens;

      while (std::getline(lineStream, token, ','))
         tokens.push_back(token);

      if(tokens.size() != 2)
         THROWERROR("invalid number of options for adaptive noise");

      nc.sn_init = strtod(tokens[0].c_str(), NULL);
      nc.sn_max = strtod(tokens[1].c_str(), NULL);
   }
   else if (nc.getNoiseType() == NoiseTypes::fixed)
   {
      nc.precision = strtod(optarg.c_str(), NULL);
   }

   if(!config.m_train)
      THROWERROR("train data is not provided");

   // set global noise model
   if (config.m_train->getNoiseConfig().getNoiseType() == NoiseTypes::noiseless)
      config.m_train->setNoiseConfig(nc);

   //set for row/col feautres
   for(auto m: config.m_row_features)
      if (m->getNoiseConfig().getNoiseType() == NoiseTypes::noiseless)
         m->setNoiseConfig(nc);

   for(auto m: config.m_col_features)
      if (m->getNoiseConfig().getNoiseType() == NoiseTypes::noiseless)
         m->setNoiseConfig(nc);
}

void fill_config(boost::program_options::variables_map& vm, Config& config)
{
   if (vm.count(ROW_PRIOR_NAME))
      config.row_prior_type = stringToPriorType(vm[ROW_PRIOR_NAME].as<std::string>());

   if (vm.count(COL_PRIOR_NAME))
      config.col_prior_type = stringToPriorType(vm[COL_PRIOR_NAME].as<std::string>());

   if (vm.count(ROW_FEATURES_NAME))
      config.m_row_features.push_back(matrix_io::read_matrix(vm[ROW_FEATURES_NAME].as<std::string>(), false));

   if (vm.count(COL_FEATURES_NAME))
      config.m_col_features.push_back(matrix_io::read_matrix(vm[COL_FEATURES_NAME].as<std::string>(), false));

   if (vm.count(TRAIN_NAME))
      config.m_train = generic_io::read_data_config(vm[TRAIN_NAME].as<std::string>(), true);

   if (vm.count(TEST_NAME))
      config.m_test = generic_io::read_data_config(vm[TEST_NAME].as<std::string>(), true);

   if (vm.count(BURNIN_NAME))
      config.burnin = vm[BURNIN_NAME].as<int>();

   if (vm.count(NSAMPLES_NAME))
      config.nsamples = vm[NSAMPLES_NAME].as<int>();

   if(vm.count(NUM_LATENT_NAME))
      config.num_latent = vm[NUM_LATENT_NAME].as<int>();

   if(vm.count(RESTORE_PREFIX_NAME))
      config.restore_prefix = vm[RESTORE_PREFIX_NAME].as<std::string>();

   if(vm.count(RESTORE_SUFFIX_NAME))
      config.restore_suffix = vm[RESTORE_SUFFIX_NAME].as<std::string>();

   if(vm.count(INIT_MODEL_NAME))
      config.model_init_type = stringToModelInitType(vm[INIT_MODEL_NAME].as<std::string>());

   if(vm.count(SAVE_PREFIX_NAME))
      config.setSavePrefix(vm[SAVE_PREFIX_NAME].as<std::string>());

   if(vm.count(SAVE_SUFFIX_NAME))
      config.save_suffix = vm[SAVE_SUFFIX_NAME].as<std::string>();

   if(vm.count(SAVE_FREQ_NAME))
      config.save_freq = vm[SAVE_FREQ_NAME].as<int>();

   if(vm.count(THRESHOLD_NAME))
   {
      config.threshold = vm[THRESHOLD_NAME].as<double>(); 
      config.classify = true;  
   }

   if(vm.count(VERBOSE_NAME))
      config.verbose = vm[VERBOSE_NAME].as<int>();

   if(vm.count(QUIET_NAME))
      config.verbose = 0;

   if(vm.count(STATUS_NAME))
      config.csv_status = vm[STATUS_NAME].as<std::string>();
      
   if(vm.count(SEED_NAME))
   {
      config.random_seed_set = true;
      config.random_seed = vm[SEED_NAME].as<int>();
   }

   if(vm.count(PRECISION_NAME))
      set_noise_model_win(config, NOISE_NAME_FIXED, vm[PRECISION_NAME].as<std::string>());

   if(vm.count(ADAPTIVE_NAME))
      set_noise_model_win(config, NOISE_NAME_ADAPTIVE, vm[ADAPTIVE_NAME].as<std::string>());

   if(vm.count(LAMBDA_BETA_NAME))
      config.lambda_beta = vm[LAMBDA_BETA_NAME].as<double>(); 

   if(vm.count(TOL_NAME))
      config.tol = vm[TOL_NAME].as<double>();

   if(vm.count(DIRECT_NAME))
      config.direct = true;
}

bool parse_options(int argc, char* argv[], Config& config)
{
   try
   {
      boost::program_options::options_description desc = get_desc();

      boost::program_options::variables_map vm;
      store(parse_command_line(argc, argv, desc), vm);
      notify(vm);

      if(vm.count(HELP_NAME))
      {
        std::cout << desc << std::endl;
        return false;
      }

      if(vm.count(VERSION_NAME))
      {
         std::cout <<  "SMURFF " << smurff::SMURFF_VERSION << std::endl;
         return false;
      }

      fill_config(vm, config);
      return true;
   }
   catch (const boost::program_options::error &ex)
   {
      std::cerr << "Failed to parse command line arguments: " << std::endl;
      std::cerr << ex.what() << std::endl;
      return false;
   }
   catch (std::runtime_error& ex)
   {
      std::cerr << "Failed to parse command line arguments: " << std::endl;
      std::cerr << ex.what() << std::endl;
      return false;
   }
}

void CmdSession::setFromArgs(int argc, char** argv)
{
   Config config;
   if(!parse_options(argc, argv, config))
      exit(0); //need a way to figure out how to handle help and version

   setFromConfig(config);
}
