#include <string>
#include <iostream>
#include <sstream>

#ifdef HAVE_BOOST
#include <boost/program_options.hpp>
#endif

#include "CmdSession.h"

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Version.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>

#include <SmurffCpp/Utils/RootFile.h>

#define HELP_NAME "help"
#define PRIOR_NAME "prior"
#define SIDE_INFO_NAME "side-info"
#define AUX_DATA_NAME "aux-data"
#define TEST_NAME "test"
#define TRAIN_NAME "train"
#define BURNIN_NAME "burnin"
#define NSAMPLES_NAME "nsamples"
#define NUM_LATENT_NAME "num-latent"
#define INIT_MODEL_NAME "init-model"
#define SAVE_PREFIX_NAME "save-prefix"
#define SAVE_EXTENSION_NAME "save-extension"
#define SAVE_FREQ_NAME "save-freq"
#define THRESHOLD_NAME "threshold"
#define VERBOSE_NAME "verbose"
#define QUIET_NAME "quiet"
#define VERSION_NAME "version"
#define STATUS_NAME "status"
#define SEED_NAME "seed"
#define PRECISION_NAME "precision"
#define ADAPTIVE_NAME "adaptive"
#define PROBIT_NAME "probit"
#define LAMBDA_BETA_NAME "lambda-beta"
#define TOL_NAME "tol"
#define DIRECT_NAME "direct"
#define INI_NAME "ini"
#define ROOT_NAME "root"

#define NONE_TOKEN "none"

using namespace smurff;

#ifdef HAVE_BOOST
boost::program_options::options_description get_desc()
{
   boost::program_options::options_description basic_desc("Basic options");
   basic_desc.add_options()
     (VERSION_NAME, "print version info")
     (HELP_NAME, "show this help information");

   boost::program_options::options_description priors_desc("Priors and side Info");
   priors_desc.add_options()
     (PRIOR_NAME, boost::program_options::value<std::vector<std::string> >()->multitoken(), 
        "provide a prior-type for each dimension of train; prior-types:  <normal|normalone|spikeandslab|macau|macauone>")
     (SIDE_INFO_NAME, boost::program_options::value<std::vector<std::string> >()->multitoken(), "Side info for each dimention")
     (AUX_DATA_NAME, boost::program_options::value<std::vector<std::string> >()->multitoken(),"Aux data for each dimention");

   boost::program_options::options_description train_test_desc("Test and train matrices");
   train_test_desc.add_options()
     (TEST_NAME, boost::program_options::value<std::string>(), "test data (for computing RMSE)")
     (TRAIN_NAME, boost::program_options::value<std::string>(), "train data file");

   boost::program_options::options_description general_desc("General parameters");
   general_desc.add_options()
      (INI_NAME, boost::program_options::value<std::string>(), "read options from this .ini file")
      (ROOT_NAME, boost::program_options::value<std::string>(), "restore session from root .ini file")
      (BURNIN_NAME, boost::program_options::value<int>()->default_value(200), "number of samples to discard")
      (NSAMPLES_NAME, boost::program_options::value<int>()->default_value(800), "number of samples to collect")
      (NUM_LATENT_NAME, boost::program_options::value<int>()->default_value(96), "number of latent dimensions")
      (INIT_MODEL_NAME, boost::program_options::value<std::string>()->default_value(MODEL_INIT_NAME_ZERO), "Initialize model using <random|zero> values")
      (SAVE_PREFIX_NAME, boost::program_options::value<std::string>()->default_value("save"), "prefix for result files")
      (SAVE_EXTENSION_NAME, boost::program_options::value<std::string>()->default_value(".csv"), "extension for result files (.csv or .ddm)")
      (SAVE_FREQ_NAME, boost::program_options::value<int>()->default_value(0), "save every n iterations (0 == never, -1 == final model)")
      (THRESHOLD_NAME, boost::program_options::value<double>(), "threshold for binary classification and AUC calculation")
      (VERBOSE_NAME, boost::program_options::value<int>()->default_value(1), "verbosity of output (0, 1 or 2)")
      (QUIET_NAME, "no output (equivalent to verbose=0)")
      (STATUS_NAME, boost::program_options::value<std::string>()->default_value(std::string()), "output progress to csv file")
      (SEED_NAME, boost::program_options::value<int>(), "random number generator seed");

   boost::program_options::options_description noise_desc("Noise model.");
   noise_desc.add_options()
      (PRECISION_NAME, boost::program_options::value<std::string>()->default_value("5.0"), "set fixed precision of observations")
      (ADAPTIVE_NAME, boost::program_options::value<std::string>()->default_value("1.0,10.0"),
        "use adaptive precision of observations, sets initial (default: 1.0) and maximum (default:10.0) SNR")
      (PROBIT_NAME, boost::program_options::value<std::string>()->default_value("0.0"), "Use probit noise model with given threshold");
 
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
#endif

void set_noise_model(Config& config, std::string noiseName, std::string optarg)
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
   else if (nc.getNoiseType() == NoiseTypes::probit)
   {
      nc.threshold = strtod(optarg.c_str(), NULL);
   }

   if(!config.getTrain())
      THROWERROR("train data is not provided");

   // set global noise model
   if (config.getTrain()->getNoiseConfig().getNoiseType() == NoiseTypes::unset)
      config.getTrain()->setNoiseConfig(nc);

   //set for side info
   for(auto& sideInfo : config.getSideInfo())
   {
      if (sideInfo && sideInfo->getNoiseConfig().getNoiseType() == NoiseTypes::unset)
         sideInfo->setNoiseConfig(nc);
   }

   // set for aux data
   for(auto& auxDataSet : config.getAuxData())
   {
      for(auto auxData : auxDataSet)
      {
         if (auxData->getNoiseConfig().getNoiseType() == NoiseTypes::unset)
            auxData->setNoiseConfig(nc);
      }
   }
}

#ifdef HAVE_BOOST
void fill_config(boost::program_options::variables_map& vm, Config& config)
{
   //create new session with ini file
   if (vm.count(INI_NAME) && !vm[INI_NAME].defaulted()) {
      auto ini_file = vm[INI_NAME].as<std::string>();
      bool success = config.restore(ini_file);
      THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + ini_file + "'");
   }

   //create new session from command line options or override options from ini file
   if (vm.count(PRIOR_NAME) && !vm[PRIOR_NAME].defaulted())
      for (auto& pr : vm[PRIOR_NAME].as<std::vector<std::string> >())
         config.getPriorTypes().push_back(stringToPriorType(pr));

   if (vm.count(SIDE_INFO_NAME) && !vm[SIDE_INFO_NAME].defaulted())
   {
      for (auto sideInfo : vm[SIDE_INFO_NAME].as<std::vector<std::string> >())
      {
         if (sideInfo == NONE_TOKEN)
            config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
         else
            config.getSideInfo().push_back(matrix_io::read_matrix(sideInfo, false));
      }
   }

   if (vm.count(AUX_DATA_NAME) && !vm[AUX_DATA_NAME].defaulted())
   {
      for (auto auxDataString : vm[AUX_DATA_NAME].as<std::vector<std::string> >())
      {
         config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
         auto& dimAuxData = config.getAuxData().back();

         std::stringstream lineStream(auxDataString);
         std::string token;

         while (std::getline(lineStream, token, ','))
         {
            //add ability to skip features for specific dimention
            if(token == NONE_TOKEN)
               continue;

            dimAuxData.push_back(matrix_io::read_matrix(token, false));
         }
      }
   }

   if (vm.count(TEST_NAME) && !vm[TEST_NAME].defaulted())
      config.setTest(generic_io::read_data_config(vm[TEST_NAME].as<std::string>(), true));

   if (vm.count(TRAIN_NAME) && !vm[TRAIN_NAME].defaulted())
      config.setTrain(generic_io::read_data_config(vm[TRAIN_NAME].as<std::string>(), true));

   if (vm.count(BURNIN_NAME) && !vm[BURNIN_NAME].defaulted())
      config.setBurnin(vm[BURNIN_NAME].as<int>());

   if (vm.count(NSAMPLES_NAME) && !vm[NSAMPLES_NAME].defaulted())
      config.setNSamples(vm[NSAMPLES_NAME].as<int>());

   if (vm.count(NUM_LATENT_NAME) && !vm[NUM_LATENT_NAME].defaulted())
      config.setNumLatent(vm[NUM_LATENT_NAME].as<int>());

   if(vm.count(INIT_MODEL_NAME))
      config.setModelInitType(stringToModelInitType(vm[INIT_MODEL_NAME].as<std::string>()));

   if (vm.count(SAVE_PREFIX_NAME) && !vm[SAVE_PREFIX_NAME].defaulted())
      config.setSavePrefix(vm[SAVE_PREFIX_NAME].as<std::string>());

   if(vm.count(SAVE_EXTENSION_NAME) && !vm[SAVE_EXTENSION_NAME].defaulted())
      config.setSaveExtension(vm[SAVE_EXTENSION_NAME].as<std::string>());

   if (vm.count(SAVE_FREQ_NAME) && !vm[SAVE_FREQ_NAME].defaulted())
      config.setSaveFreq(vm[SAVE_FREQ_NAME].as<int>());

   if (vm.count(THRESHOLD_NAME) && !vm[THRESHOLD_NAME].defaulted())
   {
      config.setThreshold(vm[THRESHOLD_NAME].as<double>());
      config.setClassify(true);
   }

   if (vm.count(VERBOSE_NAME) && !vm[VERBOSE_NAME].defaulted())
      config.setVerbose(vm[VERBOSE_NAME].as<int>());

   if (vm.count(QUIET_NAME) && !vm[QUIET_NAME].defaulted())
      config.setVerbose(0);

   if (vm.count(STATUS_NAME) && !vm[STATUS_NAME].defaulted())
      config.setCsvStatus(vm[STATUS_NAME].as<std::string>());

   if (vm.count(SEED_NAME) && !vm[SEED_NAME].defaulted())
   {
      config.setRandomSeedSet(true);
      config.setRandomSeed(vm[SEED_NAME].as<int>());
   }

   if (vm.count(PRECISION_NAME) && !vm[PRECISION_NAME].defaulted())
      set_noise_model(config, NOISE_NAME_FIXED, vm[PRECISION_NAME].as<std::string>());

   if (vm.count(ADAPTIVE_NAME) && !vm[ADAPTIVE_NAME].defaulted())
      set_noise_model(config, NOISE_NAME_ADAPTIVE, vm[ADAPTIVE_NAME].as<std::string>());

   if (vm.count(PROBIT_NAME) && !vm[PROBIT_NAME].defaulted())
      set_noise_model(config, NOISE_NAME_PROBIT, vm[PROBIT_NAME].as<std::string>());

   if (vm.count(LAMBDA_BETA_NAME) && !vm[LAMBDA_BETA_NAME].defaulted())
      config.setLambdaBeta(vm[LAMBDA_BETA_NAME].as<double>());

   if (vm.count(TOL_NAME) && !vm[TOL_NAME].defaulted())
      config.setTol(vm[TOL_NAME].as<double>());

   if (vm.count(DIRECT_NAME) && !vm[DIRECT_NAME].defaulted())
      config.setDirect(true);
}
#endif

bool CmdSession::parse_options(int argc, char* argv[])
{
   #ifdef HAVE_BOOST
   try
   {
      boost::program_options::options_description desc = get_desc();
      boost::program_options::command_line_parser parser{ argc, argv };
      parser.options(desc);

      boost::program_options::parsed_options parsed_options = parser.run();

      boost::program_options::variables_map vm;
      store(parsed_options, vm);
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

      //restore session from root file (command line arguments are already stored in file)
      if (vm.count(ROOT_NAME))
      {
         std::string root_file = vm[ROOT_NAME].as<std::string>();
         setFromRootPath(root_file);
         return true;
      }
      //create new session from config (passing command line arguments)
      else
      {
         Config config;
         fill_config(vm, config);
         setFromConfig(config);
         return true;
      }
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
   #else

   if (argc != 3) 
   {
      std::cerr << "Usage:\n\tsmurff --ini <ini_file.ini>\n\n(Limited smurff compiled w/o boost program options)" << std::endl;
      return false;
   }

   try
   {
      //restore session from root file (command line arguments are already stored in file)
      if (std::string(argv[1]) == "--" + std::string(ROOT_NAME))
      {
         std::string root_file(argv[2]);
         setFromRootPath(root_file);
         return true;
      }
      //create new session from config (passing command line arguments)
      else if (std::string(argv[1]) == "--" + std::string(INI_NAME))
      {
         Config config;

         std::string ini_file(argv[2]);
         bool success = config.restore(ini_file);
         if (!success)
         {
            std::cout << "Could not load ini file '" << ini_file << "'" << std::endl;
            return false;
         }

         setFromConfig(config);
         return true;
      }
      else
      {
         std::cerr << "Usage:\n\tsmurff --ini <ini_file.ini>\n\n(Limited smurff compiled w/o boost program options)" << std::endl;
         return false;
      }
   }
   catch (std::runtime_error& ex)
   {
      std::cerr << "Failed to parse command line arguments: " << std::endl;
      std::cerr << ex.what() << std::endl;
      return false;
   }

   #endif
}

void CmdSession::setFromArgs(int argc, char** argv)
{
   std::shared_ptr<RootFile> rootFile;

   if(!parse_options(argc, argv))
      exit(0); //need a way to figure out how to handle help and version
}

//create cmd session
//parses args with setFromArgs, then internally calls setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> smurff::create_cmd_session(int argc, char** argv)
{
   std::shared_ptr<CmdSession> session(new CmdSession());
   session->setFromArgs(argc, argv);
   return session;
}
