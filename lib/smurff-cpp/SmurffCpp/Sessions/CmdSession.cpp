#include <string>
#include <iostream>
#include <sstream>

#ifdef HAVE_BOOST
#include <boost/program_options.hpp>
#endif

#include "CmdSession.h"
#include <SmurffCpp/Predict/PredictSession.h>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Version.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>

#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/Utils/StringUtils.h>

#define PREDICT_NAME "predict"
#define HELP_NAME "help"
#define PRIOR_NAME "prior"
#define SIDE_INFO_NAME "side-info"
#define AUX_DATA_NAME "aux-data"
#define TEST_NAME "test"
#define TRAIN_NAME "train"
#define BURNIN_NAME "burnin"
#define NSAMPLES_NAME "nsamples"
#define NUM_LATENT_NAME "num-latent"
#define NUM_THREADS_NAME "num-threads"
#define INIT_MODEL_NAME "init-model"
#define SAVE_PREFIX_NAME "save-prefix"
#define SAVE_EXTENSION_NAME "save-extension"
#define SAVE_FREQ_NAME "save-freq"
#define CHECKPOINT_FREQ_NAME "checkpoint-freq"
#define THRESHOLD_NAME "threshold"
#define VERBOSE_NAME "verbose"
#define QUIET_NAME "quiet"
#define VERSION_NAME "version"
#define STATUS_NAME "status"
#define SEED_NAME "seed"
#define NOISE_MODEL_NAME "noise_model"
#define INI_NAME "ini"
#define ROOT_NAME "root"

#define NONE_TOKEN "none"

using namespace smurff;

#ifdef HAVE_BOOST
boost::program_options::options_description get_desc()
{
   boost::program_options::options_description basic_desc("Basic options");
   basic_desc.add_options()
     (PREDICT_NAME, "only prediction, no training")
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
      (BURNIN_NAME, boost::program_options::value<int>()->default_value(Config::BURNIN_DEFAULT_VALUE), "number of samples to discard")
      (NSAMPLES_NAME, boost::program_options::value<int>()->default_value(Config::NSAMPLES_DEFAULT_VALUE), "number of samples to collect")
      (NUM_LATENT_NAME, boost::program_options::value<int>()->default_value(Config::NUM_LATENT_DEFAULT_VALUE), "number of latent dimensions")
      (NUM_THREADS_NAME, boost::program_options::value<int>()->default_value(Config::NUM_THREADS_DEFAULT_VALUE), "number of threads (0 = default by OpenMP)")
      (INIT_MODEL_NAME, boost::program_options::value<std::string>()->default_value(modelInitTypeToString(Config::INIT_MODEL_DEFAULT_VALUE)), "Initialize model using <random|zero> values")
      (SAVE_PREFIX_NAME, boost::program_options::value<std::string>()->default_value(Config::SAVE_PREFIX_DEFAULT_VALUE), "prefix for result files")
      (SAVE_EXTENSION_NAME, boost::program_options::value<std::string>()->default_value(Config::SAVE_EXTENSION_DEFAULT_VALUE), "extension for result files (.csv or .ddm)")
      (SAVE_FREQ_NAME, boost::program_options::value<int>()->default_value(Config::SAVE_FREQ_DEFAULT_VALUE), "save every n iterations (0 == never, -1 == final model)")
      (CHECKPOINT_FREQ_NAME, boost::program_options::value<int>()->default_value(Config::CHECKPOINT_FREQ_DEFAULT_VALUE), "save state every n seconds, only one checkpointing state is kept")
      (THRESHOLD_NAME, boost::program_options::value<double>()->default_value(Config::THRESHOLD_DEFAULT_VALUE), "threshold for binary classification and AUC calculation")
      (VERBOSE_NAME, boost::program_options::value<int>()->default_value(Config::VERBOSE_DEFAULT_VALUE), "verbosity of output (0, 1, 2 or 3)")
      (QUIET_NAME, "no output (equivalent to verbose=0)")
      (STATUS_NAME, boost::program_options::value<std::string>()->default_value(Config::STATUS_DEFAULT_VALUE), "output progress to csv file")
      (SEED_NAME, boost::program_options::value<int>()->default_value(Config::RANDOM_SEED_DEFAULT_VALUE), "random number generator seed");

   boost::program_options::options_description noise_desc("Noise model");
   noise_desc.add_options()
      (NOISE_MODEL_NAME, boost::program_options::value<std::string>()->default_value(NoiseConfig::get_default_string()), "set properties of noise model");

   boost::program_options::options_description desc("SMURFF: Scalable Matrix Factorization Framework\n\thttp://github.com/ExaScience/smurff");
   desc.add(basic_desc);
   desc.add(priors_desc);
   desc.add(train_test_desc);
   desc.add(general_desc);
   desc.add(noise_desc);

   return desc;
}
#endif

void set_noise_configs(Config& config, const NoiseConfig nc)
{
   if(!config.getTrain())
      THROWERROR("train data is not provided");

   // set for train data
   if (config.getTrain()->getNoiseConfig().getNoiseType() == NoiseTypes::unset)
      config.getTrain()->setNoiseConfig(nc);

   //set for side info
   for (auto p :  config.getSideInfoConfigs())
   {
      for (auto configItem : p.second)
      {
          const auto& sideInfo = configItem->getSideInfo();
          if (sideInfo && sideInfo->getNoiseConfig().getNoiseType() == NoiseTypes::unset)
              sideInfo->setNoiseConfig(nc);
      }
   }

   // set for aux data
   for(auto& data : config.getAuxData())
   {
      if (data->getNoiseConfig().getNoiseType() == NoiseTypes::unset)
         data->setNoiseConfig(nc);
   }
}

NoiseConfig parse_noise_arg(std::string optarg)
{
   std::vector<std::string> tokens;
   smurff::split(optarg, tokens, ';');

   THROWERROR_ASSERT_MSG(tokens.size() == 5, "invalid number of options for noise model");

   NoiseConfig nc;

   nc.setNoiseType(smurff::stringToNoiseType(tokens.at(0)));
   nc.setPrecision(stod(tokens.at(1)));
   nc.setSnInit(stod(tokens.at(2)));
   nc.setSnMax(stod(tokens.at(3)));
   nc.setThreshold(stod(tokens.at(4)));

   return nc;
}

#ifdef HAVE_BOOST
void fill_config(boost::program_options::variables_map& vm, Config& config)
{
   if (vm.count(TEST_NAME) && !vm[TEST_NAME].defaulted())
      config.setTest(generic_io::read_data_config(vm[TEST_NAME].as<std::string>(), true));

   if (vm.count(TRAIN_NAME) && !vm[TRAIN_NAME].defaulted())
      config.setTrain(generic_io::read_data_config(vm[TRAIN_NAME].as<std::string>(), true));

   //create new session from command line options or override options from ini file
   if (vm.count(PRIOR_NAME) && !vm[PRIOR_NAME].defaulted())
      config.setPriorTypes(vm[PRIOR_NAME].as<std::vector<std::string> >());

   if (vm.count(SIDE_INFO_NAME) && !vm[SIDE_INFO_NAME].defaulted())
   {
      //parse each group of tokens into SideInfoConfig
      int mode = 0;
      for (auto sideInfoString : vm[SIDE_INFO_NAME].as<std::vector<std::string> >())
      {
         std::vector<std::string> tokens;
         smurff::split(sideInfoString, tokens, ',');

         //parse each token into SideInfoConfig
         for (auto token : tokens)
         {
            if (token == NONE_TOKEN) break;

            std::vector<std::string> properties;
            smurff::split(token, properties, ';');

            THROWERROR_ASSERT_MSG(properties.size() == 1 || properties.size() == 3,
              "Wrong number of options specified for side info token");

            auto mpci = std::make_shared<SideInfoConfig>();

            if (properties.size() == 3)
            {
              mpci->setTol(stod(properties.at(0)));
              mpci->setDirect(stoi(properties.at(1)));
              mpci->setSideInfo(matrix_io::read_matrix(properties.at(2), false));
            }
            else
            {
              mpci->setSideInfo(matrix_io::read_matrix(properties.at(0), false));
            }

            config.addSideInfoConfig(mode, mpci);
         }

         mode++;
      }
   }

   if (vm.count(AUX_DATA_NAME) && !vm[AUX_DATA_NAME].defaulted())
   {
      int dim = 0;
      int num_dim = config.getPriorTypes().size();

      THROWERROR_ASSERT_MSG(num_dim <= 2, "Only matrix and 2D tensor support aux data");

      for (auto auxDataString : vm[AUX_DATA_NAME].as<std::vector<std::string> >())
      {
         PVec<> pos(num_dim);

         std::vector<std::string> tokens;
         smurff::split(auxDataString, tokens, ',');

         for (auto token : tokens)
         {
            //add ability to skip features for specific dimention
            if (token == NONE_TOKEN)
               continue;

            //not an elegant solution but it works
            switch (dim)
            {
            case 0: //row aux data
               pos[1]++;
               break;
            case 1: //col aux data
               pos[0]++;
               break;
            default: //other dimensions data
               pos[dim]++;
               break;
            }

            auto cfg = matrix_io::read_matrix(token, false);
            cfg->setPos(pos);
            config.addAuxData(cfg);
         }

         dim++;
      }
   }

   if (vm.count(BURNIN_NAME) && !vm[BURNIN_NAME].defaulted())
      config.setBurnin(vm[BURNIN_NAME].as<int>());

   if (vm.count(NSAMPLES_NAME) && !vm[NSAMPLES_NAME].defaulted())
      config.setNSamples(vm[NSAMPLES_NAME].as<int>());

   if (vm.count(NUM_LATENT_NAME) && !vm[NUM_LATENT_NAME].defaulted())
      config.setNumLatent(vm[NUM_LATENT_NAME].as<int>());

   if (vm.count(NUM_THREADS_NAME) && !vm[NUM_THREADS_NAME].defaulted())
     config.setNumThreads(vm[NUM_THREADS_NAME].as<int>());

   if(vm.count(INIT_MODEL_NAME))
      config.setModelInitType(stringToModelInitType(vm[INIT_MODEL_NAME].as<std::string>()));

   if (vm.count(SAVE_PREFIX_NAME) && !vm[SAVE_PREFIX_NAME].defaulted())
      config.setSavePrefix(vm[SAVE_PREFIX_NAME].as<std::string>());

   if(vm.count(SAVE_EXTENSION_NAME) && !vm[SAVE_EXTENSION_NAME].defaulted())
      config.setSaveExtension(vm[SAVE_EXTENSION_NAME].as<std::string>());

   if (vm.count(SAVE_FREQ_NAME) && !vm[SAVE_FREQ_NAME].defaulted())
      config.setSaveFreq(vm[SAVE_FREQ_NAME].as<int>());

   if (vm.count(CHECKPOINT_FREQ_NAME) && !vm[CHECKPOINT_FREQ_NAME].defaulted())
      config.setCheckpointFreq(vm[CHECKPOINT_FREQ_NAME].as<int>());

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
      config.setRandomSeed(vm[SEED_NAME].as<int>());
   }

   if (vm.count(NOISE_MODEL_NAME) && !vm[NOISE_MODEL_NAME].defaulted())
      set_noise_configs(config, parse_noise_arg(vm[NOISE_MODEL_NAME].as<std::string>()));
}

bool parse_options(boost::program_options::variables_map& vm, int argc, char* argv[])
{
   try
   {
      boost::program_options::options_description desc = get_desc();

      if (argc < 2)
      {
         std::cout << desc << std::endl;
         return false;
      }

      boost::program_options::command_line_parser parser{ argc, argv };
      parser.options(desc);

      boost::program_options::parsed_options parsed_options = parser.run();

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
#endif


bool CmdSession::parse_options(int argc, char* argv[])
{
#ifdef HAVE_BOOST
   boost::program_options::variables_map vm;

   if (!::parse_options(vm, argc, argv))
   {
      return false;
   }

   //restore session from root file (command line arguments are already stored in file)
   if (vm.count(ROOT_NAME))
   {
      Config config;

      //restore config from root file
      std::string root_name = vm[ROOT_NAME].as<std::string>();
      auto root_file = std::make_shared<RootFile>(root_name);
      root_file->restoreConfig(config);

      //override root file config with options
      fill_config(vm, config);

      //create session from config, open root file
      setRestoreFromConfig(config, root_name);
      return true;
   }
   //create new session from config (passing command line arguments)
   else
   {
      Config config;

      //restore ini file if it was specified
      if (vm.count(INI_NAME) && !vm[INI_NAME].defaulted())
      {
         auto ini_file = vm[INI_NAME].as<std::string>();
         bool success = config.restore(ini_file);
         THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + ini_file + "'");
      }

      //apply options or override ini file if it was specified
      fill_config(vm, config);

      //create session from config, create root file
      setCreateFromConfig(config);
      return true;
   }

#else

   if (argc != 3)
   {
      std::cerr << "Usage:\n\tsmurff --[ini|root] <ini_file.ini>\n\n"
                << "(Limited smurff compiled w/o boost program options)" << std::endl;
      return false;
   }

   try
   {
      //restore session from root file (command line arguments are already stored in file)
      if (std::string(argv[1]) == "--" + std::string(ROOT_NAME))
      {
         std::string root_file(argv[2]);
         setRestoreFromRootPath(root_file);
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

         setCreateFromConfig(config);
         return true;
      }
      else
      {
         std::cerr << "Usage:\n\tsmurff --[ini|root] <ini_file.ini>\n\n"
                << "(Limited smurff compiled w/o boost program options)" << std::endl;
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

#ifdef HAVE_BOOST
   boost::program_options::variables_map vm;
   ::parse_options(vm, argc, argv);
   //-- prediction only
   if (vm.count(PREDICT_NAME))
   {
      if (!vm.count(ROOT_NAME))
      {
         std::cerr << "Need --root option in predict mode" << std::endl;
         exit(0);
      }

      if (!vm.count(TEST_NAME))
      {
         std::cerr << "Need --test option in predict mode" << std::endl;
         exit(0);
      }

      std::string root_name = vm[ROOT_NAME].as<std::string>();
      auto root_file = std::make_shared<RootFile>(root_name);
      Config config;
      fill_config(vm, config);
      auto session = std::make_shared<PredictSession>(root_file, config);
      return session;
   }
#endif

   //-- training
   std::shared_ptr<CmdSession> session(new CmdSession());
   session->setFromArgs(argc, argv);
   return session;
}
