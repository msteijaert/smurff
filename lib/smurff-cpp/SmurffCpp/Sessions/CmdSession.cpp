#include <string>
#include <iostream>
#include <sstream>

#ifdef HAVE_BOOST
#include <boost/program_options.hpp>
#endif

#include "CmdSession.h"
#include <SmurffCpp/Predict/PredictSession.h>
#include <SmurffCpp/Sessions/SessionFactory.h>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Version.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>

#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/Utils/StringUtils.h>

static const char *PREDICT_NAME = "predict";
static const char *HELP_NAME = "help";
static const char *PRIOR_NAME = "prior";
static const char *TEST_NAME = "test";
static const char *TRAIN_NAME = "train";
static const char *BURNIN_NAME = "burnin";
static const char *NSAMPLES_NAME = "nsamples";
static const char *NUM_LATENT_NAME = "num-latent";
static const char *NUM_THREADS_NAME = "num-threads";
static const char *SAVE_PREFIX_NAME = "save-prefix";
static const char *SAVE_EXTENSION_NAME = "save-extension";
static const char *SAVE_FREQ_NAME = "save-freq";
static const char *CHECKPOINT_FREQ_NAME = "checkpoint-freq";
static const char *THRESHOLD_NAME = "threshold";
static const char *VERBOSE_NAME = "verbose";
static const char *VERSION_NAME = "version";
static const char *STATUS_NAME = "status";
static const char *SEED_NAME = "seed";
static const char *INI_NAME = "ini";
static const char *ROOT_NAME = "root";

using namespace smurff;

#ifdef HAVE_BOOST

namespace po = boost::program_options;

po::options_description get_desc()
{
    po::options_description general_desc("General parameters");
    general_desc.add_options()
	(VERSION_NAME, "print version info (and exit)")
	(HELP_NAME, "show this help information (and exit)")
	(INI_NAME, po::value<std::string>(), "read options from this .ini file")
	(NUM_THREADS_NAME, po::value<int>()->default_value(Config::NUM_THREADS_DEFAULT_VALUE), "number of threads (0 = default by OpenMP)")
	(VERBOSE_NAME, po::value<int>()->default_value(Config::VERBOSE_DEFAULT_VALUE), "verbosity of output (0, 1, 2 or 3)")
	(SEED_NAME, po::value<int>()->default_value(Config::RANDOM_SEED_DEFAULT_VALUE), "random number generator seed");

    po::options_description train_desc("Used during training");
    train_desc.add_options()
	(TRAIN_NAME, po::value<std::string>(), "train data file")
	(TEST_NAME, po::value<std::string>(), "test data")
	(PRIOR_NAME, po::value<std::vector<std::string>>()->multitoken(), "provide a prior-type for each dimension of train; prior-types:  <normal|normalone|spikeandslab|macau|macauone>")
	(BURNIN_NAME, po::value<int>()->default_value(Config::BURNIN_DEFAULT_VALUE), "number of samples to discard")
	(NSAMPLES_NAME, po::value<int>()->default_value(Config::NSAMPLES_DEFAULT_VALUE), "number of samples to collect")
	(NUM_LATENT_NAME, po::value<int>()->default_value(Config::NUM_LATENT_DEFAULT_VALUE), "number of latent dimensions")
	(THRESHOLD_NAME, po::value<double>()->default_value(Config::THRESHOLD_DEFAULT_VALUE), "threshold for binary classification and AUC calculation");

    po::options_description predict_desc("Used during prediction");
    predict_desc.add_options()
	(PREDICT_NAME, po::value<std::string>(), "sparse matrix with values to predict")
	(THRESHOLD_NAME, po::value<double>()->default_value(Config::THRESHOLD_DEFAULT_VALUE), "threshold for binary classification and AUC calculation");

    po::options_description save_desc("Storing models and predictions");
    save_desc.add_options()
	(ROOT_NAME, po::value<std::string>(), "restore session from root .ini file")
	(SAVE_PREFIX_NAME, po::value<std::string>()->default_value(Config::SAVE_PREFIX_DEFAULT_VALUE), "prefix for result files")
	(STATUS_NAME, po::value<std::string>()->default_value(Config::STATUS_DEFAULT_VALUE), "output progress to csv file")
	(SAVE_EXTENSION_NAME, po::value<std::string>()->default_value(Config::SAVE_EXTENSION_DEFAULT_VALUE), "extension for result files (.csv or .ddm)")
	(SAVE_FREQ_NAME, po::value<int>()->default_value(Config::SAVE_FREQ_DEFAULT_VALUE), "save every n iterations (0 == never, -1 == final model)")
	(CHECKPOINT_FREQ_NAME, po::value<int>()->default_value(Config::CHECKPOINT_FREQ_DEFAULT_VALUE), "save state every n seconds, only one checkpointing state is kept");

    po::options_description desc("SMURFF: Scalable Matrix Factorization Framework\n\thttp://github.com/ExaScience/smurff");
    desc.add(general_desc);
    desc.add(train_desc);
    desc.add(predict_desc);
    desc.add(save_desc);

    return desc;
}


struct ConfigFiller
{
    const po::variables_map &vm;
    Config &config;

    template <typename T, void (Config::*Func)(T)>
    void set(std::string name)
    {
        if (vm.count(name) && !vm[name].defaulted())
            (config.*Func)(vm[name].as<T>());
    }

    template <void (Config::*Func)(std::shared_ptr<TensorConfig>)>
    void set_tensor(std::string name, bool set_noise)
    {
        if (vm.count(name) && !vm[name].defaulted())
        {
            auto tensor_config = generic_io::read_data_config(vm[name].as<std::string>(), true);
            tensor_config->setNoiseConfig(NoiseConfig(NoiseConfig::NOISE_TYPE_DEFAULT_VALUE));
            (this->config.*Func)(tensor_config); 
        }
    }
    
    void set_priors(std::string name)
    {
        if (vm.count(name) && !vm[name].defaulted())
            config.setPriorTypes(vm[name].as<std::vector<std::string>>());
    }
};

// variables_map -> Config
Config
fill_config(const po::variables_map &vm)
{
    Config config;
    ConfigFiller filler = {vm, config};


    //restore session from root file (command line arguments are already stored in file)
    if (vm.count(ROOT_NAME))
    {
        //restore config from root file
        std::string root_name = vm[ROOT_NAME].as<std::string>();
        config.setRootName(root_name);

        if (!vm.count(PREDICT_NAME))
            RootFile(root_name).restoreConfig(config);
    }

    //restore ini file if it was specified
    if (vm.count(INI_NAME))
    {
        auto ini_file = vm[INI_NAME].as<std::string>();
        bool success = config.restore(ini_file);
        THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + ini_file + "'");
        config.setIniName(ini_file);
    }

    filler.set_tensor<&Config::setPredict>(PREDICT_NAME, false);
    filler.set_tensor<&Config::setTest>(TEST_NAME, false);
    filler.set_tensor<&Config::setTrain>(TRAIN_NAME, true);

    filler.set_priors(PRIOR_NAME);

    filler.set<double,      &Config::setThreshold>(THRESHOLD_NAME);
    filler.set<int,         &Config::setBurnin>(BURNIN_NAME);
    filler.set<int,         &Config::setNSamples>(NSAMPLES_NAME);
    filler.set<int,         &Config::setNumLatent>(NUM_LATENT_NAME);
    filler.set<int,         &Config::setNumThreads>(NUM_THREADS_NAME);
    filler.set<std::string, &Config::setSavePrefix>(SAVE_PREFIX_NAME);
    filler.set<std::string, &Config::setSaveExtension>(SAVE_EXTENSION_NAME);
    filler.set<int,         &Config::setSaveFreq>(SAVE_FREQ_NAME);
    filler.set<int,         &Config::setCheckpointFreq>(CHECKPOINT_FREQ_NAME);
    filler.set<double,      &Config::setThreshold>(THRESHOLD_NAME);
    filler.set<int,         &Config::setVerbose>(VERBOSE_NAME);
    filler.set<int,         &Config::setRandomSeed>(SEED_NAME);

    return config;
}

// argc/argv -> variables_map -> Config
Config smurff::parse_options(int argc, char *argv[])
{
    po::variables_map vm;

    try
    {
        po::options_description desc = get_desc();

        if (argc < 2)
        {
            std::cout << desc << std::endl;
            return Config();
        }

        po::command_line_parser parser{argc, argv};
        parser.options(desc);

        po::parsed_options parsed_options = parser.run();

        store(parsed_options, vm);
        notify(vm);

        if (vm.count(HELP_NAME))
        {
            std::cout << desc << std::endl;
            return Config();
        }

        if (vm.count(VERSION_NAME))
        {
            std::cout << "SMURFF " << smurff::SMURFF_VERSION << std::endl;
            return Config();
        }
    }
    catch (const po::error &ex)
    {
        std::cerr << "Failed to parse command line arguments: " << std::endl;
        std::cerr << ex.what() << std::endl;
        throw(ex);
    }
    catch (std::runtime_error &ex)
    {
        std::cerr << "Failed to parse command line arguments: " << std::endl;
        std::cerr << ex.what() << std::endl;
        throw(ex);
    }

    const std::vector<std::string> train_only_options = {
        TRAIN_NAME, TEST_NAME, PRIOR_NAME, BURNIN_NAME, NSAMPLES_NAME, NUM_LATENT_NAME};

    //-- prediction only
    if (vm.count(PREDICT_NAME))
    {
        if (!vm.count(ROOT_NAME))
            THROWERROR("Need --root option in predict mode");

        for (auto name : train_only_options)
        {
            if (vm.count(name) && !vm[name].defaulted())
                THROWERROR("You're not allowed to mix train options (--" + name + ") with --predict");
        }
    }

    return fill_config(vm);
}

#else // no BOOST

// argc/argv --> Config
Config smurff::parse_options(int argc, char *argv[])
{
    auto usage = []() {
        std::cerr << "Usage:\n\tsmurff --[ini|root] <ini_file.ini>\n\n"
                  << "(Limited smurff compiled w/o boost program options)" << std::endl;
        exit(0);
    };

    if (argc != 3) usage();

    Config config;

    //restore session from root file (command line arguments are already stored in file)
    if (std::string(argv[1]) == "--" + std::string(ROOT_NAME))
    {
        std::string root_name(argv[2]);
        RootFile(root_name).restoreConfig(config);
        config.setRootName(root_name);
     }

    //create new session from config (passing command line arguments)
    else if (std::string(argv[1]) == "--" + std::string(INI_NAME))
    {
        std::string ini_file(argv[2]);
        bool success = config.restore(ini_file);
        THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + ini_file + "'");
        config.setIniName(ini_file);
    } 
    else
    {
        usage();
    }

    return config;
}
#endif

//create cmd session
//parses args with setFromArgs, then internally calls setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> smurff::create_cmd_session(int argc, char **argv)
{
    std::shared_ptr<ISession> session;

    auto config = parse_options(argc, argv);
    if (config.isActionTrain())
        session = SessionFactory::create_session(config);
    else if (config.isActionPredict())
        session = std::make_shared<PredictSession>(config);
    else
        exit(0);

    return session;
}
