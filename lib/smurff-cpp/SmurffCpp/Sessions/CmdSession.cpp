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
    general_desc.add_options()(VERSION_NAME, "print version info (and exit)")(HELP_NAME, "show this help information (and exit)")(INI_NAME, po::value<std::string>(), "read options from this .ini file")(NUM_THREADS_NAME, po::value<int>()->default_value(Config::NUM_THREADS_DEFAULT_VALUE), "number of threads (0 = default by OpenMP)")(VERBOSE_NAME, po::value<int>()->default_value(Config::VERBOSE_DEFAULT_VALUE), "verbosity of output (0, 1, 2 or 3)")(SEED_NAME, po::value<int>()->default_value(Config::RANDOM_SEED_DEFAULT_VALUE), "random number generator seed");

    po::options_description train_desc("Used during training");
    train_desc.add_options()(TRAIN_NAME, po::value<std::string>(), "train data file")(TEST_NAME, po::value<std::string>(), "test data")(PRIOR_NAME, po::value<std::vector<std::string>>()->multitoken(), "provide a prior-type for each dimension of train; prior-types:  <normal|normalone|spikeandslab|macau|macauone>")(BURNIN_NAME, po::value<int>()->default_value(Config::BURNIN_DEFAULT_VALUE), "number of samples to discard")(NSAMPLES_NAME, po::value<int>()->default_value(Config::NSAMPLES_DEFAULT_VALUE), "number of samples to collect")(NUM_LATENT_NAME, po::value<int>()->default_value(Config::NUM_LATENT_DEFAULT_VALUE), "number of latent dimensions")(THRESHOLD_NAME, po::value<double>()->default_value(Config::THRESHOLD_DEFAULT_VALUE), "threshold for binary classification and AUC calculation");

    po::options_description predict_desc("Used during prediction");
    predict_desc.add_options()(PREDICT_NAME, po::value<std::string>(), "sparse prediction matrix")(THRESHOLD_NAME, po::value<double>()->default_value(Config::THRESHOLD_DEFAULT_VALUE), "threshold for binary classification and AUC calculation");

    po::options_description save_desc("Storing models and predictions:");
    save_desc.add_options()(ROOT_NAME, po::value<std::string>(), "restore session from root .ini file")(SAVE_PREFIX_NAME, po::value<std::string>()->default_value(Config::SAVE_PREFIX_DEFAULT_VALUE), "prefix for result files")(STATUS_NAME, po::value<std::string>()->default_value(Config::STATUS_DEFAULT_VALUE), "output progress to csv file")(SAVE_EXTENSION_NAME, po::value<std::string>()->default_value(Config::SAVE_EXTENSION_DEFAULT_VALUE), "extension for result files (.csv or .ddm)")(SAVE_FREQ_NAME, po::value<int>()->default_value(Config::SAVE_FREQ_DEFAULT_VALUE), "save every n iterations (0 == never, -1 == final model)")(CHECKPOINT_FREQ_NAME, po::value<int>()->default_value(Config::CHECKPOINT_FREQ_DEFAULT_VALUE), "save state every n seconds, only one checkpointing state is kept");

    po::options_description desc("SMURFF: Scalable Matrix Factorization Framework\n\thttp://github.com/ExaScience/smurff");
    desc.add(general_desc);
    desc.add(train_desc);
    desc.add(predict_desc);
    desc.add(save_desc);

    return desc;
}

// variables_map -> Config
void fill_config(po::variables_map &vm, Config &config)
{
    auto count_and_not_defaulted = [&vm](std::string name) {
        return (vm.count(name) && !vm[name].defaulted());
    };

    //restore session from root file (command line arguments are already stored in file)
    if (vm.count(ROOT_NAME))
    {
        //restore config from root file
        std::string root_name = vm[ROOT_NAME].as<std::string>();
        RootFile(root_name).restoreConfig(config);
        config.setRootName(root_name);
    }

    //restore ini file if it was specified
    if (vm.count(INI_NAME))
    {
        auto ini_file = vm[INI_NAME].as<std::string>();
        bool success = config.restore(ini_file);
        THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + ini_file + "'");
        config.setIniName(ini_file);
    }

    if (count_and_not_defaulted(PREDICT_NAME))
    {
        config.setTest(generic_io::read_data_config(vm[PREDICT_NAME].as<std::string>(), true));
        config.setActionPredict();
    }

    if (count_and_not_defaulted(TEST_NAME))
        config.setTest(generic_io::read_data_config(vm[TEST_NAME].as<std::string>(), true));

    if (count_and_not_defaulted(TRAIN_NAME))
    {
        config.setTrain(generic_io::read_data_config(vm[TRAIN_NAME].as<std::string>(), true));
        config.setActionTrain();
    }

    if (count_and_not_defaulted(PRIOR_NAME))
        config.setPriorTypes(vm[PRIOR_NAME].as<std::vector<std::string>>());

    if (count_and_not_defaulted(BURNIN_NAME))
        config.setBurnin(vm[BURNIN_NAME].as<int>());

    if (count_and_not_defaulted(NSAMPLES_NAME))
        config.setNSamples(vm[NSAMPLES_NAME].as<int>());

    if (count_and_not_defaulted(NUM_LATENT_NAME))
        config.setNumLatent(vm[NUM_LATENT_NAME].as<int>());

    if (count_and_not_defaulted(NUM_THREADS_NAME))
        config.setNumThreads(vm[NUM_THREADS_NAME].as<int>());

    if (count_and_not_defaulted(SAVE_PREFIX_NAME))
        config.setSavePrefix(vm[SAVE_PREFIX_NAME].as<std::string>());

    if (count_and_not_defaulted(SAVE_EXTENSION_NAME))
        config.setSaveExtension(vm[SAVE_EXTENSION_NAME].as<std::string>());

    if (count_and_not_defaulted(SAVE_FREQ_NAME))
        config.setSaveFreq(vm[SAVE_FREQ_NAME].as<int>());

    if (count_and_not_defaulted(CHECKPOINT_FREQ_NAME))
        config.setCheckpointFreq(vm[CHECKPOINT_FREQ_NAME].as<int>());

    if (count_and_not_defaulted(THRESHOLD_NAME))
        config.setThreshold(vm[THRESHOLD_NAME].as<double>());

    if (count_and_not_defaulted(VERBOSE_NAME))
        config.setVerbose(vm[VERBOSE_NAME].as<int>());

    if (count_and_not_defaulted(STATUS_NAME))
        config.setCsvStatus(vm[STATUS_NAME].as<std::string>());

    if (count_and_not_defaulted(SEED_NAME))
        config.setRandomSeed(vm[SEED_NAME].as<int>());
}

// argc/argv -> variables_map
bool parse_options(po::variables_map &vm, int argc, char *argv[])
{
    try
    {
        po::options_description desc = get_desc();

        if (argc < 2)
        {
            std::cout << desc << std::endl;
            return false;
        }

        po::command_line_parser parser{argc, argv};
        parser.options(desc);

        po::parsed_options parsed_options = parser.run();

        store(parsed_options, vm);
        notify(vm);

        if (vm.count(HELP_NAME))
        {
            std::cout << desc << std::endl;
            return false;
        }

        if (vm.count(VERSION_NAME))
        {
            std::cout << "SMURFF " << smurff::SMURFF_VERSION << std::endl;
            return false;
        }

        return true;
    }
    catch (const po::error &ex)
    {
        std::cerr << "Failed to parse command line arguments: " << std::endl;
        std::cerr << ex.what() << std::endl;
        return false;
    }
    catch (std::runtime_error &ex)
    {
        std::cerr << "Failed to parse command line arguments: " << std::endl;
        std::cerr << ex.what() << std::endl;
        return false;
    }
}

void check_options(po::variables_map &vm)
{

    const std::vector<std::string> train_only_options = {
        TRAIN_NAME, TEST_NAME, PRIOR_NAME, BURNIN_NAME, NSAMPLES_NAME, NUM_LATENT_NAME};

    //-- prediction only
    if (vm.count(PREDICT_NAME))
    {
        if (!vm.count(ROOT_NAME))
            THROWERROR("Need --root option in predict mode");

        for (auto to : train_only_options)
        {
            if (vm.count(to))
                THROWERROR("You're not allowed to mix train options (--" + to + ") with --predict");
        }
    }
    else
    {
        if (!vm.count(TRAIN_NAME))
            THROWERROR("Need --train option in train mode");
    }
}

#endif

#if 0
// Config --> Session
std::shared_ptr<Session> create_train_session(const Config &config)
{

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
    catch (std::runtime_error &ex)
    {
        std::cerr << "Failed to parse command line arguments: " << std::endl;
        std::cerr << ex.what() << std::endl;
        return false;
    }

}
#endif

//create cmd session
//parses args with setFromArgs, then internally calls setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> smurff::create_cmd_session(int argc, char **argv)
{
    std::shared_ptr<ISession> session;

#ifdef HAVE_BOOST
    Config config;
    po::variables_map vm;

    parse_options(vm, argc, argv);
    fill_config(vm, config);
    if (config.isActionTrain())
        session = SessionFactory::create_session(config);
    else
        session = std::make_shared<PredictSession>(config);

#endif

    return session;
}
