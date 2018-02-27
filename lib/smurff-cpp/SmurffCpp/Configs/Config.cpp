#include "Config.h"

#include <set>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>

#include <SmurffCpp/Version.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/IO/INIReader.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>

#define NUM_PRIORS_TAG "num_priors"
#define PRIOR_PREFIX "prior"
#define TRAIN_TAG "train"
#define TEST_TAG "test"
#define NUM_SIDE_INFO_TAG "num_side_info"
#define SIDE_INFO_PREFIX "side_info"
#define NUM_AUX_DATA_TAG "num_aux_data"
#define AUX_DATA_PREFIX "aux_data"
#define SAVE_PREFIX_TAG "save_prefix"
#define SAVE_EXTENSION_TAG "save_extension"
#define SAVE_FREQ_TAG "save_freq"
#define VERBOSE_TAG "verbose"
#define BURNING_TAG "burnin"
#define NSAMPLES_TAG "nsamples"
#define NUM_LATENT_TAG "num_latent"
#define RANDOM_SEED_SET_TAG "random_seed_set"
#define RANDOM_SEED_TAG "random_seed"
#define CSV_STATUS_TAG "csv_status"
#define INIT_MODEL_TAG "init_model"
#define LAMBDA_BETA_TAG "lambda_beta"
#define TOL_TAG "tol"
#define DIRECT_TAG "direct"
#define NOISE_MODEL_TAG "noise_model"
#define PRECISION_TAG "precision"
#define SN_INIT_TAG "sn_init"
#define SN_MAX_TAG "sn_max"
#define NOISE_THRESHOLD_TAG "noise_threshold"
#define CLASSIFY_TAG "classify"
#define THRESHOLD_TAG "threshold"

using namespace smurff;

PriorTypes smurff::stringToPriorType(std::string name)
{
   if(name == PRIOR_NAME_DEFAULT)
      return PriorTypes::default_prior;
   else if(name == PRIOR_NAME_MACAU)
      return PriorTypes::macau;
   else if(name == PRIOR_NAME_MACAU_ONE)
      return PriorTypes::macauone;
   else if(name == PRIOR_NAME_SPIKE_AND_SLAB)
      return PriorTypes::spikeandslab;
   else if(name == PRIOR_NAME_NORMALONE)
      return PriorTypes::normalone;
   else if(name == PRIOR_NAME_NORMAL)
      return PriorTypes::normal;
   else
   {
      THROWERROR("Invalid prior type");
   }
}

std::string smurff::priorTypeToString(PriorTypes type)
{
   switch(type)
   {
      case PriorTypes::default_prior:
         return PRIOR_NAME_DEFAULT;
      case PriorTypes::macau:
         return PRIOR_NAME_MACAU;
      case PriorTypes::macauone:
         return PRIOR_NAME_MACAU_ONE;
      case PriorTypes::spikeandslab:
         return PRIOR_NAME_SPIKE_AND_SLAB;
      case PriorTypes::normal:
         return PRIOR_NAME_NORMAL;
      case PriorTypes::normalone:
         return PRIOR_NAME_NORMALONE;
      default:
      {
         THROWERROR("Invalid prior type");
      }
   }
}

ModelInitTypes smurff::stringToModelInitType(std::string name)
{
   if(name == MODEL_INIT_NAME_RANDOM)
      return ModelInitTypes::random;
   else if (name == MODEL_INIT_NAME_ZERO)
      return ModelInitTypes::zero;
   else
   {
      THROWERROR("Invalid model init type " + name);
   }
}

std::string smurff::modelInitTypeToString(ModelInitTypes type)
{
   switch(type)
   {
      case ModelInitTypes::random:
         return MODEL_INIT_NAME_RANDOM;
      case ModelInitTypes::zero:
         return MODEL_INIT_NAME_ZERO;
      default:
      {
         THROWERROR("Invalid model init type");
      }
   }
}

//config
int Config::BURNIN_DEFAULT_VALUE = 200;
int Config::NSAMPLES_DEFAULT_VALUE = 800;
int Config::NUM_LATENT_DEFAULT_VALUE = 96;
ModelInitTypes Config::INIT_MODEL_DEFAULT_VALUE = ModelInitTypes::zero;
const char* Config::SAVE_PREFIX_DEFAULT_VALUE = "save";
const char* Config::SAVE_EXTENSION_DEFAULT_VALUE = ".csv";
int Config::SAVE_FREQ_DEFAULT_VALUE = 0;
int Config::VERBOSE_DEFAULT_VALUE = 1;
const char* Config::STATUS_DEFAULT_VALUE = "";
double Config::LAMBDA_BETA_DEFAULT_VALUE = 10.0;
double Config::TOL_DEFAULT_VALUE = 1e-6;
double Config::THRESHOLD_DEFAULT_VALUE = 0.0;
int Config::RANDOM_SEED_DEFAULT_VALUE = 0;

//noise config
NoiseTypes Config::NOISE_TYPE_DEFAULT_VALUE = NoiseTypes::fixed;
double Config::PRECISION_DEFAULT_VALUE = 5.0;
double Config::ADAPTIVE_SN_INIT_DEFAULT_VALUE = 1.0;
double Config::ADAPTIVE_SN_MAX_DEFAULT_VALUE = 10.0;
double Config::PROBIT_DEFAULT_VALUE = 0.0;

Config::Config()
{
   m_model_init_type = Config::INIT_MODEL_DEFAULT_VALUE;

   m_save_prefix = Config::SAVE_PREFIX_DEFAULT_VALUE;
   m_save_extension = Config::SAVE_EXTENSION_DEFAULT_VALUE;
   m_save_freq = Config::SAVE_FREQ_DEFAULT_VALUE;

   m_random_seed_set = false;
   m_random_seed = Config::RANDOM_SEED_DEFAULT_VALUE;

   m_verbose = Config::VERBOSE_DEFAULT_VALUE;
   m_csv_status = Config::STATUS_DEFAULT_VALUE;
   m_burnin = Config::BURNIN_DEFAULT_VALUE;
   m_nsamples = Config::NSAMPLES_DEFAULT_VALUE;
   m_num_latent = Config::NUM_LATENT_DEFAULT_VALUE;

   m_lambda_beta = Config::LAMBDA_BETA_DEFAULT_VALUE;
   m_tol = Config::TOL_DEFAULT_VALUE;
   m_direct = false;

   m_threshold = Config::THRESHOLD_DEFAULT_VALUE;
   m_classify = false;
}

bool Config::validate() const
{
   if (!m_train || !m_train->getNNZ())
   {
      THROWERROR("Missing train data");
   }

   if (m_test && !m_test->getNNZ())
   {
      THROWERROR("Missing test data");
   }

   if (m_test && m_test->getDims() != m_train->getDims())
   {
      THROWERROR("Train and test data should have the same dimensions");
   }

   if(m_prior_types.size() != m_train->getNModes())
   {
      THROWERROR("Number of priors should equal to number of dimensions in train data");
   }

   if (m_train->getNModes() > 2)
   {
      for (auto& sideInfo : m_sideInfo)
      {
         if (sideInfo)
         {
            //it is advised to check macau and macauone priors implementation
            //as well as code in PriorFactory that creates macau priors

            //this check does not directly check that input data is Tensor (it only checks number of dimensions)
            //however TensorDataFactory will do an additional check throwing an exception
            THROWERROR("Side info is not supported for TensorData");
         }
      }

      if (!m_auxData.empty())
      {
         //it is advised to check macau and macauone priors implementation
         //as well as code in PriorFactory that creates macau priors
         
         //this check does not directly check that input data is Tensor (it only checks number of dimensions)
         //however TensorDataFactory will do an additional check throwing an exception
         THROWERROR("Aux data is not supported for TensorData");
      }
   }

   //for simplicity we store empty shared_ptr if side info is not specified for dimension
   //this way we can whether that size equals to getNModes
   if (m_sideInfo.size() != m_train->getNModes())
   {
      THROWERROR("Number of side info should equal to number of dimensions in train data");
   }

   for (std::size_t i = 0; i < m_sideInfo.size(); i++)
   {
      const std::shared_ptr<MatrixConfig>& sideInfo = m_sideInfo[i];
      if (sideInfo && sideInfo->getDims()[0] != m_train->getDims()[i])
      {
         std::stringstream ss;
         ss << "Side info should have the same number of rows as size of dimension " << i << " in train data";
         THROWERROR(ss.str());
      }
   }

   /*
   for(auto& ad : m_auxData)
   {
      auto &pos = ad.first;
      auto &data = ad.second;
      if (m_train->getDims()[i] != ad->getDims()[i]) //compare sizes in specific dimension
      {
         std::stringstream ss;
         ss << "Aux data and train data should have the same number of records in dimension " << i;
         THROWERROR(ss.str());
      }
   }
    */

   for(std::size_t i = 0; i < m_prior_types.size(); i++)
   {
      PriorTypes pt = m_prior_types[i];
      const auto& sideInfo = m_sideInfo[i];
      if(pt == PriorTypes::macau)
      {
         if(!sideInfo)
         {
            std::stringstream ss;
            ss << "Side info is always needed when using macau prior in dimension " << i;
            THROWERROR(ss.str());
         }
      }

      if(pt == PriorTypes::macauone)
      {
         if(!sideInfo || sideInfo->isDense())
         {
            std::stringstream ss;
            ss << "Sparse side info is always needed when using macauone prior in dimension " << i;
            THROWERROR(ss.str());
         }
      }
   }

   for(std::size_t i = 0; i < m_prior_types.size(); i++)
   {
      PriorTypes pt = m_prior_types[i];
      switch (pt)
      {
         case PriorTypes::normal:
         case PriorTypes::normalone:
         case PriorTypes::spikeandslab:
         case PriorTypes::default_prior:
            if (m_sideInfo[i])
            {
               std::stringstream ss;
               ss << priorTypeToString(pt) << " prior in dimension " << i << " cannot have side info";
               THROWERROR(ss.str());
            }
            break;
         case PriorTypes::macau:
         case PriorTypes::macauone:
            break;
         default:
            {
               THROWERROR("Unknown prior");
            }
            break;
      }
   }


   std::set<std::string> save_extensions = { ".csv", ".ddm" };

   if (save_extensions.find(m_save_extension) == save_extensions.end())
   {
      THROWERROR("Unknown output extension: " + m_save_extension);
   }

   m_train->getNoiseConfig().validate();

   return true;
}

void Config::save(std::string fname) const
{
   if (!m_save_freq)
      return;

   std::ofstream os(fname);

   auto print_noise_config = [&os](const NoiseConfig &cfg) -> void {
      os << NOISE_MODEL_TAG << "  = " << smurff::noiseTypeToString(cfg.getNoiseType()) << std::endl;
      os << PRECISION_TAG << " = " << cfg.precision << std::endl;
      os << SN_INIT_TAG << " = " << cfg.sn_init << std::endl;
      os << SN_MAX_TAG << " = " << cfg.sn_max << std::endl;
      os << NOISE_THRESHOLD_TAG << " = " << cfg.threshold << std::endl;
   };
      
   
   auto print_tensor_config = [&os, &print_noise_config](const std::string sec_name, int sec_idx,
           const std::shared_ptr<TensorConfig> &cfg,
           const char *noneString = "",
           const PVec<> &pos = PVec<>()
           ) -> void {

       os << std::endl << "[" << sec_name;
       if (sec_idx >= 0) os << "_" << sec_idx;
       os << "]" << std::endl;

       if (cfg) {
           if (pos) os << "pos = " << pos << std::endl;
           os << "file = " << cfg->getFilename() << std::endl;
           os << "is_dense = " << cfg->isDense() << std::endl;
           os << "is_scarce = " << cfg->isScarce() << std::endl;
           print_noise_config(cfg->getNoiseConfig());
       } else {
           os << "file = " << noneString << std::endl;
       }
   };
   
   auto print_tensor_config_pos = [&os, &print_tensor_config](const std::string sec_name, int sec_idx, const PVec<>& pos, const std::shared_ptr<TensorConfig> &cfg) -> void {
      print_tensor_config(sec_name, sec_idx, cfg);
   };
  
   os << "## SMURFF config .ini file" << std::endl;
   os << "# generated file - copy before editing!" << std::endl;
   auto t = std::time(nullptr);
   auto tm = *std::localtime(&t);
   os << "# generated on " << std::put_time(&tm, "%Y-%m-%d %H:%M::pwd%S")  << std::endl;
   os << "# generated by smurff " << SMURFF_VERSION << std::endl;
   
   os << std::endl << "# priors" << std::endl;
   os << NUM_PRIORS_TAG << " = " << m_prior_types.size() << std::endl;
   for(std::size_t pIndex = 0; pIndex < m_prior_types.size(); pIndex++)
      os << PRIOR_PREFIX << "_" << pIndex << " = " << priorTypeToString(m_prior_types.at(pIndex)) << std::endl;

   print_tensor_config("train", -1, m_train, TRAIN_NONE);
   print_tensor_config("test", -1, m_test, TEST_NONE);

   os << std::endl << "# side_info" << std::endl;
   os << NUM_SIDE_INFO_TAG << " = " << m_sideInfo.size() << std::endl;
   for(std::size_t sIndex = 0; sIndex < m_sideInfo.size(); sIndex++)
       print_tensor_config(SIDE_INFO_PREFIX, sIndex, m_sideInfo.at(sIndex), SIDE_INFO_NONE);

   os << std::endl << "# aux_data" << std::endl;
   os << NUM_AUX_DATA_TAG << " = " << m_auxData.size() << std::endl;
   int aux_data_count = 0;
   for(const auto &ad : m_auxData)
       print_tensor_config_pos(AUX_DATA_PREFIX, aux_data_count++, ad.first, ad.second);

   os << std::endl << "# save" << std::endl;
   os << SAVE_PREFIX_TAG << " = " << m_save_prefix << std::endl;
   os << SAVE_EXTENSION_TAG << " = " << m_save_extension << std::endl;
   os << SAVE_FREQ_TAG << " = " << m_save_freq << std::endl;

   os << std::endl << "# general" << std::endl;
   os << VERBOSE_TAG << " = " << m_verbose << std::endl;
   os << BURNING_TAG << " = " << m_burnin << std::endl;
   os << NSAMPLES_TAG  << " = " << m_nsamples << std::endl;
   os << NUM_LATENT_TAG << " = " << m_num_latent << std::endl;
   os << RANDOM_SEED_SET_TAG << " = " << m_random_seed_set << std::endl;
   os << RANDOM_SEED_TAG << " = " << m_random_seed << std::endl;
   os << CSV_STATUS_TAG << " = " << m_csv_status << std::endl;
   os << INIT_MODEL_TAG << " = " << modelInitTypeToString(m_model_init_type) << std::endl;

   os << std::endl << "# for macau priors" << std::endl;
   os << LAMBDA_BETA_TAG << " = " << m_lambda_beta << std::endl;
   os << TOL_TAG << " = " << m_tol << std::endl;
   os << DIRECT_TAG << " = " << m_direct << std::endl;

   os << std::endl << "# binary classification" << std::endl;
   os << CLASSIFY_TAG << " = " << m_classify << std::endl;
   os << THRESHOLD_TAG << " = " << m_threshold << std::endl;
}

bool Config::restore(std::string fname)
{
   THROWERROR_FILE_NOT_EXIST(fname);

   INIReader reader(fname);

   if (reader.ParseError() < 0)
   {
      std::cout << "Can't load '" << fname << "'\n";
      return false;
   }

   //-- test
   std::string testFilename = reader.Get("", TEST_TAG,  TEST_NONE);
   if (testFilename != TEST_NONE)
      setTest(generic_io::read_data_config(testFilename, true));

   //-- train
   std::string trainFilename = reader.Get("", TEST_TAG,  TRAIN_NONE);
   if (trainFilename != TRAIN_NONE)
      setTrain(generic_io::read_data_config(trainFilename, true));

   size_t num_priors = reader.GetInteger("", NUM_PRIORS_TAG, 0);
   for(std::size_t pIndex = 0; pIndex < num_priors; pIndex++)
   {
      std::stringstream ss;

      // -- priors
      ss << PRIOR_PREFIX << "_" << pIndex;
      std::string pName = reader.Get("", ss.str(),  PRIOR_NAME_DEFAULT);
      m_prior_types.push_back(stringToPriorType(pName));

      ss << SIDE_INFO_PREFIX << "_" << pIndex;
      std::string sideInfo = reader.Get("", ss.str(), SIDE_INFO_NONE);
      if (sideInfo == SIDE_INFO_NONE)
          m_sideInfo.push_back(std::shared_ptr<MatrixConfig>());
      else
          m_sideInfo.push_back(matrix_io::read_matrix(sideInfo, false));

      /*
      ss << AUX_DATA_PREFIX << "_" << pIndex;
      std::string auxDataString = reader.Get("", ss.str(), AUX_DATA_NONE);
      m_auxData.push_back(std::vector<std::shared_ptr<TensorConfig> >());
      auto& dimAuxData = m_auxData.back();

      std::stringstream lineStream(auxDataString);
      std::string token;

      while (std::getline(lineStream, token, ','))
      {
          if(token == AUX_DATA_NONE) 
             continue;
          dimAuxData.push_back(matrix_io::read_matrix(token, false));
      }
       */
   }

   //-- save
   m_save_prefix = reader.Get("", SAVE_PREFIX_TAG, Config::SAVE_PREFIX_DEFAULT_VALUE);
   m_save_extension = reader.Get("", SAVE_EXTENSION_TAG, Config::SAVE_EXTENSION_DEFAULT_VALUE);
   m_save_freq = reader.GetInteger("", SAVE_FREQ_TAG, Config::SAVE_FREQ_DEFAULT_VALUE);

   //-- general
   m_verbose = reader.GetInteger("", VERBOSE_TAG, Config::VERBOSE_DEFAULT_VALUE);
   m_burnin = reader.GetInteger("", BURNING_TAG, Config::BURNIN_DEFAULT_VALUE);
   m_nsamples = reader.GetInteger("", NSAMPLES_TAG, Config::NSAMPLES_DEFAULT_VALUE);
   m_num_latent = reader.GetInteger("", NUM_LATENT_TAG, Config::NUM_LATENT_DEFAULT_VALUE);
   m_random_seed_set = reader.GetBoolean("", RANDOM_SEED_SET_TAG,  false);
   m_random_seed = reader.GetInteger("", RANDOM_SEED_TAG, Config::RANDOM_SEED_DEFAULT_VALUE);
   m_csv_status = reader.Get("", CSV_STATUS_TAG, Config::STATUS_DEFAULT_VALUE);
   m_model_init_type = stringToModelInitType(reader.Get("", INIT_MODEL_TAG, modelInitTypeToString(Config::INIT_MODEL_DEFAULT_VALUE)));

   //-- for macau priors
   m_lambda_beta = reader.GetReal("", LAMBDA_BETA_TAG, Config::LAMBDA_BETA_DEFAULT_VALUE);
   m_tol = reader.GetReal("", TOL_TAG, Config::TOL_DEFAULT_VALUE);
   m_direct = reader.GetBoolean("", DIRECT_TAG,  false);

   //-- noise model
   NoiseConfig noise;
   noise.setNoiseType(smurff::stringToNoiseType(reader.Get("", NOISE_MODEL_TAG, noiseTypeToString(Config::NOISE_TYPE_DEFAULT_VALUE))));
   noise.precision = reader.GetReal("", PRECISION_TAG, Config::PRECISION_DEFAULT_VALUE);
   noise.sn_init = reader.GetReal("", SN_INIT_TAG, Config::ADAPTIVE_SN_INIT_DEFAULT_VALUE);
   noise.sn_max = reader.GetReal("", SN_MAX_TAG, Config::ADAPTIVE_SN_MAX_DEFAULT_VALUE);
   noise.threshold = reader.GetReal("", NOISE_THRESHOLD_TAG, Config::PROBIT_DEFAULT_VALUE);
   if (m_train) 
      m_train->setNoiseConfig(noise);

   //-- binary classification
   m_classify = reader.GetBoolean("", CLASSIFY_TAG,  false);
   m_threshold = reader.GetReal("", THRESHOLD_TAG, Config::THRESHOLD_DEFAULT_VALUE);

   return true;
}

