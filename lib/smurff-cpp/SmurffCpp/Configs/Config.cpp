#include "Config.h"

#include <set>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>
#include <memory>

#include <SmurffCpp/Version.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/IO/INIReader.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>

#define GLOBAL_TAG "global"
#define NUM_PRIORS_TAG "num_priors"
#define PRIOR_PREFIX "prior"
#define TRAIN_SECTION_TAG "train"
#define TEST_SECTION_TAG "test"
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
#define POS_TAG "pos"
#define FILE_TAG "file"
#define DENSE_TAG "dense"
#define SCARCE_TAG "scarce"
#define SPARSE_TAG "sparse"
#define TYPE_TAG "type"

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
      if(pt == PriorTypes::macau || pt == PriorTypes::macauone)
      {
         if(!sideInfo)
         {
            std::stringstream ss;
            ss << "Side info is always needed when using macau prior in dimension " << i;
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

   auto print_tensor_config = [&os](const std::string sec_name, int sec_idx,
           const std::shared_ptr<TensorConfig> &cfg) -> void
   {

       os << std::endl << "[" << sec_name;
       if (sec_idx >= 0) os << "_" << sec_idx;
       os << "]" << std::endl;

       if (cfg) {
           if (cfg->getPos()) os << POS_TAG << " = " << cfg->getPos() << std::endl;
           os << FILE_TAG << " = " << cfg->getFilename() << std::endl;
           std::string type_str = cfg->isDense() ? DENSE_TAG : cfg->isScarce() ? SCARCE_TAG : SPARSE_TAG;
           os << TYPE_TAG << " = " << type_str << std::endl;
           auto &noise_config = cfg->getNoiseConfig();
           if (noise_config.getNoiseType() != NoiseTypes::unset) {
               os << NOISE_MODEL_TAG << "  = " << smurff::noiseTypeToString(noise_config.getNoiseType()) << std::endl;
               os << PRECISION_TAG << " = " << noise_config.precision << std::endl;
               os << SN_INIT_TAG << " = " << noise_config.sn_init << std::endl;
               os << SN_MAX_TAG << " = " << noise_config.sn_max << std::endl;
               os << NOISE_THRESHOLD_TAG << " = " << noise_config.threshold << std::endl;
           }
       } else {
           os << "file = " << NONE_TAG << std::endl;
       }
   };
   
   os << "## SMURFF config .ini file" << std::endl;
   os << "# generated file - copy before editing!" << std::endl;
   auto t = std::time(nullptr);
   auto tm = *std::localtime(&t);
   char time_str[1024];
   strftime (time_str, 1023, "%Y-%m-%d %H:%M:%S", &tm);
   os << "# generated on " << time_str << std::endl;
   os << "# generated by smurff " << SMURFF_VERSION << std::endl;

   os << std::endl << "[" << GLOBAL_TAG << "]" << std::endl;

   os << std::endl << "# count" << std::endl;
   os << NUM_PRIORS_TAG << " = " << m_prior_types.size() << std::endl;
   os << NUM_AUX_DATA_TAG << " = " << m_auxData.size() << std::endl;
   os << NUM_SIDE_INFO_TAG << " = " << m_sideInfo.size() << std::endl;
   
   os << std::endl << "# priors" << std::endl;
   for(std::size_t pIndex = 0; pIndex < m_prior_types.size(); pIndex++)
      os << PRIOR_PREFIX << "_" << pIndex << " = " << priorTypeToString(m_prior_types.at(pIndex)) << std::endl;

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

   print_tensor_config("train", -1, m_train);
   print_tensor_config("test", -1, m_test);

   for(std::size_t sIndex = 0; sIndex < m_sideInfo.size(); sIndex++)
       print_tensor_config(SIDE_INFO_PREFIX, sIndex, m_sideInfo.at(sIndex));

   int aux_data_count = 0;
   for(const auto &ad : m_auxData)
       print_tensor_config(AUX_DATA_PREFIX, aux_data_count++, ad);
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

   auto add_index = [](const std::string name, int idx = -1) -> std::string
   {
       if (idx >=0) return name + "_" + std::to_string(idx);
       return name;
   };

   auto split = [](const std::string &str, const char delim) -> std::vector<int>
   {
       std::vector<int> ret;
       std::stringstream lineStream(str);
       std::string token;
       while (std::getline(lineStream, token, delim))
           ret.push_back(std::stoi(token));
       return ret;
   };

   auto restore_tensor_config = [&reader, &split](const std::string sec_name) -> std::shared_ptr<TensorConfig>
   {
       //-- basic stuff
       std::string filename = reader.Get(sec_name, FILE_TAG,  NONE_TAG);
       if (filename == NONE_TAG) return std::shared_ptr<TensorConfig>();
       bool is_scarce = reader.Get(sec_name, TYPE_TAG, SCARCE_TAG) == SCARCE_TAG;

       //--read file
       auto cfg = generic_io::read_data_config(filename, is_scarce);

       //-- pos 
       std::string pos_str = reader.Get(sec_name, POS_TAG, NONE_TAG);
       if (pos_str != NONE_TAG) cfg->setPos(PVec<>(split(pos_str, ',')));

       //-- noise model
       NoiseConfig noise;
       noise.setNoiseType(smurff::stringToNoiseType(reader.Get(sec_name, NOISE_MODEL_TAG, noiseTypeToString(Config::NOISE_TYPE_DEFAULT_VALUE))));
       noise.precision = reader.GetReal(sec_name, PRECISION_TAG, Config::PRECISION_DEFAULT_VALUE);
       noise.sn_init = reader.GetReal(sec_name, SN_INIT_TAG, Config::ADAPTIVE_SN_INIT_DEFAULT_VALUE);
       noise.sn_max = reader.GetReal(sec_name, SN_MAX_TAG, Config::ADAPTIVE_SN_MAX_DEFAULT_VALUE);
       noise.threshold = reader.GetReal(sec_name, NOISE_THRESHOLD_TAG, Config::PROBIT_DEFAULT_VALUE);
       cfg->setNoiseConfig(noise);
       
       return cfg;
   };

   //-- test and 
   setTest(restore_tensor_config(TEST_SECTION_TAG));
   setTrain(restore_tensor_config(TRAIN_SECTION_TAG));

   size_t num_priors = reader.GetInteger(GLOBAL_TAG, NUM_PRIORS_TAG, 0);
   for(std::size_t pIndex = 0; pIndex < num_priors; pIndex++)
   {
      std::string pName = reader.Get(GLOBAL_TAG, add_index(PRIOR_PREFIX, pIndex),  PRIOR_NAME_DEFAULT);
      m_prior_types.push_back(stringToPriorType(pName));
   }

   size_t num_side_info = reader.GetInteger(GLOBAL_TAG, NUM_SIDE_INFO_TAG, 0);
   for(std::size_t pIndex = 0; pIndex < num_side_info; pIndex++)
   {
      auto tensor_cfg = restore_tensor_config(add_index(SIDE_INFO_PREFIX, pIndex));
      auto matrix_cfg = std::dynamic_pointer_cast<MatrixConfig>(tensor_cfg);
      m_sideInfo.push_back(matrix_cfg);
   }

   size_t num_aux_data = reader.GetInteger(GLOBAL_TAG, NUM_AUX_DATA_TAG, 0);
   for(std::size_t pIndex = 0; pIndex < num_aux_data; pIndex++)
   {
       m_auxData.push_back(restore_tensor_config(add_index(AUX_DATA_PREFIX, pIndex)));
   }

   //-- save
   m_save_prefix = reader.Get(GLOBAL_TAG, SAVE_PREFIX_TAG, Config::SAVE_PREFIX_DEFAULT_VALUE);
   m_save_extension = reader.Get(GLOBAL_TAG, SAVE_EXTENSION_TAG, Config::SAVE_EXTENSION_DEFAULT_VALUE);
   m_save_freq = reader.GetInteger(GLOBAL_TAG, SAVE_FREQ_TAG, Config::SAVE_FREQ_DEFAULT_VALUE);

   //-- general
   m_verbose = reader.GetInteger(GLOBAL_TAG, VERBOSE_TAG, Config::VERBOSE_DEFAULT_VALUE);
   m_burnin = reader.GetInteger(GLOBAL_TAG, BURNING_TAG, Config::BURNIN_DEFAULT_VALUE);
   m_nsamples = reader.GetInteger(GLOBAL_TAG, NSAMPLES_TAG, Config::NSAMPLES_DEFAULT_VALUE);
   m_num_latent = reader.GetInteger(GLOBAL_TAG, NUM_LATENT_TAG, Config::NUM_LATENT_DEFAULT_VALUE);
   m_random_seed_set = reader.GetBoolean(GLOBAL_TAG, RANDOM_SEED_SET_TAG,  false);
   m_random_seed = reader.GetInteger(GLOBAL_TAG, RANDOM_SEED_TAG, Config::RANDOM_SEED_DEFAULT_VALUE);
   m_csv_status = reader.Get(GLOBAL_TAG, CSV_STATUS_TAG, Config::STATUS_DEFAULT_VALUE);
   m_model_init_type = stringToModelInitType(reader.Get(GLOBAL_TAG, INIT_MODEL_TAG, modelInitTypeToString(Config::INIT_MODEL_DEFAULT_VALUE)));

   //-- for macau priors
   m_lambda_beta = reader.GetReal(GLOBAL_TAG, LAMBDA_BETA_TAG, Config::LAMBDA_BETA_DEFAULT_VALUE);
   m_tol = reader.GetReal(GLOBAL_TAG, TOL_TAG, Config::TOL_DEFAULT_VALUE);
   m_direct = reader.GetBoolean(GLOBAL_TAG, DIRECT_TAG,  false);

   //-- binary classification
   m_classify = reader.GetBoolean(GLOBAL_TAG, CLASSIFY_TAG,  false);
   m_threshold = reader.GetReal(GLOBAL_TAG, THRESHOLD_TAG, Config::THRESHOLD_DEFAULT_VALUE);

   return true;
}

