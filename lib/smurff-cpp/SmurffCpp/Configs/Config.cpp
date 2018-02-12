#include "Config.h"

#include <set>
#include <iostream>
#include <fstream>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/IO/INIReader.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>



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

      for (auto& auxData : m_auxData)
      {
         if (!auxData.empty())
         {
            //it is advised to check macau and macauone priors implementation
            //as well as code in PriorFactory that creates macau priors

            //this check does not directly check that input data is Tensor (it only checks number of dimensions)
            //however TensorDataFactory will do an additional check throwing an exception
            THROWERROR("Aux data is not supported for TensorData");
         }
      }
   }

   //for simplicity we store empty shared_ptr if side info is not specified for dimension
   //this way we can whether that size equals to getNModes
   if (m_sideInfo.size() != m_train->getNModes())
   {
      THROWERROR("Number of side info should equal to number of dimensions in train data");
   }

   //for simplicity we store empty vector if aux data is not specified for dimension
   //this way we can whether that size equals to getNModes
   if (m_auxData.size() != m_train->getNModes())
   {
      THROWERROR("Number of aux data should equal to number of dimensions in train data");
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

   for (std::size_t i = 0; i < m_auxData.size(); i++) //go through each dimension
   {
      const auto& auxDataSet = m_auxData[i];
      for(auto& ad : auxDataSet)
      {
         //AGE: not sure how strict should be the check. which adjacent dimensions do we need to check?
         if (m_train->getDims()[i] != ad->getDims()[i]) //compare sizes in specific dimension
         {
            std::stringstream ss;
            ss << "Aux data and train data should have the same number of records in dimension " << i;
            THROWERROR(ss.str());
         }
      }
   }

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
            if (!m_auxData[i].empty())
            {
               std::stringstream ss;
               ss << priorTypeToString(pt) << " prior in dimension " << i << " cannot have aux data";
            }
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

   auto print_tensor_config = [&os](const std::shared_ptr<TensorConfig> &cfg, const std::string name, int idx = -1 ) -> void {
       os << name;
       if (idx >= 0) os << "_" << idx;
       os << " = ";
       if (cfg) cfg->save(os);
       else os << "none";
       os << std::endl;
   };

   auto print_tensor_config_vector = [&os](const std::vector<std::shared_ptr<TensorConfig>> &vec, const std::string name, int idx = -1 ) -> void {
       os << name;
       if (idx >= 0) os << "_" << idx;
       os << " = ";
       if (vec.empty()) os << "none";
       else for(const auto &cfg : vec) {
           cfg->save(os);
           os << ",";
       }
       os << std::endl;
   };


   os << "# priors" << std::endl;
   os << "num_priors = " << m_prior_types.size() << std::endl;
   for(std::size_t pIndex = 0; pIndex < m_prior_types.size(); pIndex++)
   {
      os << "prior_" << pIndex << " = " << priorTypeToString(m_prior_types.at(pIndex)) << std::endl;
   }

   print_tensor_config(m_train, "train");
   print_tensor_config(m_test , "test");

   os << "# side_info" << std::endl;
   for(std::size_t sIndex = 0; sIndex < m_sideInfo.size(); sIndex++)
       print_tensor_config(m_sideInfo.at(sIndex), "side_info", sIndex);

   os << "# aux_data" << std::endl;
   for(std::size_t sIndex = 0; sIndex < m_auxData.size(); sIndex++)
   {
       print_tensor_config_vector(m_auxData.at(sIndex), "aux_data", sIndex);
   }

   os << "# save" << std::endl;
   os << "save_prefix = " << m_save_prefix << std::endl;
   os << "save_extension = " << m_save_extension << std::endl;
   os << "save_freq = " << m_save_freq << std::endl;

   os << "# general" << std::endl;
   os << "verbose = " << m_verbose << std::endl;
   os << "burnin = " << m_burnin << std::endl;
   os << "nsamples = " << m_nsamples << std::endl;
   os << "num_latent = " << m_num_latent << std::endl;
   os << "random_seed_set = " << m_random_seed_set << std::endl;
   os << "random_seed = " << m_random_seed << std::endl;
   os << "csv_status = " << m_csv_status << std::endl;
   os << "init_model = " << modelInitTypeToString(m_model_init_type) << std::endl;

   os << "# for macau priors" << std::endl;
   os << "lambda_beta = " << m_lambda_beta << std::endl;
   os << "tol = " << m_tol << std::endl;
   os << "direct = " << m_direct << std::endl;

   os << "# noise model" << std::endl;
   os << "noise_model = " << smurff::noiseTypeToString(m_train->getNoiseConfig().getNoiseType()) << std::endl;
   os << "precision = " << m_train->getNoiseConfig().precision << std::endl;
   os << "sn_init = " << m_train->getNoiseConfig().sn_init << std::endl;
   os << "sn_max = " << m_train->getNoiseConfig().sn_max << std::endl;

   os << "# binary classification" << std::endl;
   os << "classify = " << m_classify << std::endl;
   os << "threshold = " << m_threshold << std::endl;
}

bool Config::restore(std::string fname)
{
   INIReader reader(fname);

   if (reader.ParseError() < 0)
   {
      std::cout << "Can't load '" << fname << "'\n";
      return false;
   }

   //-- test
   std::string testFilename = reader.Get("", "test",  TEST_NONE);
   if (testFilename != TEST_NONE)
      setTest(generic_io::read_data_config(testFilename, true));

   //-- train
   std::string trainFilename = reader.Get("", "train",  TRAIN_NONE);
   if (trainFilename != TRAIN_NONE)
      setTrain(generic_io::read_data_config(trainFilename, true));

   size_t num_priors = reader.GetInteger("", "num_priors", 0);
   for(std::size_t pIndex = 0; pIndex < num_priors; pIndex++)
   {
      std::stringstream ss;

      // -- priors
      ss << "prior_" << pIndex;
      std::string pName = reader.Get("", ss.str(),  PRIOR_NAME_DEFAULT);
      m_prior_types.push_back(stringToPriorType(pName));

      ss << "side_info_" << pIndex;
      std::string sideInfo = reader.Get("", ss.str(), SIDE_INFO_NONE);
      if (sideInfo == SIDE_INFO_NONE)
          m_sideInfo.push_back(std::shared_ptr<MatrixConfig>());
      else
          m_sideInfo.push_back(matrix_io::read_matrix(sideInfo, false));

      ss << "aux_data_" << pIndex;
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
   }

   //-- save
   m_save_prefix = reader.Get("", "save_prefix",  "save");
   m_save_extension = reader.Get("", "save_extension",  ".csv");
   m_save_freq = reader.GetInteger("", "save_freq",  0); // never

   //-- general
   m_verbose = reader.GetInteger("", "verbose",  false);
   m_burnin = reader.GetInteger("", "burnin",  200);
   m_nsamples = reader.GetInteger("", "nsamples",  800);
   m_num_latent = reader.GetInteger("", "num_latent",  96);
   m_random_seed_set = reader.GetBoolean("", "random_seed_set",  false);
   m_random_seed = reader.GetInteger("", "random_seed",  -1);
   m_csv_status = reader.Get("", "csv_status",  "status.csv");
   m_model_init_type = stringToModelInitType(reader.Get("", "init_model", MODEL_INIT_NAME_RANDOM));

   //-- for macau priors
   m_lambda_beta = reader.GetReal("", "lambda_beta",  10.0);
   m_tol = reader.GetReal("", "tol",  1e-6);
   m_direct = reader.GetBoolean("", "direct",  false);

   //-- noise model
   NoiseConfig noise;
   noise.setNoiseType(smurff::stringToNoiseType(reader.Get("", "noise_model",  NOISE_NAME_FIXED)));
   noise.precision = reader.GetReal("", "precision",  5.0);
   noise.sn_init = reader.GetReal("", "sn_init",  1.0);
   noise.sn_max = reader.GetReal("", "sn_max",  10.0);
   if (m_train) 
      m_train->setNoiseConfig(noise);

   //-- binary classification
   m_classify = reader.GetBoolean("", "classify",  false);
   m_threshold = reader.GetReal("", "threshold",  .0);

   return true;
}

