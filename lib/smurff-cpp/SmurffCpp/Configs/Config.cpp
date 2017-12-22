#include "Config.h"

#include <set>
#include <iostream>
#include <fstream>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/IO/INIReader.h>
#include <SmurffCpp/DataMatrices/Data.h>

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
   else if(name == PRIOR_NAME_NORMAL)
      return PriorTypes::normal;
   else if(name == PRIOR_NAME_MPI)
      return PriorTypes::mpi;
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
      case PriorTypes::mpi:
         return PRIOR_NAME_MPI;
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

   for (std::size_t i = 2; i < m_features.size(); i++)
   {
      if (!m_features.at(i).empty())
      {
         //it is advised to check macau and macauone priors implementation
         //as well as code in PriorFactory that creates macau priors

         //this check does not directly check that input data is Tensor (it only checks number of dimensions)
         //however TensorDataFactory will do an additional check throwing an exception
         THROWERROR("Features are not supported for TensorData");
      }
   }

   //for simplicity we store empty vector if features are not specified for dimension
   //this way we can that size equals to getNModes
   if(m_features.size() != m_train->getNModes())
   {
      THROWERROR("Number of feature sets should equal to number of dimensions in train data");
   }

   for(std::size_t i = 0; i < m_features.size(); i++) //go through each dimension
   {
      const auto& featureSet = m_features[i]; //features for specific dimension
      for(auto& ft : featureSet)
      {
         if(i > 1)
         {
            //it is advised to check macau and macauone priors implementation
            //as well as code in PriorFactory that creates macau priors

            //this check does not directly check that input data is Tensor (it only checks number of dimensions)
            //however TensorDataFactory will do an additional check throwing an exception
            THROWERROR("Features are not supported for TensorData");
         }

         //AGE: not sure how strict should be the check. which adjacent dimensions do we need to check?
         if (m_train->getDims()[i] != ft->getDims()[i]) //compare sizes in specific dimension
         {
            std::stringstream ss;
            ss << "Features and train data should have the same number of records in dimension " << i;
            THROWERROR(ss.str());
         }
      }
   }

   for(std::size_t i = 0; i < m_prior_types.size(); i++)
   {
      PriorTypes pt = m_prior_types[i];
      const auto& featureSet = m_features[i];
      if(pt == PriorTypes::macau)
      {
         if(i > 1)
         {
            //it is advised to check macau and macauone priors implementation
            //as well as code in PriorFactory that creates macau priors

            //this check does not directly check that input data is Tensor (it only checks number of dimensions)
            //however PriorFactory will do an additional check
            THROWERROR("Macau and MacauOne priors are not supported for TensorData");
         }

         if(featureSet.size() != 1)
         {
            std::stringstream ss;
            ss << "Exactly one set of features needed when using macau prior in dimension " << i;
            THROWERROR(ss.str());
         }
      }

      if(pt == PriorTypes::macauone)
      {
         if(i > 1)
         {
            //it is advised to check macau and macauone priors implementation
            //as well as code in PriorFactory that creates macau priors
            THROWERROR("Macau and MacauOne priors are not supported for TensorData");
         }

         if(featureSet.size() != 1 || featureSet.at(0)->isDense())
         {
            std::stringstream ss;
            ss << "Exactly one set of sparse col-features needed when using macauone prior in dimension " << i;
            THROWERROR(ss.str());
         }
      }
   }

   std::set<std::string> save_suffixes = { ".csv", ".ddm" };

   if (save_suffixes.find(m_save_suffix) == save_suffixes.end())
   {
      THROWERROR("Unknown output suffix: " + m_save_suffix);
   }

   m_train->getNoiseConfig().validate();

   return true;
}

void Config::save(std::string fname) const
{
   if (!m_save_freq)
      return;

   std::ofstream os(fname);

   os << "# train = ";
   m_train->info(os);
   os << std::endl;

   os << "# test = ";
   m_test->info(os);
   os << std::endl;

   os << "# features" << std::endl;

   auto print_features = [&os](const std::vector<std::vector<std::shared_ptr<MatrixConfig> > > &vec) -> void
   {
      os << "num_features = " << vec.size() << std::endl;

      for(std::size_t sIndex = 0; sIndex < vec.size(); sIndex++)
      {
         os << "[" << "features" << sIndex << "]\n";

         auto& fset = vec.at(sIndex);

         for(std::size_t fIndex = 0; fIndex < fset.size(); fIndex++)
         {
            os << "# " << fIndex << " ";
            fset.at(fIndex)->info(os);
            os << std::endl;
         }
      }
   };

   print_features(m_features);

   os << "# priors" << std::endl;
   os << "num_priors = " << m_prior_types.size() << std::endl;
   for(std::size_t pIndex = 0; pIndex < m_prior_types.size(); pIndex++)
   {
      os << "prior" << pIndex << " = " << priorTypeToString(m_prior_types.at(pIndex)) << std::endl;
   }

   os << "# restore" << std::endl;
   os << "restore_prefix = " << m_restore_prefix << std::endl;
   os << "restore_suffix = " << m_restore_suffix << std::endl;
   os << "init_model = " << modelInitTypeToString(m_model_init_type) << std::endl;

   os << "# save" << std::endl;
   os << "save_prefix = " << m_save_prefix << std::endl;
   os << "save_suffix = " << m_save_suffix << std::endl;
   os << "save_freq = " << m_save_freq << std::endl;

   os << "# general" << std::endl;
   os << "verbose = " << m_verbose << std::endl;
   os << "burnin = " << m_burnin << std::endl;
   os << "nsamples = " << m_nsamples << std::endl;
   os << "num_latent = " << m_num_latent << std::endl;

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

void Config::restore(std::string fname)
{
   INIReader reader(fname);

   if (reader.ParseError() < 0)
   {
      std::cout << "Can't load '" << fname << "'\n";
   }

   // -- priors
   size_t num_priors = reader.GetInteger("", "num_priors", 0);
   for(std::size_t pIndex = 0; pIndex < num_priors; pIndex++)
   {
      std::stringstream ss;
      ss << "prior" << pIndex;
      std::string pName = reader.Get("", ss.str(),  PRIOR_NAME_DEFAULT);
      m_prior_types.push_back(stringToPriorType(pName));
   }

   //-- restore
   m_restore_prefix = reader.Get("", "restore_prefix",  "");
   m_restore_suffix = reader.Get("", "restore_suffix",  ".csv");
   m_model_init_type = stringToModelInitType(reader.Get("", "init_model", MODEL_INIT_NAME_RANDOM));

   //-- save
   m_save_prefix = reader.Get("", "save_prefix",  "save");
   m_save_suffix = reader.Get("", "save_suffix",  ".csv");
   m_save_freq = reader.GetInteger("", "save_freq",  0); // never

   //-- general
   m_verbose = reader.GetBoolean("", "verbose",  false);
   m_burnin = reader.GetInteger("", "burnin",  200);
   m_nsamples = reader.GetInteger("", "nsamples",  800);
   m_num_latent = reader.GetInteger("", "num_latent",  96);

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
   m_train->setNoiseConfig(noise);

   //-- binary classification
   m_classify = reader.GetBoolean("", "classify",  false);
   m_threshold = reader.GetReal("", "threshold",  .0);
}

