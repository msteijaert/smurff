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
   else if(name == PRIOR_NAME_NORMALONE)
      return PriorTypes::normalone;
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
      case PriorTypes::normalone:
         return PRIOR_NAME_NORMALONE;
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

   for (const auto r: m_row_features)
   {
      if (m_train->getDims()[0] != r->getNRow())
      {
         THROWERROR("Row features and train data should have the same number of rows");
      }
   }

   for (const auto c: m_col_features)
   {
      if (m_train->getDims()[1] != c->getNCol())
      {
         THROWERROR("Column features and train data should have the same number of cols");
      }
   }

   if (col_prior_type == PriorTypes::macau && m_col_features.size() != 1)
   {
      THROWERROR("Exactly one set of col-features needed when using macau prior.");
   }

   if (row_prior_type == PriorTypes::macau && m_row_features.size() != 1)
   {
      THROWERROR("Exactly one set of row-features needed when using macau prior.");
   }

   if (col_prior_type == PriorTypes::macauone && (m_col_features.size() != 1 || m_col_features.at(0)->isDense()))
   {
      THROWERROR("Exactly one set of sparse col-features needed when using macauone prior.");
   }

   if (row_prior_type == PriorTypes::macauone && (m_row_features.size() != 1 || m_row_features.at(0)->isDense()))
   {
      THROWERROR("Exactly one set of sparse row-features needed when using macauone prior.");
   }

   std::set<std::string> save_suffixes = { ".csv", ".ddm" };

   if (save_suffixes.find(save_suffix) == save_suffixes.end())
   {
      THROWERROR("Unknown output suffix: " + save_suffix);
   }

   m_train->getNoiseConfig().validate();

   return true;
}

void Config::save(std::string fname) const
{
   if (!save_freq)
      return;

   std::ofstream os(fname);

   os << "# train = ";
   m_train->info(os);
   os << std::endl;

   os << "# test = ";
   m_test->info(os);
   os << std::endl;

   os << "# features" << std::endl;

   auto print_features = [&os](std::string name, const std::vector<std::shared_ptr<MatrixConfig> > &vec) -> void
   {
      os << "[" << name << "]\n";
      for (size_t i = 0; i < vec.size(); ++i)
      {
         os << "# " << i << " ";
         vec.at(i)->info(os);
         os << std::endl;
      }
   };

   print_features("row_features", m_row_features);
   print_features("col_features", m_col_features);

   os << "# priors" << std::endl;
   os << "row_prior = " << priorTypeToString(row_prior_type) << std::endl;
   os << "col_prior = " << priorTypeToString(col_prior_type) << std::endl;

   os << "# restore" << std::endl;
   os << "restore_prefix = " << restore_prefix << std::endl;
   os << "restore_suffix = " << restore_suffix << std::endl;
   os << "init_model = " << modelInitTypeToString(model_init_type) << std::endl;

   os << "# save" << std::endl;
   os << "save_prefix = " << save_prefix << std::endl;
   os << "save_suffix = " << save_suffix << std::endl;
   os << "save_freq = " << save_freq << std::endl;

   os << "# general" << std::endl;
   os << "verbose = " << verbose << std::endl;
   os << "burnin = " << burnin << std::endl;
   os << "nsamples = " << nsamples << std::endl;
   os << "num_latent = " << num_latent << std::endl;

   os << "# for macau priors" << std::endl;
   os << "lambda_beta = " << lambda_beta << std::endl;
   os << "tol = " << tol << std::endl;
   os << "direct = " << direct << std::endl;

   os << "# noise model" << std::endl;
   os << "noise_model = " << smurff::noiseTypeToString(m_train->getNoiseConfig().getNoiseType()) << std::endl;
   os << "precision = " << m_train->getNoiseConfig().precision << std::endl;
   os << "sn_init = " << m_train->getNoiseConfig().sn_init << std::endl;
   os << "sn_max = " << m_train->getNoiseConfig().sn_max << std::endl;

   os << "# binary classification" << std::endl;
   os << "classify = " << classify << std::endl;
   os << "threshold = " << threshold << std::endl;
}

void Config::restore(std::string fname)
{
   INIReader reader(fname);

   if (reader.ParseError() < 0)
   {
      std::cout << "Can't load '" << fname << "'\n";
   }

   // -- priors
   row_prior_type = stringToPriorType(reader.Get("", "row_prior",  PRIOR_NAME_DEFAULT));
   col_prior_type = stringToPriorType(reader.Get("", "col_prior",  PRIOR_NAME_DEFAULT));

   //-- restore
   restore_prefix = reader.Get("", "restore_prefix",  "");
   restore_suffix = reader.Get("", "restore_suffix",  ".csv");
   model_init_type = stringToModelInitType(reader.Get("", "init_model", MODEL_INIT_NAME_RANDOM));

   //-- save
   save_prefix = reader.Get("", "save_prefix",  "save");
   save_suffix = reader.Get("", "save_suffix",  ".csv");
   save_freq = reader.GetInteger("", "save_freq",  0); // never

   //-- general
   verbose = reader.GetBoolean("", "verbose",  false);
   burnin = reader.GetInteger("", "burnin",  200);
   nsamples = reader.GetInteger("", "nsamples",  800);
   num_latent = reader.GetInteger("", "num_latent",  96);

   //-- for macau priors
   lambda_beta = reader.GetReal("", "lambda_beta",  10.0);
   tol = reader.GetReal("", "tol",  1e-6);
   direct = reader.GetBoolean("", "direct",  false);

   //-- noise model
   NoiseConfig noise;
   noise.setNoiseType(smurff::stringToNoiseType(reader.Get("", "noise_model",  NOISE_NAME_FIXED)));
   noise.precision = reader.GetReal("", "precision",  5.0);
   noise.sn_init = reader.GetReal("", "sn_init",  1.0);
   noise.sn_max = reader.GetReal("", "sn_max",  10.0);
   m_train->setNoiseConfig(noise);

   //-- binary classification
   classify = reader.GetBoolean("", "classify",  false);
   threshold = reader.GetReal("", "threshold",  .0);
};

