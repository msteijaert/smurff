#include "Config.h"

#include <set>
#include <iostream>

#include "utils.h"
#include <IO/INIReader.h>
#include <DataMatrices/Data.h>

using namespace smurff;

bool Config::validate(bool throw_error) const
{
   if (!train.getRows().size())
      die("Missing train matrix");

   std::set<std::string> prior_names = { "default", "normal", "spikeandslab", "macau", "macauone" };

   if (prior_names.find(col_prior) == prior_names.end()) 
      die("Unknown col_prior " + col_prior);

   if (prior_names.find(row_prior) == prior_names.end()) 
      die("Unknown row_prior " + row_prior);

   if(IMeanCentering::stringToCenterMode(center_mode) == IMeanCentering::CenterModeTypes::CENTER_INVALID)
      die("Unknown center mode " + center_mode);

   if (test.getNRow() > 0 && train.getNRow() > 0 && test.getNRow() != train.getNRow())
      die("Train and test matrix should have the same number of rows");

   if (test.getNCol() > 0 && train.getNCol() > 0 && test.getNCol() != train.getNCol())
      die("Train and test matrix should have the same number of cols");

   std::set<std::string> save_suffixes = { ".csv", ".ddm" };

   if (save_suffixes.find(save_suffix) == save_suffixes.end()) 
      die("Unknown output suffix: " + save_suffix);

   std::set<std::string> init_models = { "random", "zero" };
   
   if (init_models.find(init_model) == init_models.end()) 
      die("Unknown init model " + init_model);

   train.getNoiseConfig().validate();

   return true;
}

void Config::save(std::string fname) const
{
   if (!save_freq) 
      return;

   std::ofstream os(fname);

   os << "# train = "; train.info(os); os << std::endl;
   os << "# test = "; test.info(os); os << std::endl;

   os << "# features" << std::endl;

   auto print_features = [&os](std::string name, const std::vector<MatrixConfig> &vec) -> void {
      os << "[" << name << "]\n";
      for (unsigned i=0; i<vec.size(); ++i) {
         os << "# " << i << " ";
         vec.at(i).info(os);
         os << std::endl;
      }
   };

   print_features("row_features", row_features);
   print_features("col_features", col_features);

   os << "# priors" << std::endl;
   os << "row_prior = " << row_prior << std::endl;
   os << "col_prior = " << col_prior << std::endl;

   os << "# restore" << std::endl;
   os << "restore_prefix = " << restore_prefix << std::endl;
   os << "restore_suffix = " << restore_suffix << std::endl;
   os << "init_model = " << init_model << std::endl;

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
   os << "noise_model = " << train.getNoiseConfig().name << std::endl;
   os << "precision = " << train.getNoiseConfig().precision << std::endl;
   os << "sn_init = " << train.getNoiseConfig().sn_init << std::endl;
   os << "sn_max = " << train.getNoiseConfig().sn_max << std::endl;

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
   row_prior = reader.Get("", "row_prior",  "default");
   col_prior = reader.Get("", "col_prior",  "default");

   //-- restore
   restore_prefix = reader.Get("", "restore_prefix",  "");
   restore_suffix = reader.Get("", "restore_suffix",  ".csv");
   init_model     = reader.Get("", "init_model", "random");

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
   noise.name = reader.Get("", "noise_model",  "fixed");
   noise.precision = reader.GetReal("", "precision",  5.0);
   noise.sn_init = reader.GetReal("", "sn_init",  1.0);
   noise.sn_max = reader.GetReal("", "sn_max",  10.0);
   train.setNoiseConfig(noise);

   //-- binary classification
   classify = reader.GetBoolean("", "classify",  false);
   threshold = reader.GetReal("", "threshold",  .0);
};

std::string Config::version() 
{
   return
#ifdef SMURFF_VERSION
   SMURFF_VERSION
#else
   "unknown"
#endif
   ;
}