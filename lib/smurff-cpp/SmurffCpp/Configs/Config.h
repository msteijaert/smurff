#pragma once

#include <string>
#include <vector>
#include <memory>

#include "MatrixConfig.h"

#define PRIOR_NAME_DEFAULT "default"
#define PRIOR_NAME_MACAU "macau"
#define PRIOR_NAME_MACAU_ONE "macauone"
#define PRIOR_NAME_SPIKE_AND_SLAB "spikeandslab"
#define PRIOR_NAME_NORMAL "normal"
#define PRIOR_NAME_NORMALONE "normalone"
#define PRIOR_NAME_MPI "mpi"

#define CENTER_MODE_STR_NONE "none"
#define CENTER_MODE_STR_GLOBAL "global"
#define CENTER_MODE_STR_VIEW "view"
#define CENTER_MODE_STR_ROWS "rows"
#define CENTER_MODE_STR_COLS "cols"

#define MODEL_INIT_NAME_RANDOM "random"
#define MODEL_INIT_NAME_ZERO "zero"

namespace smurff {

enum class PriorTypes
{
   default_prior,
   macau,
   macauone,
   spikeandslab,
   normal,
   normalone,
   mpi
};

enum class ModelInitTypes
{
   random,
   zero
};

PriorTypes stringToPriorType(std::string name);

std::string priorTypeToString(PriorTypes type);

ModelInitTypes stringToModelInitType(std::string name);

std::string modelInitTypeToString(ModelInitTypes type);

struct Config
{
private:
   //-- train and test
   std::shared_ptr<TensorConfig> m_train;
   std::shared_ptr<TensorConfig> m_test;

   //-- sideinfo
   std::vector<std::shared_ptr<MatrixConfig> > m_sideInfo; //set of side info matrices for macau and macauone priors

   //-- aux data
   std::vector<std::vector<std::shared_ptr<TensorConfig> > > m_auxData; //set of aux data matrices for normal and spikeandslab priors

   // -- priors
   std::vector<PriorTypes> m_prior_types;

   //-- restore
   std::string m_restore_prefix;
   std::string m_restore_suffix;

   //-- init model
   ModelInitTypes m_model_init_type;

   //-- save
   std::string m_save_prefix;
   std::string m_save_suffix;
   int m_save_freq;

   //-- general
   bool m_random_seed_set;
   int m_random_seed;
   int m_verbose;
   std::string m_csv_status;
   int m_burnin;
   int m_nsamples;
   int m_num_latent;

   //-- for macau priors
   double m_lambda_beta;
   double m_tol;
   bool m_direct;

   //-- binary classification
   bool m_classify;
   double m_threshold;

public:
   Config()
   {
      //these are default values for config
      //technically - all defaults are specified in CmdSession parser
      //these values are only needed if Config is going to be created outside of parser

      m_restore_prefix = "";
      m_restore_suffix = ".csv";

      m_model_init_type = ModelInitTypes::zero;

      m_save_prefix = "save";
      m_save_suffix = ".csv";
      m_save_freq = 0; // never

      m_random_seed_set = false;
      m_verbose = 1;
      m_csv_status = "";
      m_burnin = 200;
      m_nsamples = 800;
      m_num_latent = 96;

      m_lambda_beta = 10.0;
      m_tol = 1e-6;
      m_direct = false;

      m_classify = false;
   }

public:
   bool validate() const;
   void save(std::string) const;
   void restore(std::string);

public:
   std::shared_ptr<TensorConfig> getTrain() const
   {
      return m_train;
   }

   void setTrain(std::shared_ptr<TensorConfig> value)
   {
      m_train = value;
   }

   std::shared_ptr<TensorConfig> getTest() const
   {
      return m_test;
   }

   void setTest(std::shared_ptr<TensorConfig> value)
   {
      m_test = value;
   }

   std::vector<std::shared_ptr<MatrixConfig> >& getSideInfo()
   {
      return m_sideInfo;
   }

   std::vector<std::vector<std::shared_ptr<TensorConfig> > >& getAuxData()
   {
      return m_auxData;
   }

   std::vector<PriorTypes>& getPriorTypes()
   {
      return m_prior_types;
   }

   std::string getRestorePrefix() const
   {
      return m_restore_prefix;
   }

   void setRestorePrefix(std::string value)
   {
      m_restore_prefix = value;
   }

   std::string getRestoreSuffix() const
   {
      return m_restore_suffix;
   }

   void setRestoreSuffix(std::string value)
   {
      m_restore_suffix = value;
   }

   ModelInitTypes getModelInitType() const
   {
      return m_model_init_type;
   }

   void setModelInitType(ModelInitTypes value)
   {
      m_model_init_type = value;
   }

   std::string getSavePrefix() const
   {
      return m_save_prefix;
   }

   void setSavePrefix(std::string value)
   {
      m_save_prefix = value;
   }

   std::string getSaveSuffix() const
   {
      return m_save_suffix;
   }

   void setSaveSuffix(std::string value)
   {
      m_save_suffix = value;
   }

   int getSaveFreq() const
   {
      return m_save_freq;
   }

   void setSaveFreq(int value)
   {
      m_save_freq = value;
   }

   bool getRandomSeedSet() const
   {
      return m_random_seed_set;
   }

   void setRandomSeedSet(bool value)
   {
      m_random_seed_set = value;
   }

   int getRandomSeed() const
   {
      return m_random_seed;
   }

   void setRandomSeed(int value)
   {
      m_random_seed = value;
   }

   int getVerbose() const
   {
      return m_verbose;
   }

   void setVerbose(int value)
   {
      m_verbose = value;
   }

   std::string getCsvStatus() const
   {
      return m_csv_status;
   }

   void setCsvStatus(std::string value)
   {
      m_csv_status = value;
   }

   int getBurnin() const
   {
      return m_burnin;
   }

   void setBurnin(int value)
   {
      m_burnin = value;
   }

   int getNSamples() const
   {
      return m_nsamples;
   }

   void setNSamples(int value)
   {
      m_nsamples = value;
   }

   int getNumLatent() const
   {
      return m_num_latent;
   }

   void setNumLatent(int value)
   {
      m_num_latent = value;
   }

   double getLambdaBeta() const
   {
      return m_lambda_beta;
   }

   void setLambdaBeta(double value)
   {
      m_lambda_beta = value;
   }

   double getTol() const
   {
      return m_tol;
   }

   void setTol(double value)
   {
      m_tol = value;
   }

   bool getDirect() const
   {
      return m_direct;
   }

   void setDirect(bool value)
   {
      m_direct = value;
   }

   bool getClassify() const
   {
      return m_classify;
   }

   void setClassify(bool value)
   {
      m_classify = value;
   }

   double getThreshold() const
   {
      return m_threshold;
   }

   void setThreshold(double value)
   {
      m_threshold = value;
   }
};

} // end namespace smurff

