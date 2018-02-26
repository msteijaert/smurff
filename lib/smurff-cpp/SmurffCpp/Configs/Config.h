#pragma once

#include <string>
#include <vector>
#include <memory>

#include "MatrixConfig.h"

#define TRAIN_NONE "none"
#define TEST_NONE "none"
#define SIDE_INFO_NONE "none"
#define AUX_DATA_NONE "none"

#define PRIOR_NAME_DEFAULT "default"
#define PRIOR_NAME_MACAU "macau"
#define PRIOR_NAME_MACAU_ONE "macauone"
#define PRIOR_NAME_SPIKE_AND_SLAB "spikeandslab"
#define PRIOR_NAME_NORMAL "normal"
#define PRIOR_NAME_NORMALONE "normalone"

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
public:

   //config
   static int BURNIN_DEFAULT_VALUE;
   static int NSAMPLES_DEFAULT_VALUE;
   static int NUM_LATENT_DEFAULT_VALUE;
   static ModelInitTypes INIT_MODEL_DEFAULT_VALUE;
   static char* SAVE_PREFIX_DEFAULT_VALUE;
   static char* SAVE_EXTENSION_DEFAULT_VALUE;
   static int SAVE_FREQ_DEFAULT_VALUE;
   static int VERBOSE_DEFAULT_VALUE;
   static char* STATUS_DEFAULT_VALUE;
   static double LAMBDA_BETA_DEFAULT_VALUE;
   static double TOL_DEFAULT_VALUE;
   static double THRESHOLD_DEFAULT_VALUE;
   static int RANDOM_SEED_DEFAULT_VALUE;

   //noise config
   static NoiseTypes NOISE_TYPE_DEFAULT_VALUE;
   static double PRECISION_DEFAULT_VALUE;
   static double ADAPTIVE_SN_INIT_DEFAULT_VALUE;
   static double ADAPTIVE_SN_MAX_DEFAULT_VALUE;
   static double PROBIT_DEFAULT_VALUE;

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

   //-- init model
   ModelInitTypes m_model_init_type;

   //-- save
   std::string m_save_prefix;
   std::string m_save_extension;
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
   Config();

public:
   bool validate() const;
   void save(std::string) const;
   bool restore(std::string);

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

   const std::vector<std::shared_ptr<MatrixConfig> >& getSideInfo() const
   {
      return m_sideInfo;
   }

   std::vector<std::shared_ptr<MatrixConfig> >& getSideInfo()
   {
      return m_sideInfo;
   }

   const std::vector<std::vector<std::shared_ptr<TensorConfig> > >& getAuxData() const
   {
      return m_auxData;
   }
   
   std::vector<std::vector<std::shared_ptr<TensorConfig> > >& getAuxData()
   {
      return m_auxData;
   }

   const std::vector<PriorTypes>& getPriorTypes() const
   {
      return m_prior_types;
   }

   std::vector<PriorTypes>& getPriorTypes()
   {
      return m_prior_types;
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

   std::string getSaveExtension() const
   {
      return m_save_extension;
   }

   void setSaveExtension(std::string value)
   {
      m_save_extension = value;
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

}

