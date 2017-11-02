#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "MatrixConfig.h"

#define PRIOR_NAME_DEFAULT "default"
#define PRIOR_NAME_MACAU "macau"
#define PRIOR_NAME_MACAU_ONE "macauone"
#define PRIOR_NAME_SPIKE_AND_SLAB "spikeandslab"
#define PRIOR_NAME_NORMAL "normal"
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
   mpi
};

enum class CenterModeTypes : int
{
   CENTER_INVALID = -10,
   CENTER_NONE = -3,
   CENTER_GLOBAL = -2,
   CENTER_VIEW = -1,
   CENTER_COLS = 0,
   CENTER_ROWS = 1
};

enum class ModelInitTypes
{
   random,
   zero
};

PriorTypes stringToPriorType(std::string name);

std::string priorTypeToString(PriorTypes type);

std::string centerModeToString(CenterModeTypes cm);

CenterModeTypes stringToCenterMode(std::string c);

ModelInitTypes stringToModelInitType(std::string name);

std::string modelInitTypeToString(ModelInitTypes type);

struct Config 
{
    //-- train and test
    MatrixConfig train, test;
    CenterModeTypes center_mode_type = CenterModeTypes::CENTER_GLOBAL;

    //-- features
    std::vector<MatrixConfig> row_features;
    std::vector<MatrixConfig> col_features;

    // -- priors
    PriorTypes row_prior_type = PriorTypes::default_prior;
    PriorTypes col_prior_type = PriorTypes::default_prior;

    //-- restore
    std::string restore_prefix = "";
    std::string restore_suffix = ".csv";

    //-- init model
    ModelInitTypes model_init_type = ModelInitTypes::zero;

    //-- save
private:
    std::string save_prefix;

public:
   std::string getSavePrefix() const
   {
      return save_prefix;
   }

   void setSavePrefix(std::string value)
   {
      save_prefix = value;
   }

public:
    std::string save_suffix = ".csv";
    int save_freq           = 0; // never

    //-- general
    bool random_seed_set      = false;
    int random_seed;
    int verbose               = 1;
    std::string csv_status    = "";
    int burnin                = 200;
    int nsamples              = 800;
    int num_latent            = 96;

    //-- for macau priors
    double lambda_beta        = 10.0;
    double tol                = 1e-6;
    bool direct               = false;

    //-- binary classification
    bool classify             = false;
    double threshold;

public:
    Config()
    {
      save_prefix = "save";
    }

public:
    bool validate(bool = true) const;
    void save(std::string) const;
    void restore(std::string);
};

} // end namespace smurff

