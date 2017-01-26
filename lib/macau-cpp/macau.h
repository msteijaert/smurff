#ifndef MACAU_H
#define MACAU_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <memory>

#include "model.h"
#include "latentprior.h"
#include "noisemodels.h"

int get_num_omp_threads();

class ILatentPrior;

// try adding num_latent as template parameter to Macau
class Macau  {
  public:
      std::unique_ptr<INoiseModel>                noise;
      std::vector< std::unique_ptr<ILatentPrior>> priors;
      std::unique_ptr<Factors>                    model;

      bool        verbose     = true;
      int         nsamples    = 100;
      int         burnin      = 50;
      int         num_latent  = 32;
      bool        save_model  = false;
      std::string save_prefix = "model";

      // while running
      int         iter;

  public:
      //-- set params
      void setSamples(int burnin, int nsamples);
      void setVerbose(bool v) { verbose = v; };
      void setSaveModel(bool save) { save_model = save; };
      void setSavePrefix(std::string pref) { save_prefix = pref; };

      //-- add model
      SparseMF &sparseModel(int num_latent);
      DenseMF  &denseModel(int num_latent);

      //-- add priors
      template<class Prior>
      inline Prior& addPrior();

      // set noise models
      FixedGaussianNoise &setPrecision(double p);
      AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);

      void setFromArgs(int argc, char** argv, bool print);

      // execution of the sampler
      void init();
      void run();
      void step();

   private:
      void saveModel(int isample);
      void printStatus(double elapsedi);
};

class MPIMacau : public Macau {
  public:
    MPIMacau();
      
    void run();

    int world_rank;
    int world_size;
};


class PythonMacau : public Macau {
  public:
    void run();

  private:
    static void intHandler(int); 
    static volatile bool keepRunning;

};

template<class Prior>
Prior& Macau::addPrior()
{
    auto pos = priors.size();
    Prior *p = new Prior(*model, pos, *noise);
    priors.push_back(std::unique_ptr<ILatentPrior>(p));
    return *p;
}

void sparseFromIJV(Eigen::SparseMatrix<double> & X, int* rows, int* cols, double* values, int N);

#endif /* MACAU_H */
