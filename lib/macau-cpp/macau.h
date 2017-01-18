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

// try adding num_latent as template parameter to Macau
class Macau  {
  public:
      std::unique_ptr<INoiseModel> noise;
      std::vector< std::unique_ptr<ILatentPrior> > priors;
      Factors &model;

      bool verbose = true;
      int nsamples = 100;
      int burnin   = 50;
      bool save_model = false;
      std::string save_prefix = "model";

      double rmse_test;

  public:
      Macau(Factors &m) : model(m) {}

      template<class Prior, class Model>
      inline Prior& addPrior(Model &model);

      // noise models
      FixedGaussianNoise &setPrecision(double p);
      //AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);
      //ProbitNoise &setProbit();

      void setSamples(int burnin, int nsamples);
      void init();
      void run();
      void printStatus(int i, double elapsedi, double samples_per_sec);
      void setVerbose(bool v) { verbose = v; };
      void saveModel(int isample);
      void setSaveModel(bool save) { save_model = save; };
      void setSavePrefix(std::string pref) { save_prefix = pref; };
      ~Macau();
};

class MacauMPI : public Macau {
  public:
    MacauMPI(Factors &m);
    void run();

    int world_rank;
    int world_size;
};


template<class Prior, class Model>
Prior& Macau::addPrior(Model &model)
{
    auto pos = priors.size();
    Prior *p = new Prior(model, pos, *noise);
    priors.push_back(std::unique_ptr<ILatentPrior>(p));
    return *p;
}




void sparseFromIJV(Eigen::SparseMatrix<double> & X, int* rows, int* cols, double* values, int N);

#endif /* MACAU_H */
