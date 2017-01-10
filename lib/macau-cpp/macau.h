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
      MFactors model;

      bool verbose = true;
      int nsamples = 100;
      int burnin   = 50;
      bool save_model = false;
      std::string save_prefix = "model";

      double rmse_test;

  public:
      Macau(int D = 10) : model(D) {}

      template<class Prior>
      inline Prior& addPrior();

      // noise models
      FixedGaussianNoise &setPrecision(double p);
      AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);
      ProbitNoise &setProbit();

      void setSamples(int burnin, int nsamples);
      void init();
      void run();
      void printStatus(int i, double elapsedi, double samples_per_sec);
      void setVerbose(bool v) { verbose = v; };
      double getRmseTest();
      Eigen::VectorXd getPredictions() { return model.predictions; };
      void saveModel(int isample);
      void saveGlobalParams();
      void setSaveModel(bool save) { save_model = save; };
      void setSavePrefix(std::string pref) { save_prefix = pref; };
      ~Macau();
};

class MacauMPI : public Macau {
  public:
    MacauMPI(int D, int world_rank) : Macau(D), world_rank(world_rank) {}
    void run();
    const int world_rank;
};

template<class Prior>
Prior& Macau::addPrior()
{
    auto pos = priors.size();
    Prior *p = new Prior(model.fac(pos), *noise);
    priors.push_back(std::unique_ptr<ILatentPrior>(p));
    return *p;
}



void sparseFromIJV(Eigen::SparseMatrix<double> & X, int* rows, int* cols, double* values, int N);

#endif /* MACAU_H */
