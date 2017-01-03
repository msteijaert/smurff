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
      MFactors model;

      int nsamples = 100;
      int burnin   = 50;

      double rmse_test  = .0;
      double rmse_train = .0;

      /** BPMF model */
      std::vector< std::unique_ptr<ILatentPrior> > priors;
      bool verbose = true;

      bool save_model = false;
      std::string save_prefix = "model";

  public:
    Macau(int D) : model(D) {}
    Macau() : Macau(10) {}

    void addPrior(std::unique_ptr<ILatentPrior> & prior);
    void setPrecision(double p);
    void setAdaptivePrecision(double sn_init, double sn_max);
    void setProbit();
    void setSamples(int burnin, int nsamples);
    void init();
    void run();
    void printStatus(int i, double elapsedi, double samples_per_sec);
    void setVerbose(bool v) { verbose = v; };
    double getRmseTest();
    Eigen::VectorXd getPredictions() { return model.predictions; };
    Eigen::VectorXd getStds();
    Eigen::MatrixXd getTestData();
    void saveModel(int isample);
    void saveGlobalParams();
    void setSaveModel(bool save) { save_model = save; };
    void setSavePrefix(std::string pref) { save_prefix = pref; };
    ~Macau();
};

void sparseFromIJV(Eigen::SparseMatrix<double> & X, int* rows, int* cols, double* values, int N);

#endif /* MACAU_H */
