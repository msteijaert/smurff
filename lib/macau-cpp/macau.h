#ifndef MACAU_H
#define MACAU_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <memory>

#include "model.h"
#include "latentprior.h"
#include "noisemodels.h"


namespace Macau {

class ILatentPrior;

class BaseSession  {
   public:
      std::unique_ptr<INoiseModel>                noise;
      std::vector< std::unique_ptr<ILatentPrior>> priors;
      std::unique_ptr<Factors>                    model;
    
      //-- add model
      template<class Model>
      Model         &addModel(int num_latent);
      SparseMF      &sparseModel(int num_latent);
      DenseDenseMF  &denseDenseModel(int num_latent);
      SparseDenseMF &sparseDenseModel(int num_latent);

      //-- add priors
      template<class Prior>
      inline Prior& addPrior();

      // set noise models
      FixedGaussianNoise &setPrecision(double p);
      AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);

      void init();
      void step();

      virtual std::ostream &printInitStatus(std::ostream &, std::string indent);

      std::string name;
};

// try adding num_latent as template parameter to Session
class Session : public BaseSession {
  public:
      bool        verbose     = true;
      int         nsamples    = 100;
      int         burnin      = 50;
      bool        save_model  = false;
      std::string save_prefix = "model";

      // while running
      int         iter;

      // c'tor
      Session() { name = "MacauSession"; }

      //-- set params
      void setSamples(int burnin, int nsamples);
      void setVerbose(bool v) { verbose = v; };
      void setSaveModel(bool save) { save_model = save; };
      void setSavePrefix(std::string pref) { save_prefix = pref; };
      void setFromArgs(int argc, char** argv, bool print);

      // execution of the sampler
      void init();
      void run();
      void step();

      std::ostream &printInitStatus(std::ostream &, std::string indent) override;

   private:
      void saveModel(int isample);
      void printStatus(double elapsedi);
};

class MPISession : public Session {
  public:
    MPISession() { name = "MPISession"; }
      
    void run();
    std::ostream &printInitStatus(std::ostream &os, std::string indent) override;

    int world_rank;
    int world_size;

};


class PythonSession : public Session {
  public:
    PythonSession() { name = "PythonSession"; }

    void run();

  private:
    static void intHandler(int); 
    static volatile bool keepRunning;

};

template<class Prior>
Prior& BaseSession::addPrior()
{
    auto pos = priors.size();
    Prior *p = new Prior(*this, pos);
    priors.push_back(std::unique_ptr<ILatentPrior>(p));
    return *p;
}

}

#endif /* MACAU_H */
