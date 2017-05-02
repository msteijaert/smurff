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
      //-- data members
      std::unique_ptr<INoiseModel>                noise;
      std::vector< std::unique_ptr<ILatentPrior>> priors;
      std::unique_ptr<Model>                      model;
      Result                                      pred;
    
      //-- add model
      template<class Model>
      Model         &addModel(int num_latent);
      SparseMF      &sparseModel(int num_latent);
      SparseBinaryMF&sparseBinaryModel(int num_latent);
      DenseDenseMF  &denseDenseModel(int num_latent);
      SparseDenseMF &sparseDenseModel(int num_latent);

      //-- add priors
      template<class Prior>
      inline Prior& addPrior();

      // set noise models
      FixedGaussianNoise &setPrecision(double p);
      AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);

      void init();
      virtual void step();

      virtual std::ostream &printInitStatus(std::ostream &, std::string indent);
      void save(std::string prefix, std::string suffix);

      std::string name;

   protected:
      bool is_init = false;
};

class Session : public BaseSession {
  public:
      Config      config;
      int         iter;

      // c'tor
      Session() { name = "MacauSession"; }

      //-- set params
      void setFromConfig(const Config &);

      // execution of the sampler
      void init();
      void run();
      void step() override;

      std::ostream &printInitStatus(std::ostream &, std::string indent) override;

   private:
      void save(int isample);
      void printStatus(double elapsedi);
};

class CmdSession :  public Session {
    public:
        void setFromArgs(int argc, char** argv);
};

class MPISession : public CmdSession {
  public:
    MPISession() { name = "MPISession"; }
      
    void run();
    std::ostream &printInitStatus(std::ostream &os, std::string indent) override;

    int world_rank;
    int world_size;

};


class PythonSession : public Session {
  public:
    PythonSession() {
        name = "PythonSession"; 
        keepRunning = true;
    }

    void step() override;

  private:
    static void intHandler(int); 
    static bool keepRunning; 

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
