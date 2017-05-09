#ifndef MACAU_H
#define MACAU_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <memory>

#include "result.h"
#include "model.h"
#include "latentprior.h"
#include "noisemodels.h"


namespace Macau {

class ILatentPrior;

class BaseSession  {
   public:
      //-- data members
      Model                                       model;
      std::unique_ptr<Data>                       data;
      Result                                      pred;
      std::unique_ptr<INoiseModel>                noise;
      std::vector< std::unique_ptr<ILatentPrior>> priors;
    
      //-- add data
      //void setData(const Eigen::SparseMatrix<double> &Y, bool );
      //void sparseBinaryModel(int num_latent);
      //void addData(int num_latent);
      //void sparseDenseModel(int num_latent);

      //-- add priors
      template<class Prior>
      inline Prior& addPrior();

      // set noise models
      FixedGaussianNoise &setPrecision(double p);
      AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);

      void init();
      virtual void step();

      virtual std::ostream &info(std::ostream &, std::string indent);
      void save(std::string prefix, std::string suffix);
      void restore(std::string prefix, std::string suffix);

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

      std::ostream &info(std::ostream &, std::string indent) override;

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
    std::ostream &info(std::ostream &os, std::string indent) override;

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

} // end namespace Macau

#endif /* MACAU_H */
