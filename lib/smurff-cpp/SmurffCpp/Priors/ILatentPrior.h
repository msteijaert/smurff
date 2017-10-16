#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Sessions/BaseSession.h>
#include <SmurffCpp/Noises/INoiseModel.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Utils/Distribution.h>

#include <SmurffCpp/model.h>

namespace smurff {

//this class now has:
//session, mode, name, rrs, MMs
//there were previously no fields

//also why remove update_prior method ?

//everything else in the class were wrappers that are not needed anymore

class ILatentPrior
{
public:
   BaseSession &session;
   int mode;
   std::string name = "xxxx";

   thread_vector<Eigen::VectorXd> rrs;
   thread_vector<Eigen::MatrixXd> MMs;

public:
   // c-tor
   ILatentPrior(BaseSession &s, int m, std::string name = "xxxx");
   virtual ~ILatentPrior() {}
   virtual void init();

   // utility
   Model &model() const;
   Data  &data() const;
   double predict(const PVec<> &) const;
   INoiseModel &noise();
   Eigen::MatrixXd &U();
   Eigen::MatrixXd &V();
   int num_latent() const;
   int num_cols() const;

   virtual void save(std::string prefix, std::string suffix) = 0;
   virtual void restore(std::string prefix, std::string suffix) = 0;
   virtual std::ostream &info(std::ostream &os, std::string indent);
   virtual std::ostream &status(std::ostream &os, std::string indent) const = 0;

   // work
   virtual bool run_slave(); // returns true if some work happened...

   virtual void sample_latents();
   virtual void sample_latent(int n) = 0;
};
}
