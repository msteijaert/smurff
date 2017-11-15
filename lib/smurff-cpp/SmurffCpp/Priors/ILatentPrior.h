#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Sessions/BaseSession.h>
#include <SmurffCpp/Noises/INoiseModel.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Utils/Distribution.h>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/VMatrixIterator.hpp>

namespace smurff {

class ILatentPrior
{
public:
   std::shared_ptr<BaseSession> m_session;
   uint32_t m_mode;
   std::string m_name = "xxxx";

   thread_vector<Eigen::VectorXd> rrs;
   thread_vector<Eigen::MatrixXd> MMs;

protected:
   ILatentPrior(){}

public:
   ILatentPrior(std::shared_ptr<BaseSession> session, uint32_t mode, std::string name = "xxxx");
   virtual ~ILatentPrior() {}
   virtual void init();

   // utility
   std::shared_ptr<const Model> model() const;
   std::shared_ptr<Model> model();

   std::shared_ptr<Data> data() const;
   double predict(const PVec<> &) const;

   std::shared_ptr<INoiseModel> noise();

   std::shared_ptr<Eigen::MatrixXd> U();
   std::shared_ptr<const Eigen::MatrixXd> U() const;

   //return V matrices in the model opposite to mode
   VMatrixIterator<Eigen::MatrixXd> Vbegin();
   
   VMatrixIterator<Eigen::MatrixXd> Vend();

   ConstVMatrixIterator<Eigen::MatrixXd> CVbegin() const;
   
   ConstVMatrixIterator<Eigen::MatrixXd> CVend() const;

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

   virtual void update_prior() = 0;

public:
   void setMode(int value)
   {
      m_mode = value;
   }
};
}
