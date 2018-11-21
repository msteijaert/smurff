#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Noises/INoiseModel.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/ThreadVector.hpp>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/VMatrixIterator.hpp>
#include <SmurffCpp/ConstVMatrixIterator.hpp>

namespace smurff {

class StepFile;

class ILatentPrior
{
public:
   std::shared_ptr<Session> m_session;
   std::uint32_t m_mode;
   std::string m_name = "xxxx";

   smurff::thread_vector<Eigen::VectorXf> rrs;
   smurff::thread_vector<Eigen::MatrixXf> MMs;

protected:
   ILatentPrior(){}

public:
   ILatentPrior(std::shared_ptr<Session> session, uint32_t mode, std::string name = "xxxx");
   virtual ~ILatentPrior() {}
   virtual void init();

   // utility
   const Model& model() const;
   Model& model();

   Data& data() const;
   float predict(const PVec<> &) const;

   INoiseModel& noise();

   Eigen::MatrixXf &U();
   const Eigen::MatrixXf &U() const;

   //return V matrices in the model opposite to mode
   VMatrixIterator<Eigen::MatrixXf> Vbegin();
   
   VMatrixIterator<Eigen::MatrixXf> Vend();

   ConstVMatrixIterator<Eigen::MatrixXf> CVbegin() const;
   
   ConstVMatrixIterator<Eigen::MatrixXf> CVend() const;

   int num_latent() const;
   int num_item() const;

   const Eigen::VectorXf& getUsum() { return Usum; } 
   const Eigen::MatrixXf& getUUsum()  { return UUsum; }

   virtual bool save(std::shared_ptr<const StepFile> sf) const;
   virtual void restore(std::shared_ptr<const StepFile> sf);
   virtual std::ostream &info(std::ostream &os, std::string indent);
   virtual std::ostream &status(std::ostream &os, std::string indent) const = 0;

   // work
   virtual bool run_slave(); // returns true if some work happened...

   virtual void sample_latents();
   virtual void sample_latent(int n) = 0;

   virtual void update_prior() = 0;

private:
   void init_Usum();
   Eigen::VectorXf Usum;
   Eigen::MatrixXf UUsum;

public:
   void setMode(std::uint32_t value)
   {
      m_mode = value;
   }

public:
   std::uint32_t getMode() const
   {
      return m_mode;
   }
};
}
