#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/utils.h>
#include <SmurffCpp/Utils/PVec.hpp>

#include <SmurffCpp/Configs/Config.h>

namespace smurff {

// AGE: I dont like this cross reference between Data and Model. Need to think how we can eliminate it.
class Data;

class SubModel;

class Model 
{
private:
   std::vector<Eigen::MatrixXd> m_samples;
   int m_num_latent;
   std::unique_ptr<PVec<> > m_dims;

public:
   Model();

public:
   void init(int num_latent, const PVec<>& dims, ModelInitTypes model_init_type);
 
public:
   double dot(const PVec<> &indices) const;

   double predict(const PVec<> &pos, std::shared_ptr<Data> data) const;

public:
   // access for all
   const Eigen::MatrixXd &U(int f) const;
   Eigen::MatrixXd &U(int f);

   // for when nmodes == 2
   Eigen::MatrixXd &V(int f);
   const Eigen::MatrixXd &V(int f) const;
   
   // access for all
   Eigen::MatrixXd::ConstColXpr col(int f, int i) const;

public:
    // basic stuff
   int nmodes() const;
   int nlatent() const;
   int nsamples() const;

public:
   const PVec<>& getDims() const;
   
public:
   SubModel full();

public:
   // output to file
   void save(std::string prefix, std::string suffix);
   void restore(std::string prefix, std::string suffix);

   std::ostream& info(std::ostream &os, std::string indent) const;
   std::ostream& status(std::ostream &os, std::string indent) const;
};

class SubModel 
{
private:
   const Model& m_model;
   PVec<> m_off;
   PVec<> m_dims;

public:
   SubModel(const Model &m, const PVec<> o, const PVec<> d)
      : m_model(m), m_off(o), m_dims(d) {}

   SubModel(const SubModel &m, const PVec<> o, const PVec<> d)
      : m_model(m.m_model), m_off(o + m.m_off), m_dims(d) {}

   SubModel(const Model &m) 
      : m_model(m), m_off(m.nmodes()), m_dims(m.getDims()) {}

   Eigen::MatrixXd::ConstBlockXpr U(int f) const 
   {
      return m_model.U(f).block(0, m_off.at(f), m_model.nlatent(), m_dims.at(f));
   }

public:
   Eigen::MatrixXd::ConstBlockXpr V(int f) const 
   {
      assert(nmodes() == 2);
      return U((f+1)%2);
   }

   double dot(const PVec<> &pos) const  
   {
      return m_model.dot(m_off + pos);
   }

   int nlatent() const 
   { 
      return m_model.nlatent(); 
   }

   int nmodes() const 
   { 
      return m_model.nmodes(); 
   }
};

}; // end namespace smurff