#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/utils.h>
#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

// AGE: I dont like this cross reference between Data and Model. Need to think how we can eliminate it.
class Data;

struct SubModel;

struct Model 
{
private:
   std::vector<Eigen::MatrixXd> m_samples;
   int m_num_latent;
   std::unique_ptr<PVec<> > m_dims;

public:
   Model()
      : m_num_latent(-1)
   {
   }

   void init(int num_latent, const PVec<>& dims, std::string init_model_type);

   //-- access for all
   const Eigen::MatrixXd &U(int f) const 
   {
      return m_samples.at(f);
   }

   Eigen::MatrixXd::ConstColXpr col(int f, int i) const 
   {
      return U(f).col(i);
   }

   Eigen::MatrixXd &U(int f) 
   {
      return m_samples.at(f);
   }

   double dot(const PVec<> &indices) const
   {
      Eigen::ArrayXd P = Eigen::ArrayXd::Ones(m_num_latent);
      for(int d = 0; d < nmodes(); ++d) P *= col(d, indices.at(d)).array();
      return P.sum();
   }

   double predict(const PVec<> &pos, std::shared_ptr<Data> data) const;

   //-- for when nmodes == 2
   Eigen::MatrixXd &V(int f) 
   {
      assert(nmodes() == 2);
      return m_samples.at((f+1)%2);
   }

   const Eigen::MatrixXd &V(int f) const 
   {
      assert(nmodes() == 2);
      return m_samples.at((f+1)%2);
   }

    // basic stuff
   int nmodes() const 
   { 
      return m_samples.size(); 
   }

   int nlatent() const 
   { 
      return m_num_latent; 
   }

   int nsamples() const 
   { 
      return std::accumulate(m_samples.begin(), m_samples.end(), 0,
         [](const int &a, const Eigen::MatrixXd &b) { return a + b.cols(); }); 
   }

   SubModel full();

   //-- output to file
   void save(std::string, std::string);
   void restore(std::string, std::string);
   std::ostream &info(std::ostream &os, std::string indent) const;
   std::ostream &status(std::ostream &os, std::string indent) const;

  public:
   const PVec<>& getDims() const
   {
      return *m_dims;
   }
};

struct SubModel 
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