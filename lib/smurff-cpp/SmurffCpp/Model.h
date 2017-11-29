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

class Model : public std::enable_shared_from_this<Model>
{
private:
   std::vector<std::shared_ptr<Eigen::MatrixXd> > m_samples; //vector of U matrices
   int m_num_latent; //size of latent dimention for U matrices
   std::unique_ptr<PVec<> > m_dims; //dimentions of train data

public:
   Model();

public:
   //initialize U matrices in the model (random/zero)
   void init(int num_latent, const PVec<>& dims, ModelInitTypes model_init_type);

public:
   //dot product of i'th columns in each U matrix
   //indices - vector of column indices
   double dot(const PVec<> &indices) const;

   //same as dot
   double predict(const PVec<> &pos) const;

public:
   //return f'th U matrix in the model where number of matrices is != 2
   std::shared_ptr<const Eigen::MatrixXd> U(int f) const;
   std::shared_ptr<Eigen::MatrixXd> U(int f);

   //return f'th V matrix in the model where number of matrices is == 2
   std::shared_ptr<const Eigen::MatrixXd> V(int f) const;
   std::shared_ptr<Eigen::MatrixXd> V(int f);

   //return i'th column of f'th U matrix in the model
   Eigen::MatrixXd::ConstColXpr col(int f, int i) const;

public:
   //number of dimentions in train data
   int nmodes() const;

   //size of latent dimention
   int nlatent() const;

   //sum of number of columns in each U matrix in the model
   int nsamples() const;

public:
   //vector if dimention sizes of train data
   const PVec<>& getDims() const;

public:
   //returns SubModel proxy class with offset to the first column of each U matrix in the model
   SubModel full();

public:
   // output to file
   void save(std::string prefix, std::string suffix);
   void restore(std::string prefix, std::string suffix);

   std::ostream& info(std::ostream &os, std::string indent) const;
   std::ostream& status(std::ostream &os, std::string indent) const;
};

// SubModel is a proxy class that allows to access i'th column of each U matrix in the model
class SubModel
{
private:
   std::shared_ptr<const Model> m_model;
   PVec<> m_off;
   PVec<> m_dims;

public:
   SubModel(const std::shared_ptr<Model> &m, const PVec<> o, const PVec<> d)
      : m_model(m), m_off(o), m_dims(d) {}

   SubModel(const SubModel &m, const PVec<> o, const PVec<> d)
      : m_model(m.m_model), m_off(o + m.m_off), m_dims(d) {}

   SubModel(const std::shared_ptr<Model> &m)
      : m_model(m), m_off(m->nmodes()), m_dims(m->getDims()) {}

public:
   Eigen::MatrixXd::ConstBlockXpr U(int f) const
   {
      return m_model->U(f)->block(0, m_off.at(f), m_model->nlatent(), m_dims.at(f));
   }

   Eigen::MatrixXd::ConstBlockXpr V(int f) const
   {
      if(nmodes() != 2)
         throw std::runtime_error("nmodes value is incorrect");
         
      return U((f + 1) % 2);
   }

public:
   //dot product of i'th columns in each U matrix
   double dot(const PVec<> &pos) const
   {
      return m_model->dot(m_off + pos);
   }

   //size of latent dimention
   int nlatent() const
   {
      return m_model->nlatent();
   }

   //number of dimentions in train data
   int nmodes() const
   {
      return m_model->nmodes();
   }
};

}; // end namespace smurff