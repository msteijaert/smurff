#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>
#include <signal.h>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/Utils/Distribution.h>

#include <SmurffCpp/Model.h>

using namespace std;
using namespace Eigen;
using namespace smurff;

Model::Model()
   : m_num_latent(-1)
{
}

//num_latent - size of latent dimention
//dims - dimentions of train data
//init_model_type - samples initialization type
void Model::init(int num_latent, const PVec<>& dims, ModelInitTypes model_init_type) 
{
   m_num_latent = num_latent;
   m_dims = std::unique_ptr<PVec<> >(new PVec<>(dims));

   for(size_t i = 0; i < dims.size(); ++i) 
   {
      m_samples.push_back(Eigen::MatrixXd(m_num_latent, dims[i]));
      auto &M = m_samples.back();

      switch(model_init_type)
      {
      case ModelInitTypes::random:
         bmrandn(M);
         break;
      case ModelInitTypes::zero:
         M.setZero();
         break;
      default:
         throw std::runtime_error("Invalid model init type");
      }
   }
}

double Model::dot(const PVec<> &indices) const
{
   Eigen::ArrayXd P = Eigen::ArrayXd::Ones(m_num_latent);
   for(int d = 0; d < nmodes(); ++d) 
      P *= col(d, indices.at(d)).array();
   return P.sum();
}

double Model::predict(const PVec<> &pos) const
{
   return dot(pos);
}

const Eigen::MatrixXd& Model::U(int f) const 
{
   return m_samples.at(f);
}

Eigen::MatrixXd& Model::U(int f) 
{
   return m_samples.at(f);
}

Eigen::MatrixXd& Model::V(int f) 
{
   if(nmodes() != 2)
      throw std::runtime_error("nmodes value is incorrect");
   return m_samples.at((f + 1) % 2);
}

const Eigen::MatrixXd& Model::V(int f) const 
{
   if(nmodes() != 2)
      throw std::runtime_error("nmodes value is incorrect");
   return m_samples.at((f + 1) % 2);
}

Eigen::MatrixXd::ConstColXpr Model::col(int f, int i) const 
{
   return U(f).col(i);
}

int Model::nmodes() const 
{ 
   return m_samples.size(); 
}

int Model::nlatent() const 
{ 
   return m_num_latent; 
}

int Model::nsamples() const 
{ 
   return std::accumulate(m_samples.begin(), m_samples.end(), 0,
      [](const int &a, const Eigen::MatrixXd &b) { return a + b.cols(); }); 
}

const PVec<>& Model::getDims() const
{
   return *m_dims;
}

SubModel Model::full()
{
   return SubModel(*this);
}

void Model::save(std::string prefix, std::string suffix) 
{
   int i = 0;
   for(auto &U : m_samples)
   {
      smurff::matrix_io::eigen::write_matrix(prefix + "-U" + std::to_string(i++) + "-latents" + suffix, U);
   }
}

void Model::restore(std::string prefix, std::string suffix) 
{
   int i = 0;
   for(auto &U : m_samples)
   {
      smurff::matrix_io::eigen::read_matrix(prefix + "-U" + std::to_string(i++) + "-latents" + suffix, U);
   }
}

std::ostream& Model::info(std::ostream &os, std::string indent) const
{
   os << indent << "Num-latents: " << m_num_latent << "\n";
   return os;
}

std::ostream& Model::status(std::ostream &os, std::string indent) const
{
   Eigen::ArrayXd P = Eigen::ArrayXd::Ones(m_num_latent);
   for(int d = 0; d < nmodes(); ++d) 
      P *= U(d).rowwise().norm().array();
   os << indent << "  Latent-wise norm: " << P.transpose() << "\n";
   return os;
}