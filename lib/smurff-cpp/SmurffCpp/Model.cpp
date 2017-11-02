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

//num_latent - size of latent dimention
//dims - dimentions of train data
//init_model_type - samples initialization type
void Model::init(int num_latent, const PVec<>& dims, std::string init_model_type) 
{
   m_num_latent = num_latent;
   m_dims = std::unique_ptr<PVec<> >(new PVec<>(dims));

   for(size_t i = 0; i < dims.size(); ++i) 
   {
      m_samples.push_back(Eigen::MatrixXd(m_num_latent, dims[i]));
      auto &M = m_samples.back();

      if (init_model_type == "random") 
         bmrandn(M);
      else if (init_model_type == "zero") 
         M.setZero();
      else 
         assert(false);
   }
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

std::ostream &Model::info(std::ostream &os, std::string indent) const
{
   os << indent << "Num-latents: " << m_num_latent << "\n";
   return os;
}

std::ostream &Model::status(std::ostream &os, std::string indent) const
{
   Eigen::ArrayXd P = Eigen::ArrayXd::Ones(m_num_latent);
   for(int d = 0; d < nmodes(); ++d) P *= U(d).rowwise().norm().array();
   os << indent << "  Latent-wise norm: " << P.transpose() << "\n";
   return os;
}

SubModel Model::full()
{
   return SubModel(*this);
}

double Model::predict(const PVec<> &pos, std::shared_ptr<Data> data) const
{
   return dot(pos) + data->offset_to_mean(pos);
}