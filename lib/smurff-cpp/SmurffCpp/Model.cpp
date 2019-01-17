#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>
#include <signal.h>

#include <Eigen/Sparse>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/Utils/Distribution.h>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/VMatrixIterator.hpp>
#include <SmurffCpp/ConstVMatrixIterator.hpp>
#include <SmurffCpp/VMatrixExprIterator.hpp>
#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StepFile.h>

#include <SmurffCpp/IO/GenericIO.h>

using namespace std;
using namespace smurff;


Model::Model()
   : m_num_latent(-1), m_dims(0)
{
}

//num_latent - size of latent dimention
//dims - dimentions of train data
//init_model_type - samples initialization type
void Model::init(int num_latent, const PVec<>& dims, ModelInitTypes model_init_type, bool save_model)
{
   m_num_latent = num_latent;
   m_dims = dims;
   m_save_model = save_model;

   for(size_t i = 0; i < dims.size(); ++i)
   {
      std::shared_ptr<Eigen::MatrixXd> mat(new Eigen::MatrixXd(m_num_latent, dims[i]));

      switch(model_init_type)
      {
      case ModelInitTypes::random:
         bmrandn(*mat);
         break;
      case ModelInitTypes::zero:
         mat->setZero();
         break;
      default:
         {
            THROWERROR("Invalid model init type");
         }
      }

      m_factors.push_back(mat);
   }

   m_link_matrices.resize(nmodes());

   Pcache.init(Eigen::ArrayXd::Ones(m_num_latent));
}

void Model::setLinkMatrix(int mode, std::shared_ptr<Eigen::MatrixXd> link_matrix)
{
   m_link_matrices.at(mode) = link_matrix;
}

double Model::predict(const PVec<> &pos) const
{
   if (nmodes() == 2)
   {
      return col(0, pos[0]).dot(col(1, pos[1]));
   }

   auto &P = Pcache.local();
   P.setOnes();
   for(uint32_t d = 0; d < nmodes(); ++d)
      P *= col(d, pos.at(d)).array();
   return P.sum();
}

const Eigen::MatrixXd &Model::U(uint32_t f) const
{
   return *m_factors.at(f);
}

Eigen::MatrixXd &Model::U(uint32_t f)
{
   return *m_factors[f];
}

VMatrixIterator<Eigen::MatrixXd> Model::Vbegin(std::uint32_t mode)
{
   return VMatrixIterator<Eigen::MatrixXd>(shared_from_this(), mode, 0);
}

VMatrixIterator<Eigen::MatrixXd> Model::Vend()
{
   return VMatrixIterator<Eigen::MatrixXd>(m_factors.size());
}

ConstVMatrixIterator<Eigen::MatrixXd> Model::CVbegin(std::uint32_t mode) const
{
   return ConstVMatrixIterator<Eigen::MatrixXd>(this, mode, 0);
}

ConstVMatrixIterator<Eigen::MatrixXd> Model::CVend() const
{
   return ConstVMatrixIterator<Eigen::MatrixXd>(m_factors.size());
}

Eigen::MatrixXd::ConstColXpr Model::col(int f, int i) const
{
   return U(f).col(i);
}

std::uint64_t Model::nmodes() const
{
   return m_factors.size();
}

int Model::nlatent() const
{
   return m_num_latent;
}

int Model::nsamples() const
{
   return std::accumulate(m_factors.begin(), m_factors.end(), 0,
      [](const int &a, const std::shared_ptr<Eigen::MatrixXd> &b) { return a + b->cols(); });
}

const PVec<>& Model::getDims() const
{
   return m_dims;
}

SubModel Model::full()
{
   return SubModel(*this);
}

void Model::save(std::shared_ptr<const StepFile> sf) const
{
   std::uint64_t i = 0;
   for (auto U : m_factors)
   {
      auto path = sf->makeModelFileName(i++);
      smurff::matrix_io::eigen::write_matrix(path, *U);
   }
}

void Model::restore(std::shared_ptr<const StepFile> sf)
{
   unsigned nmodes = sf->getNModes();
   m_factors.clear();
   m_dims = PVec<>(nmodes);
   
   for(std::uint64_t i = 0; i<nmodes; ++i)
   {
      auto U = std::make_shared<Eigen::MatrixXd>();
      std::string path = sf->getModelFileName(i);
      THROWERROR_FILE_NOT_EXIST(path);
      smurff::matrix_io::eigen::read_matrix(path, *U);
      m_dims.at(i) = U->cols();
      m_num_latent = U->rows();
      m_factors.push_back(U);
   }

   m_link_matrices.resize(nmodes);
   Pcache.init(Eigen::ArrayXd::Ones(m_num_latent));
}

std::ostream& Model::info(std::ostream &os, std::string indent) const
{
   os << indent << "Num-latents: " << m_num_latent << std::endl;
   return os;
}

std::ostream& Model::status(std::ostream &os, std::string indent) const
{
   os << indent << "  Umean: " << std::endl;
   for(std::uint64_t d = 0; d < nmodes(); ++d)
      os << indent << "    U(" << d << ").colwise().mean: "
         << U(d).rowwise().mean().transpose()
         << std::endl;

   return os;
}

Eigen::MatrixXd::ConstBlockXpr SubModel::U(int f) const
{
   const Eigen::MatrixXd &u = m_model.U(f); //force const
   return u.block(0, m_off.at(f), m_model.nlatent(), m_dims.at(f));
}

ConstVMatrixExprIterator<Eigen::MatrixXd::ConstBlockXpr> SubModel::CVbegin(std::uint32_t mode) const
{
   return ConstVMatrixExprIterator<Eigen::MatrixXd::ConstBlockXpr>(&m_model, m_off, m_dims, mode, 0);
}

ConstVMatrixExprIterator<Eigen::MatrixXd::ConstBlockXpr> SubModel::CVend() const
{
   return ConstVMatrixExprIterator<Eigen::MatrixXd::ConstBlockXpr>(m_model.nmodes());
}
