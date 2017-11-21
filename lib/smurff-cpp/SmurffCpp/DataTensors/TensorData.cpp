#include "TensorData.h"

#include <iostream>
#include <sstream>

#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

using namespace Eigen;
using namespace smurff;

//convert array of coordinates to [nnz x nmodes] matrix
MatrixXui32 toMatrixNew(const std::vector<std::uint32_t>& columns, std::uint64_t nnz, std::uint64_t nmodes) 
{
   MatrixXui32 idx(nnz, nmodes);
   for (std::uint64_t row = 0; row < nnz; row++) 
   {
      for (std::uint64_t col = 0; col < nmodes; col++) 
      {
         idx(row, col) = columns[col * nnz + row];
      }
   }
   return idx;
}

TensorData::TensorData(const smurff::TensorConfig& tc) 
   : m_dims(tc.getDims()),
     m_Y(std::make_shared<std::vector<std::shared_ptr<SparseMode> > >())
{
   //combine coordinates into [nnz x nmodes] matrix
   MatrixXui32 idx = toMatrixNew(tc.getColumns(), tc.getNNZ(), tc.getNModes());

   for (std::uint64_t mode = 0; mode < tc.getNModes(); mode++) 
   {
      m_Y->push_back(std::make_shared<SparseMode>(idx, tc.getValues(), mode, m_dims[mode]));
   }
}

std::shared_ptr<SparseMode> TensorData::Y(std::uint64_t mode) const
{
   return m_Y->operator[](mode);
}

std::uint64_t TensorData::getNModes() const
{
   return m_dims.size();
}


void TensorData::init_pre()
{
   throw std::runtime_error("not implemented");
}

double TensorData::sum() const
{
   throw std::runtime_error("not implemented");
}

int TensorData::nmode() const
{
   throw std::runtime_error("not implemented");
}

int TensorData::nnz() const
{
   throw std::runtime_error("not implemented");
}

int TensorData::nna() const
{
   throw std::runtime_error("not implemented");
}

PVec<> TensorData::dim() const
{
   throw std::runtime_error("not implemented");
}

double TensorData::train_rmse(const SubModel& model) const
{
   throw std::runtime_error("not implemented");
}

void TensorData::get_pnm(const SubModel& model, uint32_t mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
   const double alpha = this->noise()->getAlpha();
   std::shared_ptr<SparseMode> sview = Y(mode); //get tensor rotation for mode
   auto V0 = model.CVbegin(mode); //get first V matrix

   for (std::uint64_t j = sview->beginMode(d); j < sview->endMode(d); j++) //go through hyperplane in tensor rotation
   {
      VectorXd col = (*V0).col(sview->getIndices()(j, 0)); //create a copy of m'th column from V (m = 0)
      auto V = model.CVbegin(mode); //get V matrices for mode      
      for (std::uint64_t m = 1; m < sview->getNCoords(); m++) //go through each coordinate of value
      {
         ++V; //inc iterator prior to access since we are starting from m = 1
         col.noalias() = col.cwiseProduct((*V).col(sview->getIndices()(j, m))); //multiply by m'th column from V
      }
      
      MM.triangularView<Eigen::Lower>() += alpha * col * col.transpose(); // MM = MM + (col * colT) * alpha (where col = product of columns in each V)
      rr.noalias() += col * sview->getValues()[j] * alpha; // rr = rr + (col * value) * alpha (where value = j'th value of Y)
   }
}

void TensorData::update_pnm(const SubModel& model, uint32_t mode)
{
   
}

double TensorData::sumsq(const SubModel& model) const
{
   throw std::runtime_error("not implemented");
}

double TensorData::var_total() const
{
   throw std::runtime_error("not implemented");
}

std::ostream& TensorData::info(std::ostream& os, std::string indent)
{
   throw std::runtime_error("not implemented");
}

double TensorData::compute_mode_mean_mn(int mode, int pos)
{
   throw std::runtime_error("not implemented");
}

void TensorData::center(double global_mean)
{
   throw std::runtime_error("not implemented");
}

double TensorData::offset_to_mean(const PVec<>& pos) const
{
   throw std::runtime_error("not implemented");
}