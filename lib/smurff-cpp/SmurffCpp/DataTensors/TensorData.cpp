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
     m_nnz(tc.getNNZ()),
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


void TensorData::init_pre()
{
   throw std::runtime_error("not implemented");
}

double TensorData::sum() const
{
   throw std::runtime_error("not implemented");
}

std::uint64_t TensorData::nmode() const
{
   return m_dims.size();
}

std::uint64_t TensorData::nnz() const
{
   return m_nnz;
}

std::uint64_t TensorData::nna() const
{
   return 0;
}

PVec<> TensorData::dim() const
{
   std::vector<int> pvec_dims;
   for(auto& d : m_dims)
      pvec_dims.push_back(static_cast<int>(d));
   return PVec<>(pvec_dims);
}

double TensorData::train_rmse(const SubModel& model) const
{
   //there was no implementation of train_rmse for tensors in macau
   throw std::runtime_error("not implemented");
}

//d is an index of column in U matrix
//this function selects d'th hyperplane from mode`th SparseMode
//it does j multiplications
//where each multiplication is a cwiseProduct of columns from each V matrix
void TensorData::get_pnm(const SubModel& model, uint32_t mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
   const double alpha = this->noise()->getAlpha();
   std::shared_ptr<SparseMode> sview = Y(mode); //get tensor rotation for mode
   
   auto V0 = model.CVbegin(mode); //get first V matrix
   for (std::uint64_t j = sview->beginPlane(d); j < sview->endPlane(d); j++) //go through hyperplane in tensor rotation
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
   //do not need to cache VV here
}

double TensorData::sumsq(const SubModel& model) const
{
   //there is old implementation down below
   throw std::runtime_error("not implemented");
}

double TensorData::var_total() const
{
   //there is old implementation down below
   throw std::runtime_error("not implemented");
}

std::ostream& TensorData::info(std::ostream& os, std::string indent)
{
   throw std::runtime_error("not implemented");
}

//macau
/*
double sumsq(TensorData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
{
   double sumsq = 0.0;
   double mean_value = data.mean_value;

   auto& sparseMode = (*data.Y)[0];
   auto& U = samples[0];

   const int nmodes = samples.size();
   const int num_latents = U->rows();

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
   for (int n = 0; n < data.dims(0); n++) {
      Eigen::VectorXd u = U->col(n);
      for (int j = sparseMode->row_ptr(n);
               j < sparseMode->row_ptr(n + 1);
               j++)
      {
         VectorXi idx = sparseMode->indices.row(j);
         // computing prediction from tensor
         double Yhat = mean_value;
         for (int d = 0; d < num_latents; d++) {
         double tmp = u(d);

         for (int m = 1; m < nmodes; m++) {
            tmp *= (*samples[m])(d, idx(m - 1));
         }
         Yhat += tmp;
         }
         sumsq += square(Yhat - sparseMode->values(j));
      }

   }

   return sumsq;
}
*/

//macau
/*
double var_total(TensorData & data)
{
   double se = 0.0;
   double mean_value = data.mean_value;

   auto& sparseMode   = (*data.Y)[0];
   VectorXd & values  = sparseMode->values;

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
   for (int i = 0; i < values.size(); i++) {
      se += square(values(i) - mean_value);
   }
   var_total = se / values.size();
   if (var_total <= 0.0 || std::isnan(var_total)) {
      var_total = 1.0;
   }

   return var_total;
}
*/