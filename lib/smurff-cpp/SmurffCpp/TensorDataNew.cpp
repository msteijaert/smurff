#include "TensorDataNew.h"

#include <iostream>
#include <sstream>

using namespace Eigen;

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

TensorDataNew::TensorDataNew(const smurff::TensorConfig& tc) 
   : m_dims(tc.getDims()),
     m_Y(std::make_shared<std::vector<std::shared_ptr<SparseModeNew> > >())
{
   //combine coordinates into [nnz x nmodes] matrix
   MatrixXui32 idx = toMatrixNew(tc.getColumns(), tc.getNNZ(), tc.getNModes());

   for (std::uint64_t mode = 0; mode < tc.getNModes(); mode++) 
   {
      m_Y->push_back(std::make_shared<SparseModeNew>(idx, tc.getValues(), mode, m_dims[mode]));
   }
}

std::shared_ptr<SparseModeNew> TensorDataNew::Y(std::uint64_t mode) const
{
   return m_Y->operator[](mode);
}

std::uint64_t TensorDataNew::getNModes() const
{
   return m_dims.size();
}