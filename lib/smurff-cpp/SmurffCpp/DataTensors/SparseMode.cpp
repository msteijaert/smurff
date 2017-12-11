#include "SparseMode.h"

#include <iostream>
#include <sstream>

#include <SmurffCpp/Utils/Error.h>

using namespace Eigen;
using namespace smurff;

SparseMode::SparseMode() 
: m_mode(0) 
{
}

// idx - [nnz x nmodes] matrix of coordinates
// vals - vector of values
// m - index of dimention to fix
// mode_size - size of dimention to fix
SparseMode::SparseMode(const MatrixXui32& idx, const std::vector<double>& vals, std::uint64_t mode, std::uint64_t mode_size) 
{
   if ((size_t)idx.rows() != vals.size()) 
      THROWERROR("Number of rows in 'idx' should equal number of values in 'vals'");

   m_row_ptr.resize(mode_size + 1); // mode_size + 1 because this vector will hold commulative sum of number of elements
   m_values.resize(vals.size()); // resize values vector
   m_indices.resize(idx.rows(), idx.cols() - 1); // reduce number of columns by 1 (exclude fixed dimension)

   m_mode = mode; // save dimension index that is fixed
   std::uint64_t nnz  = idx.rows(); // get nnz from index matrix

   auto rows = idx.col(m_mode); // get column with coordinates for fixed dimension
   
   // compute number of non-zero entries per each element for the mode
   // (compute number of non-zero elements for each coordinate in specific dimension)
   for (std::uint64_t i = 0; i < nnz; i++) 
   {
      if (rows(i) >= mode_size) //index in column should be within dimension size
         THROWERROR("'idx' value is larger than 'mode_size'");
      m_row_ptr[rows(i)]++; //count item with specific index
   }

   // compute commulative sum of number of elements for each coordinate in specific dimension
   for (std::uint64_t row = 0, cumsum = 0; row < mode_size; row++)
   {
      std::uint64_t temp = m_row_ptr[row];
      m_row_ptr[row] = cumsum;
      cumsum += temp;
   }
   m_row_ptr[mode_size] = nnz; //last element should be equal to nnz

   // transform index matrix into index matrix with one reduced/fixed dimension
   for (std::uint64_t i = 0; i < nnz; i++) 
   {
      std::uint32_t row  = rows(i); // coordinate in fixed dimension
      std::uint64_t dest = m_row_ptr[row]; // commulative number of elements for this index

      for (std::uint64_t j = 0, nj = 0; j < (std::uint64_t)idx.cols(); j++) //go through each column in index matrix
      {
         if (j == m_mode) //skip fixed dimension
            continue;

         m_indices(dest, nj) = idx(i, j);
         nj++;
      }

      m_values[dest] = vals[i];
      m_row_ptr[row]++; //update commulative sum vector
   }

   // restore commulative row_ptr
   for (std::uint64_t row = 0, prev = 0; row <= mode_size; row++) 
   {
      std::uint64_t temp = m_row_ptr[row];
      m_row_ptr[row] = prev;
      prev = temp;
   }
}

std::uint64_t SparseMode::getNNZ() const
{ 
   return m_indices.rows(); 
}

std::uint64_t SparseMode::getNPlanes() const
{ 
   return m_row_ptr.size() - 1;
}

std::uint64_t SparseMode::getNCoords() const
{
   return m_indices.cols();
}

const std::vector<double>& SparseMode::getValues() const
{
   return m_values;
}

std::uint64_t SparseMode::getMode() const
{
   return m_mode;
}

std::uint64_t SparseMode::beginPlane(std::uint64_t n) const
{
   return m_row_ptr[n];
}

std::uint64_t SparseMode::endPlane(std::uint64_t n) const
{
   return m_row_ptr[n + 1];
}

const MatrixXui32& SparseMode::getIndices() const
{
   return m_indices;
}