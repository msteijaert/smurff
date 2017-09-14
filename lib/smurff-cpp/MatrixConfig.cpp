#include "MatrixConfig.h"

#include <cassert>

using namespace smurff;

// TODO: probably remove default constructor
MatrixConfig::MatrixConfig()
   : TensorConfig(true, false, 2, 0, NoiseConfig())
{
   m_dims.push_back(0);
   m_dims.push_back(0);
   m_columns.clear();
   m_values.clear();
}

MatrixConfig::MatrixConfig(int nrow, int ncol, double* values, const NoiseConfig& noiseConfig)
   : TensorConfig(true, false, 2, nrow * ncol, noiseConfig)
{
   m_dims.push_back(nrow);
   m_dims.push_back(ncol);
   m_columns.resize(m_nnz * m_nmodes);
   m_values.resize(m_nnz);

   for (int row = 0; row < nrow; row++)
   {
      for (int col = 0; col < ncol; col++)
      {
         m_columns[nrow * col + row] = row;
         m_columns[nrow * col + row + m_nnz] = col;
      }
   }

   for (int i = 0; i < m_nnz; i++)
   {
      m_values[i] = values[i];
   }
}

MatrixConfig::MatrixConfig(int nrow, int ncol, int nnz, int* rows, int* cols, double* values, const NoiseConfig& noiseConfig)
   : TensorConfig(false, false, 2, nnz, noiseConfig)
{
   m_dims.push_back(nrow);
   m_dims.push_back(ncol);
   m_columns.resize(m_nnz * m_nmodes);
   m_values.resize(m_nnz);

   for (int i = 0; i < m_nnz; i++)
   {
      m_columns[i] = rows[i];
      m_columns[i + m_nnz] = cols[i];
      m_values[i] = values[i];
   }
}

MatrixConfig::MatrixConfig(int nrow, int ncol, int nnz, int* rows, int* cols, const NoiseConfig& noiseConfig)
   : TensorConfig(false, true, 2, nnz, noiseConfig)
{
   m_dims.push_back(nrow);
   m_dims.push_back(ncol);
   m_columns.resize(m_nnz * m_nmodes);
   m_values.clear();

   for (int i = 0; i < m_nnz; i++)
   {
      m_columns[i] = rows[i];
      m_columns[i + m_nnz] = cols[i];
   }
}

MatrixConfig::MatrixConfig(int nrow, int ncol, int nnz, int* columns, double* values, const NoiseConfig& noiseConfig)
   : TensorConfig(columns, 2, values, nnz, std::vector<int>({ nrow, ncol }).data(), noiseConfig)
{
}

// TODO: cache the data
std::vector<int> MatrixConfig::getRows() const
{
   if (m_nnz == 0)
      return std::vector<int>();

   std::vector<int> rows;
   rows.reserve(m_nnz);
   for (int i = 0; i < m_nnz; i++)
      rows.push_back(m_columns[i]);
   return rows;
}

// TODO: cache the data
std::vector<int> MatrixConfig::getCols() const
{
   if (m_nnz == 0)
      return std::vector<int>();

   std::vector<int> cols;
   cols.reserve(m_nnz);
   for (int i = 0; i < m_nnz; i++)
      cols.push_back(m_columns[i + m_nnz]);
   return cols;
}

int MatrixConfig::getNRow() const
{
   return m_dims[0];
}
int MatrixConfig::getNCol() const
{
   return m_dims[1];
}