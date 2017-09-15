#include "MatrixConfig.h"

#include <cassert>

using namespace smurff;

MatrixConfig::MatrixConfig( size_t nrow
                          , size_t ncol
                          , const std::vector<double>& values
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig(true, false, 2, nrow * ncol, noiseConfig)
{
   if (nrow == 0)
      throw std::runtime_error("Cannot create MatrixConfig instance: 'nrow' cannot be zero.");

   if (ncol == 0)
      throw std::runtime_error("Cannot create MatrixConfig instance: 'ncol' cannot be zero.");

   m_dims->push_back(nrow);
   m_dims->push_back(ncol);
   m_columns->resize(m_nnz * m_nmodes);
   m_values->resize(m_nnz);

   for (size_t row = 0; row < nrow; row++)
   {
      for (size_t col = 0; col < ncol; col++)
      {
         m_columns->operator[](nrow * col + row) = row;
         m_columns->operator[](nrow * col + row + m_nnz) = col;
      }
   }

   std::copy(values.begin(), values.end(), m_values->begin());
}

MatrixConfig::MatrixConfig( size_t nrow
                          , size_t ncol
                          , const std::vector<size_t>& rows
                          , const std::vector<size_t>& cols
                          , const std::vector<double>& values
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig(false, false, 2, values.size(), noiseConfig)
{
   if (nrow == 0)
      throw std::runtime_error("Cannot create MatrixConfig instance: 'nrow' cannot be zero.");

   if (ncol == 0)
      throw std::runtime_error("Cannot create MatrixConfig instance: 'ncol' cannot be zero.");

   if (rows.size() != cols.size() || rows.size() != values.size())
      throw std::runtime_error("Cannot create MatrixConfig instance: 'rows', 'cols' and 'values' should all be the same size.");

   m_dims->push_back(nrow);
   m_dims->push_back(ncol);
   m_columns->resize(m_nnz * m_nmodes);
   m_values->resize(m_nnz);

   for (size_t i = 0; i < m_nnz; i++)
   {
      m_columns->operator[](i) = rows[i];
      m_columns->operator[](i + m_nnz) = cols[i];
      m_values->operator[](i) = values[i];
   }
}

MatrixConfig::MatrixConfig( size_t nrow
                          , size_t ncol
                          , const std::vector<size_t>& rows
                          , const std::vector<size_t>& cols
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig(false, true, 2, rows.size(), noiseConfig)
{
   if (nrow == 0)
      throw std::runtime_error("Cannot create MatrixConfig instance: 'nrow' cannot be zero.");

   if (ncol == 0)
      throw std::runtime_error("Cannot create MatrixConfig instance: 'ncol' cannot be zero.");

   if (rows.size() != cols.size())
      throw std::runtime_error("Cannot create MatrixConfig instance: 'rows' and 'cols' should all be the same size.");

   m_dims->push_back(nrow);
   m_dims->push_back(ncol);
   m_columns->resize(m_nnz * m_nmodes);
   m_values->clear();

   for (size_t i = 0; i < m_nnz; i++)
   {
      m_columns->operator[](i) = rows[i];
      m_columns->operator[](i + m_nnz) = cols[i];
   }
}

MatrixConfig::MatrixConfig( size_t nrow
                          , size_t ncol
                          , const std::vector<size_t>& columns
                          , const std::vector<double>& values
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig({ nrow, ncol }, columns, values, noiseConfig)
{
}

MatrixConfig::MatrixConfig( size_t nrow
                          , size_t ncol
                          , std::shared_ptr<std::vector<size_t> > columns
                          , std::shared_ptr<std::vector<double> > values
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig(std::make_shared<std::vector<size_t> >(std::initializer_list<size_t>({ nrow, ncol })), columns, values, noiseConfig)
{
}

// TODO: probably remove default constructor
MatrixConfig::MatrixConfig()
   : TensorConfig(true, false, 2, 0, NoiseConfig())
{
   m_dims->push_back(0);
   m_dims->push_back(0);
   m_columns->clear();
   m_values->clear();
}

size_t MatrixConfig::getNRow() const
{
   return m_dims->operator[](0);
}
size_t MatrixConfig::getNCol() const
{
   return m_dims->operator[](1);
}

const std::vector<size_t>& MatrixConfig::getRows() const
{
   return *getRowsPtr();
}

const std::vector<size_t>& MatrixConfig::getCols() const
{
   return *getColsPtr();
}

std::shared_ptr<std::vector<size_t> > MatrixConfig::getRowsPtr() const
{
   if (!m_rows)
   {
      m_rows = std::make_shared<std::vector<size_t> >();
      if (m_nnz != 0)
      {
         m_rows->reserve(m_nnz);
         for (size_t i = 0; i < m_nnz; i++)
            m_rows->push_back(m_columns->operator[](i));
      }
   }
   return m_rows;
}

std::shared_ptr<std::vector<size_t> > MatrixConfig::getColsPtr() const
{
   if (!m_cols)
   {
      m_cols = std::make_shared<std::vector<size_t> >();
      if (m_nnz != 0)
      {
         m_cols->reserve(m_nnz);
         for (size_t i = 0; i < m_nnz; i++)
            m_cols->push_back(m_columns->operator[](i + m_nnz));
      }
   }
   return m_cols;
}