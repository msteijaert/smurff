#include "MatrixConfig.h"

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/IO/IDataWriter.h>
#include <SmurffCpp/DataMatrices/IDataCreator.h>
#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

//
// Dense double matrix constructos
//

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , const std::vector<double>& values
                          , const NoiseConfig& noiseConfig
                          )
   : MatrixConfig(nrow, ncol, std::shared_ptr<std::vector<double> >(), noiseConfig)
{
   // Wait for nrow and ncol checks pass in constructor in initializer list to prevent wasteful vector copying
   m_values = std::make_shared<std::vector<double> >(values);
}

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::vector<double>&& values
                          , const NoiseConfig& noiseConfig
                          )
   : MatrixConfig(nrow, ncol, std::make_shared<std::vector<double> >(std::move(values)), noiseConfig)
{
}

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::shared_ptr<std::vector<double> > values
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig(true, false, false, 2, nrow * ncol, noiseConfig)
{
   if (nrow == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'nrow' cannot be zero.");
   }

   if (ncol == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'ncol' cannot be zero.");
   }

   m_dims->push_back(nrow);
   m_dims->push_back(ncol);
   m_columns->resize(m_nnz * m_nmodes);
   m_values = values;

   for (std::uint64_t row = 0; row < nrow; row++)
   {
      for (std::uint64_t col = 0; col < ncol; col++)
      {
         m_columns->operator[](nrow * col + row) = row;
         m_columns->operator[](nrow * col + row + m_nnz) = col;
      }
   }
}

//
// Sparse double matrix constructors
//

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , const std::vector<std::uint32_t>& rows
                          , const std::vector<std::uint32_t>& cols
                          , const std::vector<double>& values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig(false, false, isScarce, 2, values.size(), noiseConfig)
{
   if (nrow == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'nrow' cannot be zero.");
   }

   if (ncol == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'ncol' cannot be zero.");
   }

   if (rows.size() != cols.size() || rows.size() != values.size())
   {
      THROWERROR("Cannot create MatrixConfig instance: 'rows', 'cols' and 'values' should all be the same size.");
   }

   m_dims->push_back(nrow);
   m_dims->push_back(ncol);
   m_columns->resize(m_nnz * m_nmodes);
   m_values->resize(m_nnz);

   for (std::uint64_t i = 0; i < m_nnz; i++)
   {
      m_columns->operator[](i) = rows[i];
      m_columns->operator[](i + m_nnz) = cols[i];
      m_values->operator[](i) = values[i];
   }
}

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::vector<std::uint32_t>&& rows
                          , std::vector<std::uint32_t>&& cols
                          , std::vector<double>&& values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : MatrixConfig( nrow
                 , ncol
                 , std::make_shared<std::vector<std::uint32_t> >(std::move(rows))
                 , std::make_shared<std::vector<std::uint32_t> >(std::move(cols))
                 , std::make_shared<std::vector<double> >(std::move(values))
                 , noiseConfig, isScarce
                 )
{
}

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::shared_ptr<std::vector<std::uint32_t> > rows
                          , std::shared_ptr<std::vector<std::uint32_t> > cols
                          , std::shared_ptr<std::vector<double> > values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig(false, false, isScarce, 2, values->size(), noiseConfig)
{
   if (nrow == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'nrow' cannot be zero.");
   }

   if (ncol == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'ncol' cannot be zero.");
   }

   if (rows->size() != cols->size() || rows->size() != values->size())
   {
      THROWERROR("Cannot create MatrixConfig instance: 'rows', 'cols' and 'values' should all be the same size.");
   }

   m_dims->push_back(nrow);
   m_dims->push_back(ncol);
   m_columns->resize(m_nnz * m_nmodes);
   m_values = values;

   m_rows = rows;
   m_cols = cols;

   for (std::uint64_t i = 0; i < m_nnz; i++)
   {
      m_columns->operator[](i) = rows->operator[](i);
      m_columns->operator[](i + m_nnz) = cols->operator[](i);
   }
}

//
// Sparse binary matrix constructors
//

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , const std::vector<std::uint32_t>& rows
                          , const std::vector<std::uint32_t>& cols
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig(false, true, isScarce, 2, rows.size(), noiseConfig)
{
   if (nrow == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'nrow' cannot be zero.");
   }

   if (ncol == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'ncol' cannot be zero.");
   }

   if (rows.size() != cols.size())
   {
      THROWERROR("Cannot create MatrixConfig instance: 'rows' and 'cols' should all be the same size.");
   }

   m_dims->push_back(nrow);
   m_dims->push_back(ncol);
   m_columns->resize(m_nnz * m_nmodes);
   m_values->resize(m_nnz);

   for (std::uint64_t i = 0; i < m_nnz; i++)
   {
      m_columns->operator[](i) = rows[i];
      m_columns->operator[](i + m_nnz) = cols[i];
      m_values->operator[](i) = 1;
   }
}

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::vector<std::uint32_t>&& rows
                          , std::vector<std::uint32_t>&& cols
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : MatrixConfig( nrow
                 , ncol
                 , std::make_shared<std::vector<std::uint32_t> >(std::move(rows))
                 , std::make_shared<std::vector<std::uint32_t> >(std::move(cols))
                 , noiseConfig, isScarce
                 )
{
}

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::shared_ptr<std::vector<std::uint32_t> > rows
                          , std::shared_ptr<std::vector<std::uint32_t> > cols
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig(false, true, isScarce, 2, rows->size(), noiseConfig)
{
   if (nrow == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'nrow' cannot be zero.");
   }

   if (ncol == 0)
   {
      THROWERROR("Cannot create MatrixConfig instance: 'ncol' cannot be zero.");
   }

   if (rows->size() != cols->size())
   {
      THROWERROR("Cannot create MatrixConfig instance: 'rows' and 'cols' should all be the same size.");
   }

   m_dims->push_back(nrow);
   m_dims->push_back(ncol);
   m_columns->resize(m_nnz * m_nmodes);
   m_values->resize(m_nnz);

   m_rows = rows;
   m_cols = cols;

   for (std::uint64_t i = 0; i < m_nnz; i++)
   {
      m_columns->operator[](i) = rows->operator[](i);
      m_columns->operator[](i + m_nnz) = cols->operator[](i);
      m_values->operator[](i) = 1;
   }
}

//
// Constructors for constructing sparse matrix as a tensor
//

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , const std::vector<std::uint32_t>& columns
                          , const std::vector<double>& values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig({ nrow, ncol }, columns, values, noiseConfig, isScarce)
{
}

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::vector<std::uint32_t>&& columns
                          , std::vector<double>&& values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig({ nrow, ncol }, std::move(columns), std::move(values), noiseConfig, isScarce)
{
}

MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::shared_ptr<std::vector<std::uint32_t> > columns
                          , std::shared_ptr<std::vector<double> > values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig(std::make_shared<std::vector<std::uint64_t> >(std::initializer_list<std::uint64_t>({ nrow, ncol })), columns, values, noiseConfig, isScarce)
{
}

//
// Constructors for constructing sparse binary matrix as a tensor
//

MatrixConfig::MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
               const std::vector<std::uint32_t>& columns,
               const NoiseConfig& noiseConfig, bool isScarce)
   : TensorConfig({ nrow, ncol }, columns, noiseConfig, isScarce)
{
}

MatrixConfig::MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
               std::vector<std::uint32_t>&& columns,
               const NoiseConfig& noiseConfig, bool isScarce)
   : TensorConfig({ nrow, ncol }, std::move(columns), noiseConfig, isScarce)
{
}

MatrixConfig::MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
               std::shared_ptr<std::vector<std::uint32_t> > columns,
               const NoiseConfig& noiseConfig, bool isScarce)
   : TensorConfig(std::make_shared<std::vector<std::uint64_t> >(std::initializer_list<std::uint64_t>({ nrow, ncol })), columns, noiseConfig, isScarce)
{
}

// TODO: probably remove default constructor
MatrixConfig::MatrixConfig()
   : TensorConfig(true, false, false, 2, 0, NoiseConfig())
{
   m_dims->push_back(0);
   m_dims->push_back(0);
   m_columns->clear();
   m_values->clear();
}

//
// other methods
//

std::uint64_t MatrixConfig::getNRow() const
{
   return m_dims->operator[](0);
}

std::uint64_t MatrixConfig::getNCol() const
{
   return m_dims->operator[](1);
}

const std::vector<std::uint32_t>& MatrixConfig::getRows() const
{
   return *getRowsPtr();
}

const std::vector<std::uint32_t>& MatrixConfig::getCols() const
{
   return *getColsPtr();
}

std::shared_ptr<std::vector<std::uint32_t> > MatrixConfig::getRowsPtr() const
{
   if (!m_rows)
   {
      m_rows = std::make_shared<std::vector<std::uint32_t> >();
      if (m_nnz != 0)
      {
         m_rows->reserve(m_nnz);
         for (std::uint64_t i = 0; i < m_nnz; i++)
            m_rows->push_back(m_columns->operator[](i));
      }
   }
   return m_rows;
}

std::shared_ptr<std::vector<std::uint32_t> > MatrixConfig::getColsPtr() const
{
   if (!m_cols)
   {
      m_cols = std::make_shared<std::vector<std::uint32_t> >();
      if (m_nnz != 0)
      {
         m_cols->reserve(m_nnz);
         for (std::uint64_t i = 0; i < m_nnz; i++)
            m_cols->push_back(m_columns->operator[](i + m_nnz));
      }
   }
   return m_cols;
}

std::shared_ptr<Data> MatrixConfig::create(std::shared_ptr<IDataCreator> creator) const
{
   //have to use dynamic cast here but only because shared_from_this() can only return base pointer even from child
   return creator->create(std::dynamic_pointer_cast<const MatrixConfig>(shared_from_this()));
}

void MatrixConfig::write(std::shared_ptr<IDataWriter> writer) const
{
   //have to use dynamic cast here but only because shared_from_this() can only return base pointer even from child
   writer->write(std::dynamic_pointer_cast<const MatrixConfig>(shared_from_this()));
}