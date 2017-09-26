#pragma once

#include <vector>
#include <iostream>
#include <memory>

#include "TensorConfig.h"
#include "NoiseConfig.h"

namespace smurff
{
   class MatrixConfig : public TensorConfig
   {
   private:
      mutable std::shared_ptr<std::vector<std::uint64_t> > m_rows;
      mutable std::shared_ptr<std::vector<std::uint64_t> > m_cols;

   //
   // Dense double matrix constructos
   //
   public:
      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , const std::vector<double>& values
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , std::vector<double>&& values
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , std::shared_ptr<std::vector<double> > values
                  , const NoiseConfig& noiseConfig
                  );

   //
   // Sparse double matrix constructors
   //
   public:
      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , const std::vector<std::uint64_t>& rows
                  , const std::vector<std::uint64_t>& cols
                  , const std::vector<double>& values
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , std::vector<std::uint64_t>&& rows
                  , std::vector<std::uint64_t>&& cols
                  , std::vector<double>&& values
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , std::shared_ptr<std::vector<std::uint64_t> > rows
                  , std::shared_ptr<std::vector<std::uint64_t> > cols
                  , std::shared_ptr<std::vector<double> > values
                  , const NoiseConfig& noiseConfig
                  );

   //
   // Sparse binary matrix constructors
   //
   public:
      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , const std::vector<std::uint64_t>& rows
                  , const std::vector<std::uint64_t>& cols
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , std::vector<std::uint64_t>&& rows
                  , std::vector<std::uint64_t>&& cols
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , std::shared_ptr<std::vector<std::uint64_t> > rows
                  , std::shared_ptr<std::vector<std::uint64_t> > cols
                  , const NoiseConfig& noiseConfig
                  );

   //
   // Constructors for constructing matrix as a tensor
   //
   public:
      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , const std::vector<std::uint64_t>& columns
                  , const std::vector<double>& values
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , std::vector<std::uint64_t>&& columns
                  , std::vector<double>&& values
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( std::uint64_t nrow
                  , std::uint64_t ncol
                  , std::shared_ptr<std::vector<std::uint64_t> > columns
                  , std::shared_ptr<std::vector<double> > values
                  , const NoiseConfig& noiseConfig
                  );

   public:
      MatrixConfig();

   public:
      std::uint64_t getNRow() const;
      std::uint64_t getNCol() const;

      const std::vector<std::uint64_t>& getRows() const;
      const std::vector<std::uint64_t>& getCols() const;

      std::shared_ptr<std::vector<std::uint64_t> > getRowsPtr() const;
      std::shared_ptr<std::vector<std::uint64_t> > getColsPtr() const;
   };
}