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
      mutable std::shared_ptr<std::vector<size_t> > m_rows;
      mutable std::shared_ptr<std::vector<size_t> > m_cols;

   public:
      MatrixConfig( size_t nrow
                  , size_t ncol
                  , const std::vector<double>& values
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( size_t nrow
                  , size_t ncol
                  , const std::vector<size_t>& rows
                  , const std::vector<size_t>& cols
                  , const std::vector<double>& values
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( size_t nrow
                  , size_t ncol
                  , const std::vector<size_t>& rows
                  , const std::vector<size_t>& cols
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( size_t nrow
                  , size_t ncol
                  , const std::vector<size_t>& columns
                  , const std::vector<double>& values
                  , const NoiseConfig& noiseConfig
                  );

      MatrixConfig( size_t nrow
                  , size_t ncol
                  , std::shared_ptr<std::vector<size_t> > columns
                  , std::shared_ptr<std::vector<double> > values
                  , const NoiseConfig& noiseConfig
                  );

   public:
      MatrixConfig();

   public:
      size_t getNRow() const;
      size_t getNCol() const;

      const std::vector<size_t>& getRows() const;
      const std::vector<size_t>& getCols() const;

      std::shared_ptr<std::vector<size_t> > getRowsPtr() const;
      std::shared_ptr<std::vector<size_t> > getColsPtr() const;
   };
}