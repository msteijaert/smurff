#pragma once

#include <vector>
#include <iostream>

#include "TensorConfig.h"
#include "NoiseConfig.h"

namespace smurff
{
   class MatrixConfig : public TensorConfig
   {
   public:
      MatrixConfig();
      MatrixConfig(int nrow, int ncol, double* values, const NoiseConfig& noiseConfig);
      MatrixConfig(int nrow, int ncol, int nnz, int* rows, int* cols, double* values, const NoiseConfig& noiseConfig);
      MatrixConfig(int nrow, int ncol, int nnz, int* rows, int* cols, const NoiseConfig& noiseConfig);
      MatrixConfig(int nrow, int ncol, int nnz, int* columns, double* values, const NoiseConfig& noiseConfig);

   public:
      std::vector<int> getRows() const;
      std::vector<int> getCols() const;

      int getNRow() const;
      int getNCol() const;
   };
}