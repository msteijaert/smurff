#pragma once

#include <vector>
#include <iostream>

#include "TensorConfig.h"

namespace smurff
{
   class MatrixConfig : public TensorConfig
   {
   public:
      MatrixConfig();
      MatrixConfig(int nrow, int ncol, double* values);
      MatrixConfig(int nrow, int ncol, int nnz, int* rows, int* cols, double* values);
      MatrixConfig(int nrow, int ncol, int nnz, int* rows, int* cols);
      MatrixConfig(int nrow, int ncol, int nnz, int* columns, double* values);
      // MatrixConfig(int nrow, int ncol, bool dense, bool binary);
      // MatrixConfig(int nrow, int ncol, int nnz, bool binary = false);

   public:
      std::vector<int> getRows() const;
      std::vector<int> getCols() const;

      int getNRow() const;
      int getNCol() const;
   };
}