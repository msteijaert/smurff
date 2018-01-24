#pragma once

#include <memory>
#include <vector>

#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/DataMatrices/Data.h>

namespace smurff {

class MatrixDataFactory
{
public:
   static std::shared_ptr<Data> create_matrix_data(std::shared_ptr<const MatrixConfig> matrixConfig, 
                                                   const std::vector<std::shared_ptr<MatrixConfig> >& row_features, 
                                                   const std::vector<std::shared_ptr<MatrixConfig> >& col_features);
};

}
