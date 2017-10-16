#pragma once

#include <memory>
#include <vector>

#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/DataMatrices/MatrixData.h>

namespace smurff {

   std::unique_ptr<MatrixData> matrix_config_to_matrix(const MatrixConfig &train, 
      const std::vector<MatrixConfig> &row_features, 
      const std::vector<MatrixConfig> &col_features);

}