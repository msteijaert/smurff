#pragma once

#include <memory>
#include <vector>

#include "MatrixConfig.h"
#include "MatrixData.h"

namespace smurff {

   std::unique_ptr<MatrixData> matrix_config_to_matrix(const MatrixConfig &train, 
      const std::vector<MatrixConfig> &row_features, 
      const std::vector<MatrixConfig> &col_features);

}