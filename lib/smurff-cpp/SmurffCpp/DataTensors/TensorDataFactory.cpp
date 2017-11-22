#include "TensorDataFactory.h"

#include "TensorData.h"
#include <SmurffCpp/Configs/TensorConfig.h>

using namespace smurff;

std::shared_ptr<TensorData> create_to_tensor(const TensorConfig &config, bool scarce)
{
  throw std::runtime_error("not implemented");
}

std::shared_ptr<TensorData> create_tensor(const TensorConfig &train, const std::vector<TensorConfig> &row_features, const std::vector<TensorConfig> &col_features)
{
  throw std::runtime_error("not implemented");
}

std::shared_ptr<Data> TensorDataFactory::create_tensor(std::shared_ptr<Session> session)
{
  throw std::runtime_error("not implemented");
}