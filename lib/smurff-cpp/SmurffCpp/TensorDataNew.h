#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "SparseModeNew.h"
#include <SmurffCpp/Configs/TensorConfig.h>

class TensorDataNew
{
private:
   std::vector<std::uint64_t> m_dims; //vector of dimention sizes
   std::shared_ptr<std::vector<std::shared_ptr<SparseModeNew> > > m_Y; // this is a vector of tensor rotations

public:
   TensorDataNew(const smurff::TensorConfig& tc);

   std::shared_ptr<SparseModeNew> Y(std::uint64_t mode) const;

   std::uint64_t getNModes() const;
};