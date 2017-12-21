#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "SparseMode.h"
#include <SmurffCpp/Configs/TensorConfig.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

class TensorData : public Data
{
private:
   std::vector<std::uint64_t> m_dims; //vector of dimention sizes
   std::uint64_t m_nnz;
   std::shared_ptr<std::vector<std::shared_ptr<SparseMode> > > m_Y; // this is a vector of tensor rotations

public:
   TensorData(const smurff::TensorConfig& tc);

   std::shared_ptr<SparseMode> Y(std::uint64_t mode) const;

protected:
   void init_pre() override;

public:
   double sum() const override;

public:
   std::uint64_t nmode() const override;
   std::uint64_t nnz() const override;
   std::uint64_t nna() const override;
   PVec<> dim() const override;

public:
   double train_rmse(const SubModel& model) const override;
   void get_pnm(const SubModel& model, uint32_t mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override;
   void update_pnm(const SubModel& model, uint32_t mode) override;

public:
   double sumsq(const SubModel& model) const override;
   double var_total() const override;

public:
   PVec<> pos(std::uint64_t mode, std::uint64_t hyperplane, std::uint64_t item) const;
};

}