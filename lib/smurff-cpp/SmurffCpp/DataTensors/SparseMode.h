#pragma once

#include <vector>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

typedef Eigen::Matrix<std::uint32_t, Eigen::Dynamic, Eigen::Dynamic > MatrixXui32;

//this is a tensor rotation where one dimention is fixed (excluded)
class SparseMode
{
private:
   std::uint64_t m_mode; // index of dimension that it fixed

   std::vector<std::uint64_t> m_row_ptr; // vector of offsets (in values and in indices) to each hyperplane
   MatrixXui32 m_indices; // [m_nnz x (m_nmodes - 1)] matrix of coordinates
   std::vector<double> m_values; // vector of values
   
public:
   SparseMode();

   // idx - [nnz x nmodes] matrix of coordinates
   // vals - vector of values
   // mode - index of dimension to fix
   // mode_size - size of dimension to fix
   SparseMode(const MatrixXui32& idx, const std::vector<double>& vals, std::uint64_t mode, std::uint64_t mode_size);

   std::uint64_t getNNZ() const;

   std::uint64_t getNPlanes() const;

   std::uint64_t getNCoords() const;

   const std::vector<double>& getValues() const;

   std::uint64_t getMode() const;

   std::uint64_t beginPlane(std::uint64_t hyperplane) const;

   std::uint64_t endPlane(std::uint64_t hyperplane) const;

   std::uint64_t nItemsOnPlane(std::uint64_t hyperplane) const;

   const MatrixXui32& getIndices() const;

public:
   std::pair<PVec<>, double> item(std::uint64_t hyperplane, std::uint64_t item) const;
   
   PVec<> pos(std::uint64_t hyperplane, std::uint64_t item) const;
};

}
