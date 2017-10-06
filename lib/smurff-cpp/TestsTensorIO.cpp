#include "catch.hpp"

#include "tensor_io.h"
#include "matrix_io.h"
#include "MatrixUtils.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

using namespace smurff;

TEST_CASE("tensor_io/read_dense_float64_bin | tensor_io/write_dense_float64_bin")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<double> tensorConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   TensorConfig tensorConfig(std::move(tensorConfigDims), std::move(tensorConfigValues), NoiseConfig());

   std::stringstream matrixConfigStream;
   tensor_io::write_dense_float64_bin(matrixConfigStream, tensorConfig);

   TensorConfig actualTensorConfig = tensor_io::read_dense_float64_bin(matrixConfigStream);
   MatrixConfig actualMatrixConfig = tensor_to_matrix(actualTensorConfig);
   
   Eigen::MatrixXd actualMatrix0 = sparse_to_eigen(actualTensorConfig);
   Eigen::MatrixXd actualMatrix1 = sparse_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(actualMatrix0.isApprox(expectedMatrix));
   REQUIRE(actualMatrix1.isApprox(expectedMatrix));
}

TEST_CASE("tensor_io/read_sparse_float64_bin | tensor_io/write_sparse_float64_bin")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::uint32_t> tensorConfigColumns = { 0, 0, 0, 0, 2, 2, 2, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> tensorConfigValues = { 1, 2, 3, 4, 9, 10, 11, 12 };
   TensorConfig tensorConfig( std::move(tensorConfigDims), std::move(tensorConfigColumns), std::move(tensorConfigValues), NoiseConfig());

   std::stringstream tensorStream;
   tensor_io::write_sparse_float64_bin(tensorStream, tensorConfig);

   TensorConfig actualTensorConfig = tensor_io::read_sparse_float64_bin(tensorStream);
   MatrixConfig actualMatrixConfig = tensor_to_matrix(actualTensorConfig);
   
   Eigen::SparseMatrix<double> actualMatrix0 = sparse_to_eigen(actualTensorConfig);
   Eigen::SparseMatrix<double> actualMatrix1 = sparse_to_eigen(actualMatrixConfig);

   Eigen::SparseMatrix<double> expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(actualMatrix0.isApprox(expectedMatrix));
   REQUIRE(actualMatrix1.isApprox(expectedMatrix));
}

TEST_CASE("tensor_io/read_sparse_binary_bin | tensor_io/write_sparse_binary_bin")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::uint32_t> tensorConfigColumns = { 0, 0, 0, 0, 2, 2, 2, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
   TensorConfig tensorConfig(std::move(tensorConfigDims), std::move(tensorConfigColumns), NoiseConfig());

   std::stringstream tensorStream;
   tensor_io::write_sparse_binary_bin(tensorStream, tensorConfig);

   TensorConfig actualTensorConfig = tensor_io::read_sparse_binary_bin(tensorStream);
   MatrixConfig actualMatrixConfig = tensor_to_matrix(actualTensorConfig);
   
   Eigen::SparseMatrix<double> actualMatrix0 = sparse_to_eigen(actualTensorConfig);
   Eigen::SparseMatrix<double> actualMatrix1 = sparse_to_eigen(actualMatrixConfig);

   Eigen::SparseMatrix<double> expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 1));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(actualMatrix0.isApprox(expectedMatrix));
   REQUIRE(actualMatrix1.isApprox(expectedMatrix));
}

TEST_CASE("tensor_io/read_dense_float64_csv | tensor_io/write_dense_float64_csv")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<double> tensorConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   TensorConfig tensorConfig(std::move(tensorConfigDims), std::move(tensorConfigValues), NoiseConfig());

   std::stringstream matrixConfigStream;
   tensor_io::write_dense_float64_csv(matrixConfigStream, tensorConfig);

   TensorConfig actualTensorConfig = tensor_io::read_dense_float64_csv(matrixConfigStream);
   MatrixConfig actualMatrixConfig = tensor_to_matrix(actualTensorConfig);
   
   Eigen::MatrixXd actualMatrix0 = sparse_to_eigen(actualTensorConfig);
   Eigen::MatrixXd actualMatrix1 = sparse_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(actualMatrix0.isApprox(expectedMatrix));
   REQUIRE(actualMatrix1.isApprox(expectedMatrix));
}

TEST_CASE("tensor_io/read_sparse_float64_tns | tensor_io/write_sparse_float64_tns")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::uint32_t> tensorConfigColumns = { 0, 0, 0, 0, 2, 2, 2, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> tensorConfigValues = { 1, 2, 3, 4, 9, 10, 11, 12 };
   TensorConfig tensorConfig( std::move(tensorConfigDims), std::move(tensorConfigColumns), std::move(tensorConfigValues), NoiseConfig());

   std::stringstream tensorStream;
   tensor_io::write_sparse_float64_tns(tensorStream, tensorConfig);

   TensorConfig actualTensorConfig = tensor_io::read_sparse_float64_tns(tensorStream);
   MatrixConfig actualMatrixConfig = tensor_to_matrix(actualTensorConfig);
   
   Eigen::SparseMatrix<double> actualMatrix0 = sparse_to_eigen(actualTensorConfig);
   Eigen::SparseMatrix<double> actualMatrix1 = sparse_to_eigen(actualMatrixConfig);

   Eigen::SparseMatrix<double> expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(actualMatrix0.isApprox(expectedMatrix));
   REQUIRE(actualMatrix1.isApprox(expectedMatrix));
}