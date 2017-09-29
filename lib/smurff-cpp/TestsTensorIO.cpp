#include "catch.hpp"

#include "tensor_io.h"
#include "matrix_io.h"
#include "MatrixUtils.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

using namespace smurff;

TEST_CASE("tensor_io/read_dense_float64_bin | tensor_io/write_dense_float64_bin")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 3;
   std::vector<double> matrixConfigValues = { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   std::stringstream matrixConfigStream;
   tensor_io::write_dense_float64_bin(matrixConfigStream, matrixConfig);

   TensorConfig actualTensorConfig = tensor_io::read_dense_float64_bin(matrixConfigStream);
   MatrixConfig actualMatrixConfig( matrixConfigNRow
                                  , matrixConfigNCol
                                  , actualTensorConfig.getColumns()
                                  , actualTensorConfig.getValues()
                                  , NoiseConfig()
                                  );
   Eigen::MatrixXd actualMatrix = sparse_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("tensor_io/read_sparse_float64_bin | tensor_io/write_sparse_float64_bin")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 3;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 0, 1, 2 };
   std::vector<double> matrixConfigValues      = { 1, 2, 3, 7, 8, 9 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigRows)
                            , std::move(matrixConfigCols)
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   std::stringstream tensorStream;
   tensor_io::write_sparse_float64_bin(tensorStream, matrixConfig);
   TensorConfig actualTensorConfig = tensor_io::read_sparse_float64_bin(tensorStream);
   MatrixConfig actualMatrixConfig( matrixConfigNRow
                                  , matrixConfigNCol
                                  , actualTensorConfig.getColumns()
                                  , actualTensorConfig.getValues()
                                  , NoiseConfig()
                                  );
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 7));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 8));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 9));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("tensor_io/read_sparse_binary_bin | tensor_io/write_sparse_binary_bin")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 3;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 0, 1, 2 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigRows)
                            , std::move(matrixConfigCols)
                            , NoiseConfig()
                            );

   std::stringstream tensorConfigStream;
   tensor_io::write_sparse_binary_bin(tensorConfigStream, matrixConfig);

   TensorConfig actualTensorConfig = tensor_io::read_sparse_binary_bin(tensorConfigStream);
   MatrixConfig actualMatrixConfig( matrixConfigNRow
                                  , matrixConfigNCol
                                  , actualTensorConfig.getColumns()
                                  , actualTensorConfig.getValues()
                                  , NoiseConfig()
                                  );
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 1));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}