#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/TensorUtils.h>
#include <SmurffCpp/Configs/TensorConfig.h>

using namespace smurff;

TEST_CASE("TensorConfig(const std::vector<std::uint64_t>& dims, const std::vector<double> values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<double> tensorConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   TensorConfig tensorConfig(tensorConfigDims, tensorConfigValues, NoiseConfig());
   MatrixConfig matrixConfig = tensor_utils::tensor_to_matrix(tensorConfig);

   Eigen::MatrixXd actualMatrix0 = tensor_utils::sparse_to_eigen(tensorConfig);
   Eigen::MatrixXd actualMatrix1 = matrix_utils::sparse_to_eigen(matrixConfig);
   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix0, expectedMatrix));
   REQUIRE(matrix_utils::equals(actualMatrix1, expectedMatrix));
}

TEST_CASE("TensorConfig(std::vector<std::uint64_t>&& dims, std::vector<double>&& values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<double> tensorConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   TensorConfig tensorConfig(std::move(tensorConfigDims), std::move(tensorConfigValues), NoiseConfig());
   MatrixConfig matrixConfig = tensor_utils::tensor_to_matrix(tensorConfig);

   Eigen::MatrixXd actualMatrix0 = tensor_utils::sparse_to_eigen(tensorConfig);
   Eigen::MatrixXd actualMatrix1 = matrix_utils::sparse_to_eigen(matrixConfig);
   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(tensorConfigDims.data() == NULL);
   REQUIRE(tensorConfigValues.data() == NULL);

   REQUIRE(matrix_utils::equals(actualMatrix0, expectedMatrix));
   REQUIRE(matrix_utils::equals(actualMatrix1, expectedMatrix));
}

TEST_CASE("TensorConfig(std::shared_ptr<std::vector<std::uint64_t> > dims, std::shared_ptr<std::vector<double> > values, const NoiseConfig& noiseConfig)")
{
   std::shared_ptr<std::vector<std::uint64_t> > tensorConfigDims =
      std::make_shared<std::vector<std::uint64_t> >(
         std::initializer_list<std::uint64_t>({ 3, 4 })
      );
   std::shared_ptr<std::vector<double> > tensorConfigValues =
      std::make_shared<std::vector<double> >(
         std::initializer_list<double>({ 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 })
      );
   TensorConfig tensorConfig(tensorConfigDims, tensorConfigValues, NoiseConfig());
   MatrixConfig matrixConfig = tensor_utils::tensor_to_matrix(tensorConfig);

   Eigen::MatrixXd actualMatrix0 = tensor_utils::sparse_to_eigen(tensorConfig);
   Eigen::MatrixXd actualMatrix1 = matrix_utils::sparse_to_eigen(matrixConfig);
   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix0, expectedMatrix));
   REQUIRE(matrix_utils::equals(actualMatrix1, expectedMatrix));
}

TEST_CASE("TensorConfig(const std::vector<std::uint64_t>& dims, const std::vector<std::uint32_t>& columns, const std::vector<double>& values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::uint32_t> tensorConfigColumns = { 0, 0, 0, 0, 2, 2, 2, 2,
                                                      0, 1, 2, 3, 0, 1, 2, 3
                                                    };
   std::vector<double> tensorConfigValues = { 1, 2, 3, 4, 9, 10, 11, 12 };
   TensorConfig tensorConfig(tensorConfigDims, tensorConfigColumns, tensorConfigValues, NoiseConfig());
   MatrixConfig matrixConfig = tensor_utils::tensor_to_matrix(tensorConfig);

   Eigen::SparseMatrix<double> actualMatrix0 = tensor_utils::sparse_to_eigen(tensorConfig);
   Eigen::SparseMatrix<double> actualMatrix1 = matrix_utils::sparse_to_eigen(matrixConfig);
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

   REQUIRE(matrix_utils::equals(actualMatrix0, expectedMatrix));
   REQUIRE(matrix_utils::equals(actualMatrix1, expectedMatrix));
}

TEST_CASE("TensorConfig(std::vector<std::uint64_t>&& dims, std::vector<std::uint32_t>&& columns, std::vector<double>&& values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::uint32_t> tensorConfigColumns = { 0, 0, 0, 0, 2, 2, 2, 2,
                                                      0, 1, 2, 3, 0, 1, 2, 3
                                                    };
   std::vector<double> tensorConfigValues = { 1, 2, 3, 4, 9, 10, 11, 12 };
   TensorConfig tensorConfig( std::move(tensorConfigDims)
                            , std::move(tensorConfigColumns)
                            , std::move(tensorConfigValues)
                            , NoiseConfig()
                            );
   MatrixConfig matrixConfig = tensor_utils::tensor_to_matrix(tensorConfig);

   Eigen::SparseMatrix<double> actualMatrix0 = tensor_utils::sparse_to_eigen(tensorConfig);
   Eigen::SparseMatrix<double> actualMatrix1 = matrix_utils::sparse_to_eigen(matrixConfig);
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

   REQUIRE(tensorConfigDims.data() == NULL);
   REQUIRE(tensorConfigColumns.data() == NULL);
   REQUIRE(tensorConfigValues.data() == NULL);

   REQUIRE(matrix_utils::equals(actualMatrix0, expectedMatrix));
   REQUIRE(matrix_utils::equals(actualMatrix1, expectedMatrix));
}

TEST_CASE("TensorConfig(std::shared_ptr<std::vector<std::uint64_t> > dims, std::shared_ptr<std::vector<std::uint32_t> > columns, std::shared_ptr<std::vector<double> > values, const NoiseConfig& noiseConfig)")
{
   std::shared_ptr<std::vector<std::uint64_t> > tensorConfigDims =
      std::make_shared<std::vector<std::uint64_t> >(
         std::initializer_list<std::uint64_t>({ 3, 4 })
      );
   std::shared_ptr<std::vector<std::uint32_t> > tensorConfigColumns =
      std::make_shared<std::vector<std::uint32_t> >(
         std::initializer_list<std::uint32_t>({
            0, 0, 0, 0, 2, 2, 2, 2,
            0, 1, 2, 3, 0, 1, 2, 3
         })
      );
   std::shared_ptr<std::vector<double> > tensorConfigValues =
      std::make_shared<std::vector<double> >(
         std::initializer_list<double>({ 1, 2, 3, 4, 9, 10, 11, 12 })
      );
   TensorConfig tensorConfig(tensorConfigDims, tensorConfigColumns, tensorConfigValues, NoiseConfig());
   MatrixConfig matrixConfig = tensor_utils::tensor_to_matrix(tensorConfig);

   Eigen::SparseMatrix<double> actualMatrix0 = tensor_utils::sparse_to_eigen(tensorConfig);
   Eigen::SparseMatrix<double> actualMatrix1 = matrix_utils::sparse_to_eigen(matrixConfig);
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

   REQUIRE(matrix_utils::equals(actualMatrix0, expectedMatrix));
   REQUIRE(matrix_utils::equals(actualMatrix1, expectedMatrix));
}

TEST_CASE("TensorConfig(const std::vector<std::uint64_t>& dims, const std::vector<std::uint32_t>& columns, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::uint32_t> tensorConfigColumns = { 0, 0, 0, 0, 2, 2, 2, 2,
                                                      0, 1, 2, 3, 0, 1, 2, 3
                                                    };
   TensorConfig tensorConfig(tensorConfigDims, tensorConfigColumns, NoiseConfig());
   MatrixConfig matrixConfig = tensor_utils::tensor_to_matrix(tensorConfig);

   Eigen::SparseMatrix<double> actualMatrix0 = tensor_utils::sparse_to_eigen(tensorConfig);
   Eigen::SparseMatrix<double> actualMatrix1 = matrix_utils::sparse_to_eigen(matrixConfig);
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

   REQUIRE(matrix_utils::equals(actualMatrix0, expectedMatrix));
   REQUIRE(matrix_utils::equals(actualMatrix1, expectedMatrix));
}

TEST_CASE("TensorConfig(std::vector<std::uint64_t>&& dims, std::vector<std::uint32_t>&& columns, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::uint32_t> tensorConfigColumns = { 0, 0, 0, 0, 2, 2, 2, 2,
                                                      0, 1, 2, 3, 0, 1, 2, 3
                                                    };
   TensorConfig tensorConfig(std::move(tensorConfigDims), std::move(tensorConfigColumns), NoiseConfig());
   MatrixConfig matrixConfig = tensor_utils::tensor_to_matrix(tensorConfig);

   Eigen::SparseMatrix<double> actualMatrix0 = tensor_utils::sparse_to_eigen(tensorConfig);
   Eigen::SparseMatrix<double> actualMatrix1 = matrix_utils::sparse_to_eigen(matrixConfig);
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

   REQUIRE(tensorConfigDims.data() == NULL);
   REQUIRE(tensorConfigColumns.data() == NULL);

   REQUIRE(matrix_utils::equals(actualMatrix0, expectedMatrix));
   REQUIRE(matrix_utils::equals(actualMatrix1, expectedMatrix));
}

TEST_CASE("TensorConfig(std::shared_ptr<std::vector<std::uint64_t> > dims, std::shared_ptr<std::vector<std::uint32_t> > columns, const NoiseConfig& noiseConfig)")
{
   std::shared_ptr<std::vector<std::uint64_t> > tensorConfigDims =
      std::make_shared<std::vector<std::uint64_t> >(
         std::initializer_list<std::uint64_t>({ 3, 4 })
      );
   std::shared_ptr<std::vector<std::uint32_t> > tensorConfigColumns =
      std::make_shared<std::vector<std::uint32_t> >(
         std::initializer_list<std::uint32_t>({
            0, 0, 0, 0, 2, 2, 2, 2,
            0, 1, 2, 3, 0, 1, 2, 3
         })
      );
   TensorConfig tensorConfig(tensorConfigDims, tensorConfigColumns, NoiseConfig());
   MatrixConfig matrixConfig = tensor_utils::tensor_to_matrix(tensorConfig);

   Eigen::SparseMatrix<double> actualMatrix0 = tensor_utils::sparse_to_eigen(tensorConfig);
   Eigen::SparseMatrix<double> actualMatrix1 = matrix_utils::sparse_to_eigen(matrixConfig);
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

   REQUIRE(matrix_utils::equals(actualMatrix0, expectedMatrix));
   REQUIRE(matrix_utils::equals(actualMatrix1, expectedMatrix));
}