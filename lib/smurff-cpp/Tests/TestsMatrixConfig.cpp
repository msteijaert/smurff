#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

using namespace smurff;

TEST_CASE("MatrixConfig::MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<double>& values, const NoiseConfig& noiseConfig)")
{
   std::vector<double> actualMatrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigValues, NoiseConfig());
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix(0, 0) = 1; expectedMatrix(0, 1) = 2; expectedMatrix(0, 2) = 3; expectedMatrix(0, 3) = 4;
   expectedMatrix(1, 0) = 5; expectedMatrix(1, 1) = 6; expectedMatrix(1, 2) = 7; expectedMatrix(1, 3) = 8;
   expectedMatrix(2, 0) = 9; expectedMatrix(2, 1) = 10; expectedMatrix(2, 2) = 11; expectedMatrix(2, 3) = 12;

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::vector<double>&& values, const NoiseConfig& noiseConfig)")
{
   std::vector<double> actualMatrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   MatrixConfig actualMatrixConfig(3, 4, std::move(actualMatrixConfigValues), NoiseConfig());
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix(0, 0) = 1; expectedMatrix(0, 1) = 2; expectedMatrix(0, 2) = 3; expectedMatrix(0, 3) = 4;
   expectedMatrix(1, 0) = 5; expectedMatrix(1, 1) = 6; expectedMatrix(1, 2) = 7; expectedMatrix(1, 3) = 8;
   expectedMatrix(2, 0) = 9; expectedMatrix(2, 1) = 10; expectedMatrix(2, 2) = 11; expectedMatrix(2, 3) = 12;

   REQUIRE(actualMatrixConfigValues.data() == NULL);
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::shared_ptr<std::vector<double> > values, const NoiseConfig& noiseConfig)")
{
   std::shared_ptr<std::vector<double> > actualMatrixConfigValues =
      std::make_shared<std::vector<double> >(std::initializer_list<double>({ 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 }));
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigValues, NoiseConfig());
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix(0, 0) = 1; expectedMatrix(0, 1) = 2; expectedMatrix(0, 2) = 3; expectedMatrix(0, 3) = 4;
   expectedMatrix(1, 0) = 5; expectedMatrix(1, 1) = 6; expectedMatrix(1, 2) = 7; expectedMatrix(1, 3) = 8;
   expectedMatrix(2, 0) = 9; expectedMatrix(2, 1) = 10; expectedMatrix(2, 2) = 11; expectedMatrix(2, 3) = 12;

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<std::uint32_t>& rows, const std::vector<std::uint32_t>& cols, const std::vector<double>& values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint32_t> actualMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> actualMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> actualMatrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigRows, actualMatrixConfigCols, actualMatrixConfigValues, NoiseConfig());
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

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

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::vector<std::uint32_t>&& rows, std::vector<std::uint32_t>&& cols, std::vector<double>&& values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint32_t> actualMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> actualMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> actualMatrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   MatrixConfig actualMatrixConfig(3, 4, std::move(actualMatrixConfigRows), std::move(actualMatrixConfigCols), std::move(actualMatrixConfigValues), NoiseConfig());
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

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

   REQUIRE(actualMatrixConfigRows.data() == NULL);
   REQUIRE(actualMatrixConfigCols.data() == NULL);
   REQUIRE(actualMatrixConfigValues.data() == NULL);
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::shared_ptr<std::vector<std::uint32_t> > rows, std::shared_ptr<std::vector<std::uint32_t> > cols, std::shared_ptr<std::vector<double> > values, const NoiseConfig& noiseConfig)")
{
   std::shared_ptr<std::vector<std::uint32_t> > actualMatrixConfigRows =
      std::make_shared<std::vector<std::uint32_t> >(std::initializer_list<std::uint32_t>({ 0, 0, 0, 0, 2, 2, 2, 2 }));
   std::shared_ptr<std::vector<std::uint32_t> > actualMatrixConfigCols =
      std::make_shared<std::vector<std::uint32_t> >(std::initializer_list<std::uint32_t>({ 0, 1, 2, 3, 0, 1, 2, 3}));
   std::shared_ptr<std::vector<double> > actualMatrixConfigValues =
      std::make_shared<std::vector<double> >(std::initializer_list<double>({ 1, 2, 3, 4, 9, 10, 11, 12 }));
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigRows, actualMatrixConfigCols, actualMatrixConfigValues, NoiseConfig());
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

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

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<std::uint32_t>& rows, const std::vector<std::uint32_t>& cols, const NoiseConfig& noiseConfig)")
{
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

   std::vector<std::uint32_t> actualMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> actualMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigRows, actualMatrixConfigCols, NoiseConfig());
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::vector<std::uint32_t>&& rows, std::vector<std::uint32_t>&& cols, const NoiseConfig& noiseConfig)")
{
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

   std::vector<std::uint32_t> actualMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> actualMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   MatrixConfig actualMatrixConfig(3, 4, std::move(actualMatrixConfigRows), std::move(actualMatrixConfigCols), NoiseConfig());
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   REQUIRE(actualMatrixConfigRows.data() == NULL);
   REQUIRE(actualMatrixConfigCols.data() == NULL);
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::shared_ptr<std::vector<std::uint32_t> > rows, std::shared_ptr<std::vector<std::uint32_t> > cols, const NoiseConfig& noiseConfig)")
{
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

   std::shared_ptr<std::vector<std::uint32_t> > actualMatrixConfigRows =
      std::make_shared<std::vector<std::uint32_t> >(std::initializer_list<uint32_t>({ 0, 0, 0, 0, 2, 2, 2, 2 }));
   std::shared_ptr<std::vector<std::uint32_t> > actualMatrixConfigCols =
      std::make_shared<std::vector<std::uint32_t> >(std::initializer_list<uint32_t>({ 0, 1, 2, 3, 0, 1, 2, 3 }));
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigRows, actualMatrixConfigCols, NoiseConfig());
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

//---

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<std::uint32_t>& columns, const std::vector<double>& values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint32_t> actualMatrixConfigColumns = { 
         0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
         0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
   };
   std::vector<double> actualMatrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };

   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigColumns, actualMatrixConfigValues, NoiseConfig());
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix(0, 0) = 1; expectedMatrix(0, 1) = 2; expectedMatrix(0, 2) = 3; expectedMatrix(0, 3) = 4;
   expectedMatrix(1, 0) = 5; expectedMatrix(1, 1) = 6; expectedMatrix(1, 2) = 7; expectedMatrix(1, 3) = 8;
   expectedMatrix(2, 0) = 9; expectedMatrix(2, 1) = 10; expectedMatrix(2, 2) = 11; expectedMatrix(2, 3) = 12;

   REQUIRE(actualMatrix.isApprox(expectedMatrix));   
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::vector<std::uint32_t>&& columns, std::vector<double>&& values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint32_t> actualMatrixConfigColumns = { 
      0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
      0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
   };
   std::vector<double> actualMatrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };

   MatrixConfig actualMatrixConfig(3, 4, std::move(actualMatrixConfigColumns), std::move(actualMatrixConfigValues), NoiseConfig());
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix(0, 0) = 1; expectedMatrix(0, 1) = 2; expectedMatrix(0, 2) = 3; expectedMatrix(0, 3) = 4;
   expectedMatrix(1, 0) = 5; expectedMatrix(1, 1) = 6; expectedMatrix(1, 2) = 7; expectedMatrix(1, 3) = 8;
   expectedMatrix(2, 0) = 9; expectedMatrix(2, 1) = 10; expectedMatrix(2, 2) = 11; expectedMatrix(2, 3) = 12;

   REQUIRE(actualMatrixConfigColumns.data() == NULL);
   REQUIRE(actualMatrixConfigValues.data() == NULL);
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::shared_ptr<std::vector<std::uint32_t> > columns, std::shared_ptr<std::vector<double> > values, const NoiseConfig& noiseConfig)")
{
   std::shared_ptr<std::vector<std::uint32_t> > actualMatrixConfigColumns =
      std::make_shared<std::vector<std::uint32_t> >(
         std::initializer_list<std::uint32_t>{ 
            0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
            }
         );
   std::shared_ptr<std::vector<double> > actualMatrixConfigValues =
      std::make_shared<std::vector<double> >(
         std::initializer_list<double>{ 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 }
      );
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigColumns, actualMatrixConfigValues, NoiseConfig());
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix(0, 0) = 1; expectedMatrix(0, 1) = 2; expectedMatrix(0, 2) = 3; expectedMatrix(0, 3) = 4;
   expectedMatrix(1, 0) = 5; expectedMatrix(1, 1) = 6; expectedMatrix(1, 2) = 7; expectedMatrix(1, 3) = 8;
   expectedMatrix(2, 0) = 9; expectedMatrix(2, 1) = 10; expectedMatrix(2, 2) = 11; expectedMatrix(2, 3) = 12;

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}