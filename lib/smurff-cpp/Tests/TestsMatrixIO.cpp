#include "catch.hpp"

#include <sstream>
#include <cstdio>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/IO/MatrixIO.h>

using namespace smurff;

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .ddm")
{
   std::string matrixFilename = "matrixConfig.ddm";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   matrix_io::write_matrix(matrixFilename, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_matrix(matrixFilename);
   Eigen::MatrixXd actualMatrix = matrix_utils::dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .csv")
{
   std::string matrixFilename = "matrixConfig.csv";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   matrix_io::write_matrix(matrixFilename, matrixConfig);

   MatrixConfig actualMatrixConfig = matrix_io::read_matrix(matrixFilename);
   Eigen::MatrixXd actualMatrix = matrix_utils::dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .sdm")
{
   std::string matrixFilename = "matrixConfig.sdm";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> matrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigRows)
                            , std::move(matrixConfigCols)
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   matrix_io::write_matrix(matrixFilename, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_matrix(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = matrix_utils::sparse_to_eigen(actualMatrixConfig);

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

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .mtx")
{
   std::string matrixFilename = "matrixConfig.mtx";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> matrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigRows)
                            , std::move(matrixConfigCols)
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   matrix_io::write_matrix(matrixFilename, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_matrix(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = matrix_utils::sparse_to_eigen(actualMatrixConfig);

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

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .sbm")
{
   std::string matrixFilename = "matrixConfig.sbm";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigRows)
                            , std::move(matrixConfigCols)
                            , NoiseConfig()
                            );

   matrix_io::write_matrix(matrixFilename, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_matrix(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = matrix_utils::sparse_to_eigen(actualMatrixConfig);

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

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

// ===

TEST_CASE("matrix_io/read_matrix_market | matrix_io/write_matrix_market | dense")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   std::stringstream matrixStream;
   matrix_io::write_matrix_market(matrixStream, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_matrix_market(matrixStream);
   Eigen::MatrixXd actualMatrix = matrix_utils::dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix_market | matrix_io/write_matrix_market | sparse")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> matrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigRows)
                            , std::move(matrixConfigCols)
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   std::stringstream matrixStream;
   matrix_io::write_matrix_market(matrixStream, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_matrix_market(matrixStream);
   Eigen::SparseMatrix<double> actualMatrix = matrix_utils::sparse_to_eigen(actualMatrixConfig);

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

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_dense_float64_bin | matrix_io/write_dense_float64_bin")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   std::stringstream matrixStream;
   matrix_io::write_dense_float64_bin(matrixStream, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_dense_float64_bin(matrixStream);
   Eigen::MatrixXd actualMatrix = matrix_utils::dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_dense_float64_csv | matrix_io/write_dense_float64_csv")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   std::stringstream matrixConfigStream;
   matrix_io::write_dense_float64_csv(matrixConfigStream, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_dense_float64_csv(matrixConfigStream);
   Eigen::MatrixXd actualMatrix = matrix_utils::dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse_float64_bin | matrix_io/write_sparse_float64_bin")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> matrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigRows)
                            , std::move(matrixConfigCols)
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   std::stringstream matrixConfigStream;
   matrix_io::write_sparse_float64_bin(matrixConfigStream, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_sparse_float64_bin(matrixConfigStream);
   Eigen::SparseMatrix<double> actualMatrix = matrix_utils::sparse_to_eigen(actualMatrixConfig);

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

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse_float64_mtx | matrix_io/write_sparse_float64_mtx")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> matrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigRows)
                            , std::move(matrixConfigCols)
                            , std::move(matrixConfigValues)
                            , NoiseConfig()
                            );

   std::stringstream matrixConfigStream;
   matrix_io::write_sparse_float64_mtx(matrixConfigStream, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_sparse_float64_mtx(matrixConfigStream);
   Eigen::SparseMatrix<double> actualMatrix = matrix_utils::sparse_to_eigen(actualMatrixConfig);

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

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse_binary_bin | matrix_io/write_sparse_binary_bin")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   MatrixConfig matrixConfig( matrixConfigNRow
                            , matrixConfigNCol
                            , std::move(matrixConfigRows)
                            , std::move(matrixConfigCols)
                            , NoiseConfig()
                            );

   std::stringstream matrixConfigStream;
   matrix_io::write_sparse_binary_bin(matrixConfigStream, matrixConfig);
   MatrixConfig actualMatrixConfig = matrix_io::read_sparse_binary_bin(matrixConfigStream);
   Eigen::SparseMatrix<double> actualMatrix = matrix_utils::sparse_to_eigen(actualMatrixConfig);

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

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

// ===

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Eigen::VectorXd& V) | matrix_io/eigen::write_matrix(const std::string& filename, const Eigen::MatrixXd& X) | .ddm")
{
   std::string matrixFilename = "eigenVector.ddm";

   Eigen::VectorXd expectedMatrix(3);
   expectedMatrix(0) = 1;
   expectedMatrix(1) = 4;
   expectedMatrix(2) = 9;
   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Eigen::VectorXd actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Eigen::VectorXd& V) | matrix_io/eigen::write_matrix(const std::string& filename, const Eigen::MatrixXd& X) | .csv")
{
   std::string matrixFilename = "eigenVector.csv";

   Eigen::VectorXd expectedMatrix(3);
   expectedMatrix(0) = 1;
   expectedMatrix(1) = 4;
   expectedMatrix(2) = 9;
   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Eigen::VectorXd actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

// ===

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Eigen::MatrixXd& X) | matrix_io/eigen::write_matrix(const std::string& filename, const Eigen::MatrixXd& X) | .ddm")
{
   std::string matrixFilename = "denseEigenMatrix.ddm";

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Eigen::MatrixXd actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Eigen::MatrixXd& X) | matrix_io/eigen::write_matrix(const std::string& filename, const Eigen::MatrixXd& X) | .csv")
{
   std::string matrixFilename = "denseEigenMatrix.csv";

   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Eigen::MatrixXd actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

// ===

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Eigen::SparseMatrix<double>& X) | matrix_io/eigen::write_matrix(const std::string& filename, const Eigen::SparseMatrix<double>& X) | .sdm")
{
   std::string matrixFilename = "sparseEigenMatrix.sdm";

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

   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Eigen::SparseMatrix<double> actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Eigen::SparseMatrix<double>& X) | matrix_io/eigen::write_matrix(const std::string& filename, const Eigen::SparseMatrix<double>& X) | .mtx")
{
   std::string matrixFilename = "sparseEigenMatrix.mtx";

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

   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Eigen::SparseMatrix<double> actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Eigen::SparseMatrix<double>& X) | matrix_io/eigen::write_matrix(const std::string& filename, const Eigen::SparseMatrix<double>& X) | .sbm")
{
   std::string matrixFilename = "sparseEigenMatrix.sbm";

   Eigen::SparseMatrix<double> expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 1));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Eigen::SparseMatrix<double> actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

// ===

TEST_CASE("matrix_io/eigen::read_dense_float64_bin | matrix_io/eigen::write_dense_float64_bin")
{
   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   std::stringstream matrixStream;
   matrix_io::eigen::write_dense_float64_bin(matrixStream, expectedMatrix);
   Eigen::MatrixXd actualMatrix;
   matrix_io::eigen::read_dense_float64_bin(matrixStream, actualMatrix);

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_dense_float64_csv | matrix_io/eigen::write_dense_float64_csv")
{
   Eigen::MatrixXd expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   std::stringstream matrixStream;
   matrix_io::eigen::write_dense_float64_csv(matrixStream, expectedMatrix);
   Eigen::MatrixXd actualMatrix;
   matrix_io::eigen::read_dense_float64_csv(matrixStream, actualMatrix);

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_sparse_float64_bin | matrix_io/eigen::write_sparse_float64_bin")
{
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

   std::stringstream matrixStream;
   matrix_io::eigen::write_sparse_float64_bin(matrixStream, expectedMatrix);
   Eigen::SparseMatrix<double> actualMatrix;;
   matrix_io::eigen::read_sparse_float64_bin(matrixStream, actualMatrix);

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_sparse_float64_mtx | matrix_io/eigen::write_sparse_float64_mtx")
{
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

   std::stringstream matrixStream;
   matrix_io::eigen::write_sparse_float64_mtx(matrixStream, expectedMatrix);
   Eigen::SparseMatrix<double> actualMatrix;
   matrix_io::eigen::read_sparse_float64_mtx(matrixStream, actualMatrix);

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_sparse_binary_bin | matrix_io/eigen::write_sparse_binary_bin")
{
   Eigen::SparseMatrix<double> expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 0));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 1));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   std::stringstream matrixStream;
   matrix_io::eigen::write_sparse_binary_bin(matrixStream, expectedMatrix);
   Eigen::SparseMatrix<double> actualMatrix;
   matrix_io::eigen::read_sparse_binary_bin(matrixStream, actualMatrix);

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}