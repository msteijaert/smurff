TEST_CASE("matrix_io/writeToCSVfile | matrix_io/readFromCSVfile")
{
   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   std::string matrixFilename = "writeToCSVfile.csv";
   writeToCSVfile(matrixFilename, expectedMatrix);

   Eigen::MatrixXd actualMatrix;
   readFromCSVfile(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/writeToCSVstream | matrix_io/readFromCSVstream")
{
   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   std::stringstream matrixStream;
   writeToCSVstream(matrixStream, expectedMatrix);

   Eigen::MatrixXd actualMatrix;
   readFromCSVstream(matrixStream, actualMatrix);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}


TEST_CASE("matrix_io/write_ddm(const std::string& filename, const Eigen::MatrixXd& matrix) | matrix_io/read_ddm(const std::string& filename, Eigen::MatrixXd& matrix)")
{
   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   std::string matrixFilename = "write_read_ddm.ddm";
   write_ddm(matrixFilename, expectedMatrix);

   Eigen::MatrixXd actualMatrix;
   read_ddm(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/write_ddm(std::ostream& out, const Eigen::MatrixXd& matrix) | matrix_io/read_ddm(std::istream& in, Eigen::MatrixXd& matrix)")
{
   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   std::stringstream matrixStream;
   write_ddm(matrixStream, expectedMatrix);

   Eigen::MatrixXd actualMatrix;
   read_ddm(matrixStream, actualMatrix);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/write_dense(const std::string& fname, const Eigen::MatrixXd&) | matrix_io/read_dense(const std::string& fname, Eigen::MatrixXd& X). .ddm")
{
   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   std::string matrixFilename = "write_read_dense1.ddm";
   write_dense(matrixFilename, expectedMatrix);

   Eigen::MatrixXd actualMatrix;
   read_dense(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/write_dense(const std::string& fname, const Eigen::MatrixXd&) | matrix_io/read_dense(const std::string& fname, Eigen::MatrixXd& X). .csv")
{
   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   std::string matrixFilename = "write_read_dense1.csv";
   write_dense(matrixFilename, expectedMatrix);

   Eigen::MatrixXd actualMatrix;
   read_dense(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/write_dense(std::ostream& out, DenseMatrixType denseMatrixType, const Eigen::MatrixXd&) | read_dense(std::istream& in, DenseMatrixType denseMatrixType, Eigen::MatrixXd& X). DenseMatrixType::ddm")
{
   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   std::stringstream matrixStream;
   write_dense(matrixStream, DenseMatrixType::ddm, expectedMatrix);

   Eigen::MatrixXd actualMatrix;
   read_dense(matrixStream, DenseMatrixType::ddm, actualMatrix);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense(const std::string& fname, Eigen::VectorXd &). .ddm")
{
   Eigen::VectorXd expectedMatrix(3);
   expectedMatrix(0) = 1;
   expectedMatrix(1) = 4;
   expectedMatrix(2) = 9;

   std::string matrixFilename = "write_read_dense2.ddm";
   write_dense(matrixFilename, expectedMatrix);

   Eigen::VectorXd actualMatrix;
   read_dense(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense(const std::string& fname, Eigen::VectorXd &). .csv")
{
   Eigen::VectorXd expectedMatrix(3);
   expectedMatrix(0) = 1;
   expectedMatrix(1) = 4;
   expectedMatrix(2) = 9;

   std::string matrixFilename = "write_read_dense2.csv";
   write_dense(matrixFilename, expectedMatrix);

   Eigen::VectorXd actualMatrix;
   read_dense(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense(std::istream& in, DenseMatrixType denseMatrixType, Eigen::VectorXd& V). DenseMatrixType::ddm")
{
   Eigen::VectorXd expectedMatrix(3);
   expectedMatrix(0) = 1;
   expectedMatrix(1) = 4;
   expectedMatrix(2) = 9;

   std::stringstream matrixStream;
   write_dense(matrixStream, DenseMatrixType::ddm, expectedMatrix);

   Eigen::VectorXd actualMatrix;
   read_dense(matrixStream, DenseMatrixType::ddm, actualMatrix);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense(std::istream& in, DenseMatrixType denseMatrixType, Eigen::VectorXd& V). DenseMatrixType::csv")
{
   Eigen::VectorXd expectedMatrix(3);
   expectedMatrix(0) = 1;
   expectedMatrix(1) = 4;
   expectedMatrix(2) = 9;

   std::stringstream matrixStream;
   write_dense(matrixStream, DenseMatrixType::csv, expectedMatrix);

   Eigen::VectorXd actualMatrix;
   read_dense(matrixStream, DenseMatrixType::csv, actualMatrix);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse(const std::string& fname, Eigen::SparseMatrix<double> &). .sdm")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3; std::uint64_t matrixNNZ = 6;
   std::vector<std::uint32_t> matrixRows    = { 1, 1, 1, 3, 3, 3 };
   std::vector<std::uint32_t> matrixCols    = { 1, 2, 3, 1, 2, 3 };
   std::vector<double> matrixVals = { 1, 2, 3, 7, 8, 9 };
   std::string matrixFilename = "read_sparse1.sdm";
   std::ofstream matrixFile(matrixFilename);
   matrixFile.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNNZ), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(matrixRows.data()), matrixRows.size() * sizeof(std::uint32_t));
   matrixFile.write(reinterpret_cast<char*>(matrixCols.data()), matrixCols.size() * sizeof(std::uint32_t));
   matrixFile.write(reinterpret_cast<char*>(matrixVals.data()), matrixVals.size() * sizeof(double));
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   Eigen::SparseMatrix<double> actualMatrix;
   read_sparse(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse(const std::string& fname, Eigen::SparseMatrix<double> &). .sbm")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3; std::uint64_t matrixNNZ = 6;
   std::vector<int> matrixRows = { 1, 1, 1, 3, 3, 3 };
   std::vector<int> matrixCols = { 1, 2, 3, 1, 2, 3 };
   std::string matrixFilename = "read_sparse1.sbm";
   std::ofstream matrixFile(matrixFilename);
   matrixFile.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNNZ), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(matrixRows.data()), matrixRows.size() * sizeof(std::uint32_t));
   matrixFile.write(reinterpret_cast<char*>(matrixCols.data()), matrixCols.size() * sizeof(std::uint32_t));
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 1 },
      { 0, 2, 1 },
      { 2, 0, 1 },
      { 2, 1, 1 },
      { 2, 2, 1 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   Eigen::SparseMatrix<double> actualMatrix;
   read_sparse(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse(const std::string& fname, Eigen::SparseMatrix<double> &). .mtx")
{
   std::string matrixFilename = "read_sparse1.mtx";
   std::ofstream matrixFile(matrixFilename);
   matrixFile << 3 << '\t' << 3 << '\t' << 6 << std::endl;
   matrixFile << 1 << '\t' << 1 << '\t' << 1 << std::endl;
   matrixFile << 1 << '\t' << 2 << '\t' << 2 << std::endl;
   matrixFile << 1 << '\t' << 3 << '\t' << 3 << std::endl;
   matrixFile << 3 << '\t' << 1 << '\t' << 7 << std::endl;
   matrixFile << 3 << '\t' << 2 << '\t' << 8 << std::endl;
   matrixFile << 3 << '\t' << 3 << '\t' << 9 << std::endl;
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   Eigen::SparseMatrix<double> actualMatrix;
   read_sparse(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_mtx(const std::string& filename)")
{
   std::string matrixFilename = "read_mtx.mtx";
   std::ofstream matrixFile(matrixFilename);
   matrixFile << 3 << '\t' << 3 << '\t' << 6 << std::endl;
   matrixFile << 1 << '\t' << 1 << '\t' << 1 << std::endl;
   matrixFile << 1 << '\t' << 2 << '\t' << 2 << std::endl;
   matrixFile << 1 << '\t' << 3 << '\t' << 3 << std::endl;
   matrixFile << 3 << '\t' << 1 << '\t' << 7 << std::endl;
   matrixFile << 3 << '\t' << 2 << '\t' << 8 << std::endl;
   matrixFile << 3 << '\t' << 3 << '\t' << 9 << std::endl;
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   MatrixConfig actualMatrixConfig = read_mtx(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_mtx(std::istream& in)")
{
   std::stringstream matrixStream;
   matrixStream << 3 << '\t' << 3 << '\t' << 6 << std::endl;
   matrixStream << 1 << '\t' << 1 << '\t' << 1 << std::endl;
   matrixStream << 1 << '\t' << 2 << '\t' << 2 << std::endl;
   matrixStream << 1 << '\t' << 3 << '\t' << 3 << std::endl;
   matrixStream << 3 << '\t' << 1 << '\t' << 7 << std::endl;
   matrixStream << 3 << '\t' << 2 << '\t' << 8 << std::endl;
   matrixStream << 3 << '\t' << 3 << '\t' << 9 << std::endl;

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   MatrixConfig actualMatrixConfig = read_mtx(matrixStream);
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_csv(const std::string& filename)")
{
   std::string matrixFilename = "read_csv.csv";
   std::ofstream matrixFile(matrixFilename);
   matrixFile << 3 << std::endl;
   matrixFile << 3 << std::endl;
   matrixFile << "1,2,3" << std::endl;
   matrixFile << "4,5,6" << std::endl;
   matrixFile << "7,8,9" << std::endl;
   matrixFile.close();

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_csv(matrixFilename);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_csv(std::istream& in)")
{
   std::stringstream matrixStream;
   matrixStream << 3 << std::endl;
   matrixStream << 3 << std::endl;
   matrixStream << "1,2,3" << std::endl;
   matrixStream << "4,5,6" << std::endl;
   matrixStream << "7,8,9" << std::endl;

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_csv(matrixStream);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_ddm(const std::string& filename)")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3;
   std::vector<double> matrixVals = { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
   std::string matrixFilename = "read_ddm.csv";
   std::ofstream matrixFile(matrixFilename);
   matrixFile.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(matrixVals.data()), matrixVals.size() * sizeof(double));
   matrixFile.close();

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_ddm(matrixFilename);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_ddm(std::istream& in)")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3;
   std::vector<double> matrixVals = { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
   std::stringstream matrixStream;
   matrixStream.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixStream.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixStream.write(reinterpret_cast<char*>(matrixVals.data()), matrixVals.size() * sizeof(double));

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_ddm(matrixStream);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense(const std::string& fname). .ddm")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3;
   std::vector<double> matrixVals = { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
   std::string matrixFilename = "read_dense.ddm";
   std::ofstream matrixFile(matrixFilename);
   matrixFile.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(matrixVals.data()), matrixVals.size() * sizeof(double));
   matrixFile.close();

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_dense(matrixFilename);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense(const std::string& fname). .csv")
{
   std::string matrixFilename = "read_dense.csv";
   std::ofstream matrixFile(matrixFilename);
   matrixFile << 3 << std::endl;
   matrixFile << 3 << std::endl;
   matrixFile << "1,2,3" << std::endl;
   matrixFile << "4,5,6" << std::endl;
   matrixFile << "7,8,9" << std::endl;
   matrixFile.close();

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_dense(matrixFilename);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense(std::istream& in, DenseMatrixType denseMatrixType). DenseMatrixType::ddm")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3;
   std::vector<double> matrixVals = { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
   std::stringstream matrixStream;
   matrixStream.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixStream.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixStream.write(reinterpret_cast<char*>(matrixVals.data()), matrixVals.size() * sizeof(double));

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_dense(matrixStream, DenseMatrixType::ddm);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense(std::istream& in, DenseMatrixType denseMatrixType). DenseMatrixType::csv")
{
   std::stringstream matrixStream;
   matrixStream << 3 << std::endl;
   matrixStream << 3 << std::endl;
   matrixStream << "1,2,3" << std::endl;
   matrixStream << "4,5,6" << std::endl;
   matrixStream << "7,8,9" << std::endl;

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_dense(matrixStream, DenseMatrixType::csv);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse(const std::string& fname). .sdm")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3; std::uint64_t matrixNNZ = 6;
   std::vector<std::uint32_t> matrixRows    = { 1, 1, 1, 3, 3, 3 };
   std::vector<std::uint32_t> matrixCols    = { 1, 2, 3, 1, 2, 3 };
   std::vector<double> matrixVals = { 1, 2, 3, 7, 8, 9 };
   std::string matrixFilename = "read_sparse2.sdm";
   std::ofstream matrixFile(matrixFilename);
   matrixFile.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNNZ), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(matrixRows.data()), matrixRows.size() * sizeof(std::uint32_t));
   matrixFile.write(reinterpret_cast<char*>(matrixCols.data()), matrixCols.size() * sizeof(std::uint32_t));
   matrixFile.write(reinterpret_cast<char*>(matrixVals.data()), matrixVals.size() * sizeof(double));
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   MatrixConfig actualMatrixConfig = read_sparse(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse(const std::string& fname). .sbm")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3; std::uint64_t matrixNNZ = 6;
   std::vector<std::uint32_t> matrixRows    = { 1, 1, 1, 3, 3, 3 };
   std::vector<std::uint32_t> matrixCols    = { 1, 2, 3, 1, 2, 3 };
   std::string matrixFilename = "read_sparse2.sbm";
   std::ofstream matrixFile(matrixFilename);
   matrixFile.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNNZ), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(matrixRows.data()), matrixRows.size() * sizeof(std::uint32_t));
   matrixFile.write(reinterpret_cast<char*>(matrixCols.data()), matrixCols.size() * sizeof(std::uint32_t));
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 1 },
      { 0, 2, 1 },
      { 2, 0, 1 },
      { 2, 1, 1 },
      { 2, 2, 1 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   MatrixConfig actualMatrixConfig = read_sparse(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse(const std::string& fname). .mtx")
{
   std::string matrixFilename = "read_sparse2.mtx";
   std::ofstream matrixFile(matrixFilename);
   matrixFile << 3 << '\t' << 3 << '\t' << 6 << std::endl;
   matrixFile << 1 << '\t' << 1 << '\t' << 1 << std::endl;
   matrixFile << 1 << '\t' << 2 << '\t' << 2 << std::endl;
   matrixFile << 1 << '\t' << 3 << '\t' << 3 << std::endl;
   matrixFile << 3 << '\t' << 1 << '\t' << 7 << std::endl;
   matrixFile << 3 << '\t' << 2 << '\t' << 8 << std::endl;
   matrixFile << 3 << '\t' << 3 << '\t' << 9 << std::endl;
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   MatrixConfig actualMatrixConfig = read_sparse(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix(const std::string& fname). .ddm")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3;
   std::vector<double> matrixVals = { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
   std::string matrixFilename = "read_matrix.ddm";
   std::ofstream matrixFile(matrixFilename);
   matrixFile.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(matrixVals.data()), matrixVals.size() * sizeof(double));
   matrixFile.close();

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_matrix(matrixFilename);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix(const std::string& fname). .csv")
{
   std::string matrixFilename = "read_matrix.csv";
   std::ofstream matrixFile(matrixFilename);
   matrixFile << 3 << std::endl;
   matrixFile << 3 << std::endl;
   matrixFile << "1,2,3" << std::endl;
   matrixFile << "4,5,6" << std::endl;
   matrixFile << "7,8,9" << std::endl;
   matrixFile.close();

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MatrixConfig actualMatrixConfig = read_matrix(matrixFilename);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix(const std::string& fname). .sdm")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3; std::uint64_t matrixNNZ = 6;
   std::vector<std::uint32_t> matrixRows    = { 1, 1, 1, 3, 3, 3 };
   std::vector<std::uint32_t> matrixCols    = { 1, 2, 3, 1, 2, 3 };
   std::vector<double> matrixVals = { 1, 2, 3, 7, 8, 9 };
   std::string matrixFilename = "read_matrix.sdm";
   std::ofstream matrixFile(matrixFilename);
   matrixFile.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNNZ), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(matrixRows.data()), matrixRows.size() * sizeof(std::uint32_t));
   matrixFile.write(reinterpret_cast<char*>(matrixCols.data()), matrixCols.size() * sizeof(std::uint32_t));
   matrixFile.write(reinterpret_cast<char*>(matrixVals.data()), matrixVals.size() * sizeof(double));
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   MatrixConfig actualMatrixConfig = read_matrix(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix(const std::string& fname). .sbm")
{
   std::uint64_t matrixNRow = 3; std::uint64_t matrixNCol = 3; std::uint64_t matrixNNZ = 6;
   std::vector<std::uint32_t> matrixRows    = { 1, 1, 1, 3, 3, 3 };
   std::vector<std::uint32_t> matrixCols    = { 1, 2, 3, 1, 2, 3 };
   std::string matrixFilename = "read_matrix.sbm";
   std::ofstream matrixFile(matrixFilename);
   matrixFile.write(reinterpret_cast<char*>(&matrixNRow), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNCol), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(&matrixNNZ), sizeof(std::uint64_t));
   matrixFile.write(reinterpret_cast<char*>(matrixRows.data()), matrixRows.size() * sizeof(std::uint32_t));
   matrixFile.write(reinterpret_cast<char*>(matrixCols.data()), matrixCols.size() * sizeof(std::uint32_t));
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 1 },
      { 0, 2, 1 },
      { 2, 0, 1 },
      { 2, 1, 1 },
      { 2, 2, 1 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   MatrixConfig actualMatrixConfig = read_matrix(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix(const std::string& fname). .mtx")
{
   std::string matrixFilename = "read_matrix.mtx";
   std::ofstream matrixFile(matrixFilename);
   matrixFile << 3 << '\t' << 3 << '\t' << 6 << std::endl;
   matrixFile << 1 << '\t' << 1 << '\t' << 1 << std::endl;
   matrixFile << 1 << '\t' << 2 << '\t' << 2 << std::endl;
   matrixFile << 1 << '\t' << 3 << '\t' << 3 << std::endl;
   matrixFile << 3 << '\t' << 1 << '\t' << 7 << std::endl;
   matrixFile << 3 << '\t' << 2 << '\t' << 8 << std::endl;
   matrixFile << 3 << '\t' << 3 << '\t' << 9 << std::endl;
   matrixFile.close();

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 }
   };
   Eigen::SparseMatrix<double> expectedMatrix(3, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   MatrixConfig actualMatrixConfig = read_matrix(matrixFilename);
   Eigen::SparseMatrix<double> actualMatrix = sparse_to_eigen(actualMatrixConfig);

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense_float64(const std::string& filename) | write_dense_float64(const std::string& filename, const smurff::MatrixConfig& Y)")
{
   std::string matrixFilename = "dense_float64.ddm";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 3;
   std::vector<double> matrixConfigValues = { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
   MatrixConfig matrixConfig(matrixConfigNRow, matrixConfigNCol, std::move(matrixConfigValues), NoiseConfig());

   write_dense_float64(matrixFilename, matrixConfig);
   MatrixConfig actualMatrixConfig = read_dense_float64(matrixFilename);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_dense_float64(std::istream& in) | write_dense_float64(std::ostream& out, const smurff::MatrixConfig& Y)")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 3;
   std::vector<double> matrixConfigValues = { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
   MatrixConfig matrixConfig(matrixConfigNRow, matrixConfigNCol, std::move(matrixConfigValues), NoiseConfig());

   std::stringstream matrixConfigStream;
   write_dense_float64(matrixConfigStream, matrixConfig);

   MatrixConfig actualMatrixConfig = read_dense_float64(matrixConfigStream);
   Eigen::MatrixXd actualMatrix = dense_to_eigen(actualMatrixConfig);

   Eigen::MatrixXd expectedMatrix(3, 3);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse_float64(const std::string& filename) | write_sparse_float64(const std::string& filename, const smurff::MatrixConfig& Y)")
{
   std::string matrixFilename = "sparse_float64.sdm";

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

   write_sparse_float64(matrixFilename, matrixConfig);

   MatrixConfig actualMatrixConfig = read_sparse_float64(matrixFilename);
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

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse_float64(std::istream& in) | write_sparse_float64(std::ostream& out, const smurff::MatrixConfig& Y)")
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

   std::stringstream matrixConfigStream;
   write_sparse_float64(matrixConfigStream, matrixConfig);

   MatrixConfig actualMatrixConfig = read_sparse_float64(matrixConfigStream);
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

TEST_CASE("matrix_io/read_sparse_binary_matrix(const std::string& filename) | write_sparse_binary_matrix(const std::string& filename, const smurff::MatrixConfig& Y)")
{
   std::string matrixFilename = "sparse_binary_matrix.sbm";

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

   write_sparse_binary_matrix(matrixFilename, matrixConfig);

   MatrixConfig actualMatrixConfig = read_sparse_binary_matrix(matrixFilename);
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

   std::remove(matrixFilename.c_str());
   REQUIRE(actualMatrix.isApprox(expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse_binary_matrix(std::istream& in) | write_sparse_binary_matrix(std::ostream& out, const smurff::MatrixConfig& Y)")
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

   std::stringstream matrixConfigStream;
   write_sparse_binary_matrix(matrixConfigStream, matrixConfig);

   MatrixConfig actualMatrixConfig = read_sparse_binary_matrix(matrixConfigStream);
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