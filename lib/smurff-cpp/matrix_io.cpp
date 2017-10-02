#include <iostream>
#include <sstream>

#include <unsupported/Eigen/SparseExtra>

#include "utils.h"
#include "matrix_io.h"

using namespace smurff;

#define EXTENSION_SDM ".sdm"
#define EXTENSION_SBM ".sbm"
#define EXTENSION_MTX ".mtx"
#define EXTENSION_MM  ".mm"
#define EXTENSION_CSV ".csv"
#define EXTENSION_DDM ".ddm"

matrix_io::MatrixType ExtensionToMatrixType(const std::string& fname)
{
   std::string extension = fname.substr(fname.find_last_of("."));
   if (extension == EXTENSION_SDM)
   {
      return matrix_io::MatrixType::sdm;
   }
   else if (extension == EXTENSION_SBM)
   {
      return matrix_io::MatrixType::sbm;
   }
   else if (extension == EXTENSION_MTX || extension == EXTENSION_MM)
   {
       return matrix_io::MatrixType::mtx;
   }
   else if (extension == EXTENSION_CSV)
   {
      return matrix_io::MatrixType::csv;
   }
   else if (extension == EXTENSION_DDM)
   {
      return matrix_io::MatrixType::ddm;
   }
   else
   {
      throw "Unknown file type: " + extension;
   }
   return matrix_io::MatrixType::none;
}

std::string MatrixTypeToExtension(matrix_io::MatrixType matrixType)
{
   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      return EXTENSION_SDM;
   case matrix_io::MatrixType::sbm:
      return EXTENSION_SBM;
   case matrix_io::MatrixType::mtx:
      return EXTENSION_MTX;
   case matrix_io::MatrixType::csv:
      return EXTENSION_CSV;
   case matrix_io::MatrixType::ddm:
      return EXTENSION_DDM;
   case matrix_io::MatrixType::none:
      throw "Unknown matrix type";
   default:
      throw "Unknown matrix type";
   }
   return std::string();
}

MatrixConfig matrix_io::read_matrix(const std::string& filename)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);

   die_unless_file_exists(filename);

   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         return matrix_io::read_sparse_float64_bin(fileStream);
      }
   case matrix_io::MatrixType::sbm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         return matrix_io::read_sparse_binary_bin(fileStream);
      }
   case matrix_io::MatrixType::mtx:
      {
         std::ifstream fileStream(filename);
         return matrix_io::read_sparse_float64_mtx(fileStream);
      }
   case matrix_io::MatrixType::csv:
      {
         std::ifstream fileStream(filename);
         return matrix_io::read_dense_float64_csv(fileStream);
      }
   case matrix_io::MatrixType::ddm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         return matrix_io::read_dense_float64_bin(fileStream);
      }
   case matrix_io::MatrixType::none:
      throw "Unknown matrix type";
   default:
      throw "Unknown matrix type";
   }
}

MatrixConfig matrix_io::read_dense_float64_bin(std::istream& in)
{
   std::uint64_t nrow;
   std::uint64_t ncol;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));

   std::vector<double> values(nrow * ncol);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   return smurff::MatrixConfig(nrow, ncol, std::move(values), smurff::NoiseConfig());
}

MatrixConfig matrix_io::read_dense_float64_csv(std::istream& in)
{
   std::stringstream ss;
   std::string line;

   // rows and cols
   getline(in, line); 
   ss.clear();
   ss << line;
   std::uint64_t nrow;
   ss >> nrow;

   getline(in, line);
   ss.clear();
   ss << line;
   std::uint64_t ncol;
   ss >> ncol;

   std::uint64_t nnz = nrow * ncol;

   std::vector<double> values;
   values.resize(nnz);

   std::uint64_t row = 0;
   std::uint64_t col = 0;

   while(getline(in, line) && row < nrow)
   {
      col = 0;
      
      std::stringstream lineStream(line);
      std::string cell;

      while (std::getline(lineStream, cell, ',') && col < ncol)
      {
         values[(nrow * col++) + row] = stod(cell);
      }

      row++;
   }

   assert(row == nrow);
   assert(col == ncol);

   return smurff::MatrixConfig(nrow, ncol, values, smurff::NoiseConfig());
}

MatrixConfig matrix_io::read_sparse_float64_bin(std::istream& in)
{
   std::uint64_t nrow;
   std::uint64_t ncol;
   std::uint64_t nnz;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   std::vector<std::uint32_t> rows(nnz);
   in.read(reinterpret_cast<char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row--; });

   std::vector<std::uint32_t> cols(nnz);
   in.read(reinterpret_cast<char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col--; });

   std::vector<double> values(nnz);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   return smurff::MatrixConfig(nrow, ncol, std::move(rows), std::move(cols), std::move(values), smurff::NoiseConfig());
}

MatrixConfig matrix_io::read_sparse_float64_mtx(std::istream& in)
{
   // Ignore headers and comments:
   while (in.peek() == '%') in.ignore(2048, '\n');

   // Read defining parameters:
   std::uint64_t nrow;
   std::uint64_t ncol;
   std::uint64_t nnz;
   in >> nrow >> ncol >> nnz;
   in.ignore(2048, '\n'); // skip to end of line

   std::vector<std::uint32_t> rows(nnz);
   std::vector<std::uint32_t> cols(nnz);
   std::vector<double> values(nnz);

   // Read the data
   char line[2048];
   std::uint32_t r,c;
   double v;
   for (std::uint64_t l = 0; l < nnz; l++)
   {
      in.getline(line, 2048);
      std::stringstream ls(line);
      ls >> r >> c;
      assert(!ls.fail());

      ls >> v;
      if (ls.fail()) 
         v = 1.0;

      r--;
      c--;

      assert(r < nrow);
      assert(r >= 0);
      assert(c < ncol);
      assert(c >= 0);

      rows[l] = r;
      cols[l] = c;
      values[l] = v;
   }

   return smurff::MatrixConfig(nrow, ncol, rows, cols, values, smurff::NoiseConfig());
}

MatrixConfig matrix_io::read_sparse_binary_bin(std::istream& in)
{
   std::uint64_t nrow;
   std::uint64_t ncol;
   std::uint64_t nnz;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   std::vector<std::uint32_t> rows(nnz);
   in.read(reinterpret_cast<char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row--; });

   std::vector<std::uint32_t> cols(nnz);
   in.read(reinterpret_cast<char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col--; });

   return smurff::MatrixConfig(nrow, ncol, std::move(rows), std::move(cols), smurff::NoiseConfig());
}

// ======================================================================================================

void matrix_io::write_matrix(const std::string& filename, const MatrixConfig& matrixConfig)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         matrix_io::write_sparse_float64_bin(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::sbm:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         matrix_io::write_sparse_binary_bin(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::mtx:
      {
         std::ofstream fileStream(filename);
         matrix_io::write_sparse_float64_mtx(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::csv:
      {
         std::ofstream fileStream(filename);
         matrix_io::write_dense_float64_csv(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::ddm:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         matrix_io::write_dense_float64_bin(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::none:
      throw "Unknown matrix type";
   default:
      throw "Unknown matrix type";
   }
}

void matrix_io::write_dense_float64_bin(std::ostream& out, const MatrixConfig& matrixConfig)
{
   std::uint64_t nrow = matrixConfig.getNRow();
   std::uint64_t ncol = matrixConfig.getNCol();
   const std::vector<double>& values = matrixConfig.getValues();

   out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void matrix_io::write_dense_float64_csv(std::ostream& out, const MatrixConfig& matrixConfig)
{
   //write rows and cols
   std::uint64_t nrow = matrixConfig.getNRow();
   std::uint64_t ncol = matrixConfig.getNCol();
   std::uint64_t nnz = nrow * ncol;

   out << nrow << std::endl;
   out << ncol << std::endl;
   
   const std::vector<double>& values = matrixConfig.getValues();

   assert(values.size() == nnz);

   //write values
   for(std::uint64_t i = 0; i < nrow; i++)
   {
      for(std::uint64_t j = 0; j < ncol; j++)
      {
         if(j == ncol - 1)
            out << values[j * nrow + i];
         else
            out << values[j * nrow + i] << ",";
      }
      out << std::endl;
   }
}

void matrix_io::write_sparse_float64_bin(std::ostream& out, const MatrixConfig& matrixConfig)
{
   std::uint64_t nrow = matrixConfig.getNRow();
   std::uint64_t ncol = matrixConfig.getNCol();
   std::uint64_t nnz = matrixConfig.getNNZ();

   //get values copy
   std::vector<std::uint32_t> rows = matrixConfig.getRows();
   std::vector<std::uint32_t> cols = matrixConfig.getCols();
   std::vector<double> values = matrixConfig.getValues();

   //increment coordinates
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row++; });
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col++; });

   out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void matrix_io::write_sparse_float64_mtx(std::ostream& out, const MatrixConfig& matrixConfig)
{
   std::uint64_t nrow = matrixConfig.getNRow();
   std::uint64_t ncol = matrixConfig.getNCol();
   std::uint64_t nnz = matrixConfig.getNNZ();

   //write row col nnz
   out << nrow << "\t" << ncol << "\t" << nnz << std::endl;

   const std::vector<std::uint32_t>& rows = matrixConfig.getRows();
   const std::vector<std::uint32_t>& cols = matrixConfig.getCols();
   const std::vector<double>& values = matrixConfig.getValues();
   
   //write values
   for(std::uint64_t i = 0; i < nnz; i++)
   {
      out << rows[i] + 1 << "\t" << cols[i] + 1 << "\t" << values[i] << std::endl;
   }
}

void matrix_io::write_sparse_binary_bin(std::ostream& out, const MatrixConfig& matrixConfig)
{
   std::uint64_t nrow = matrixConfig.getNRow();
   std::uint64_t ncol = matrixConfig.getNCol();
   std::uint64_t nnz = matrixConfig.getNNZ();

   //get values copy
   std::vector<std::uint32_t> rows = matrixConfig.getRows();
   std::vector<std::uint32_t> cols = matrixConfig.getCols();

   //increment coordinates
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row++; });
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col++; });

   out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
}

// ======================================================================================================

void matrix_io::eigen::read_matrix(const std::string& filename, Eigen::VectorXd& V)
{
   Eigen::MatrixXd X;
   matrix_io::eigen::read_matrix(filename, X);
   V = X; // this will fail if X has more than one column
}

void matrix_io::eigen::read_matrix(const std::string& filename, Eigen::MatrixXd& X)
{
   matrix_io::MatrixType matrixType = ExtensionToMatrixType(filename);

   die_unless_file_exists(filename);

   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::sbm:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::mtx:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::csv:
      {
         std::ifstream fileStream(filename);
         matrix_io::eigen::read_dense_float64_csv(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::ddm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         matrix_io::eigen::read_dense_float64_bin(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::none:
      throw "Unknown matrix type";
   default:
      throw "Unknown matrix type";
   }
}

void matrix_io::eigen::read_matrix(const std::string& filename, Eigen::SparseMatrix<double>& X)
{
   matrix_io::MatrixType matrixType = ExtensionToMatrixType(filename);

   die_unless_file_exists(filename);

   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         matrix_io::eigen::read_sparse_float64_bin(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::sbm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         matrix_io::eigen::read_sparse_binary_bin(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::mtx:
      {
         std::ifstream fileStream(filename);
         matrix_io::eigen::read_sparse_float64_mtx(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::csv:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::ddm:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::none:
      throw "Unknown matrix type";
   default:
      throw "Unknown matrix type";
   }
}

void matrix_io::eigen::read_dense_float64_bin(std::istream& in, Eigen::MatrixXd& X)
{
   /*
   std::uint64_t rows=0, cols=0;
   in.read((char*) (&rows),sizeof(std::uint64_t));
   in.read((char*) (&cols),sizeof(std::uint64_t));
   matrix.resize(rows, cols);
   in.read( (char *) matrix.data() , rows*cols*sizeof(double) );
   */

  throw "Not implemented";
}

void matrix_io::eigen::read_dense_float64_csv(std::istream& in, Eigen::MatrixXd& X)
{
   /*
   std::string line;

   // rows and cols
   getline(in, line);
   std::uint64_t nrow = atol(line.c_str());
   getline(in, line);
   std::uint64_t ncol = atol(line.c_str());
   matrix.resize(nrow, ncol);

   std::uint32_t row = 0;
   std::uint32_t col = 0;
   while (getline(in, line))
   {
      col = 0;
      std::stringstream lineStream(line);
      std::string cell;
      while (std::getline(lineStream, cell, ','))
      {
         matrix(row, col++) = strtod(cell.c_str(), NULL);
      }
      row++;
   }
   assert(row == nrow);
   assert(col == ncol);
   */

  throw "Not implemented";
}

void matrix_io::eigen::read_sparse_float64_bin(std::istream& in, Eigen::SparseMatrix<double>& X)
{
   //we need to use our functions instead of libfastsparse
   /*
   auto sdm_ptr = read_sdm(fname.c_str());
   M = sparse_to_eigen(*sdm_ptr);
   free_sdm(sdm_ptr);
   delete sdm_ptr;
   */

   throw "Not implemented";
}

void matrix_io::eigen::read_sparse_float64_mtx(std::istream& in, Eigen::SparseMatrix<double>& X)
{
   //we need to use our functions instead of libfastsparse
   /*
   loadMarket(M, fname.c_str());
   */

   throw "Not implemented";
}

void matrix_io::eigen::read_sparse_binary_bin(std::istream& in, Eigen::SparseMatrix<double>& X)
{
   //we need to use our functions instead of libfastsparse
   /*
   auto sbm_ptr = read_sbm(fname.c_str());
   M = sparse_to_eigen(*sbm_ptr);
   free_sbm(sbm_ptr);
   delete sbm_ptr;
   */

   throw "Not implemented";
}

// ======================================================================================================

void matrix_io::eigen::write_matrix(const std::string& filename, const Eigen::MatrixXd& X)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::sbm:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::mtx:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::csv:
      {
         std::ofstream fileStream(filename);
         matrix_io::eigen::write_dense_float64_csv(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::ddm:
      {
         std::ofstream fileStream(filename, std::ios_base::binary | std::ios::trunc);
         matrix_io::eigen::write_dense_float64_bin(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::none:
      throw "Unknown matrix type";
   default:
      throw "Unknown matrix type";
   }
}

void matrix_io::eigen::write_matrix(const std::string& filename, const Eigen::SparseMatrix<double>& X)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         matrix_io::eigen::write_sparse_float64_bin(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::sbm:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         matrix_io::eigen::write_sparse_binary_bin(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::mtx:
      {
         std::ofstream fileStream(filename);
         matrix_io::eigen::write_sparse_float64_mtx(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::csv:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::ddm:
      throw "Invalid matrix type";
   case matrix_io::MatrixType::none:
      throw "Unknown matrix type";
   default:
      throw "Unknown matrix type";
   }
}

void matrix_io::eigen::write_dense_float64_bin(std::ostream& out, const Eigen::MatrixXd& X)
{
   /*
    std::uint64_t rows = matrix.rows();
   std::uint64_t cols = matrix.cols();
   out.write((char*) (&rows), sizeof(std::uint64_t));
   out.write((char*) (&cols), sizeof(std::uint64_t));
   out.write((char*) matrix.data(), rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );
   */

   throw "Not implemented";
}

const static Eigen::IOFormat csvFormat(6, Eigen::DontAlignCols, ",", "\n");

void matrix_io::eigen::write_dense_float64_csv(std::ostream& out, const Eigen::MatrixXd& X)
{
   /*
   out << matrix.rows() << std::endl;
   out << matrix.cols() << std::endl;
   out << matrix.format(csvFormat) << std::endl;
   */

   throw "Not implemented";
}

void matrix_io::eigen::write_sparse_float64_bin(std::ostream& out, const Eigen::SparseMatrix<double>& X)
{
   throw "Not implemented";
}

void matrix_io::eigen::write_sparse_float64_mtx(std::ostream& out, const Eigen::SparseMatrix<double>& X)
{
   throw "Not implemented";
}

void matrix_io::eigen::write_sparse_binary_bin(std::ostream& out, const Eigen::SparseMatrix<double>& X)
{
   throw "Not implemented";
}