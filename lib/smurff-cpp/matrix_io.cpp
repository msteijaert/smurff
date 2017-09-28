#include <set>
#include <cstdlib>

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

MatrixType ExtensionToMatrixType(const std::string& fname)
{
   std::string extension = fname.substr(fname.find_last_of("."));
   if (extension == EXTENSION_SDM)
   {
      return MatrixType::sdm;
   }
   else if (extension == EXTENSION_SBM)
   {
      return MatrixType::sbm;
   }
   else if (extension == EXTENSION_MTX || extension == EXTENSION_MM)
   {
       return MatrixType::mtx;
   }
   else if (extension == EXTENSION_CSV)
   {
      return MatrixType::csv;
   }
   else if (extension == EXTENSION_DDM)
   {
      return MatrixType::ddm;
   }
   else
   {
      die("Unknown file type: " + extension);
   }
   return MatrixType::none;
}

std::string MatrixTypeToExtension(MatrixType matrixType)
{
   switch (matrixType)
   {
   case MatrixType::sdm:
      return EXTENSION_SDM;
   case MatrixType::sbm:
      return EXTENSION_SBM;
   case MatrixType::mtx:
      return EXTENSION_MTX;
   case MatrixType::csv:
      return EXTENSION_CSV;
   case MatrixType::ddm:
      return EXTENSION_DDM;
   case MatrixType::none:
      die("Unknown matrix type");
   default:
      die("Unknown matrix type");
   }
   return std::string();
}

MatrixConfig matrix_io::read_matrix(const std::string& filename)
{
   MatrixType matrixType = ExtensionToMatrixType(fname);
   switch (matrixType)
   {
   case sdm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         return matrix_io::read_sparse_float64_bin(fileStream);
      }
   case sbm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         return matrix_io::read_sparse_binary_bin(fileStream);
      }
   case mtx:
      {
         std::ifstream fileStream(filename);
         return matrix_io::read_sparse_float64_mtx(fileStream);
      }
   case csv:
      {
         std::ifstream fileStream(filename);
         return matrix_io::read_dense_float64_csv(fileStream);
      }
   case ddm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         return matrix_io::read_dense_float64_bin(fileStream);
      }
   case none:
      die("Unknown matrix type");
   default:
      die("Unknown matrix type");
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
   std::string line;

   // rows and cols
   getline(in, line);
   std::uint64_t nrow = stol(line);
   getline(in, line);
   std::uint64_t ncol = stol(line);
   std::uint64_t nnz = nrow * ncol;
   std::vector<double> values;
   values.resize(nnz);

   std::uint32_t row = 0;
   std::uint32_t col = 0;
   while (getline(in, line))
   {
      col = 0;
      std::stringstream lineStream(line);
      std::string cell;
      while (std::getline(lineStream, cell, ','))
      {
         values[row + (nrow*col++)] = stod(cell);
      }
      row++;
   }
   assert(row == nrow);
   assert(col == ncol);

   smurff::MatrixConfig ret(nrow, ncol, values, smurff::NoiseConfig());
   return ret;
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

   std::vector<std::uint32_t> rows;
   std::vector<std::uint32_t> cols;
   std::vector<double> values;
   rows.resize(nnz);
   cols.resize(nnz);
   values.resize(nnz);

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
      if (ls.fail()) v = 1.0;

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

   smurff::MatrixConfig ret(nrow, ncol, rows, cols, values, smurff::NoiseConfig());
   return ret;
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

void matrix_io::write_matrix(const std::string& filename, const MatrixConfig& matrixConfig, MatrixType matrixType)
{
   std::string extension = MatrixTypeToExtension(matrixType);
   std::string filepath = filename + extesion;

   switch (matrixType)
   {
   case MatrixType::sdm:
      {
         std::ofstream fileStream(filepath, std::ios_base::binary);
         matrix_io::write_sparse_float64_bin(fileStream, matrixConfig);
      }
      break;
   case MatrixType::sbm:
      {
         std::ofstream fileStream(filepath, std::ios_base::binary);
         matrix_io::write_sparse_binary_bin(fileStream, matrixConfig);
      }
      break;
   case MatrixType::mtx:
      {
         std::ofstream fileStream(filepath);
         matrix_io::write_sparse_float64_mtx(fileStream, matrixConfig);
      }
      break;
   case MatrixType::csv:
      {
         std::ofstream fileStream(filepath);
         matrix_io::write_dense_float64_csv(fileStream, matrixConfig);
      }
      break;
   case MatrixType::ddm:
      {
         std::ofstream fileStream(filepath, std::ios_base::binary);
         matrix_io::write_dense_float64_bin(fileStream, matrixConfig);
      }
      break;
   case MatrixType::none:
      die("Unknown matrix type");
   default:
      die("Unknown matrix type");
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
   throw "Not implemented yet";
}

void matrix_io::write_sparse_float64_bin(std::ostream& out, const MatrixConfig& matrixConfig)
{
   std::uint64_t nrow = matrixConfig.getNRow();
   std::uint64_t ncol = matrixConfig.getNCol();
   std::uint64_t nnz = matrixConfig.getNNZ();
   std::vector<std::uint32_t> rows = matrixConfig.getRows();
   std::vector<std::uint32_t> cols = matrixConfig.getCols();
   const std::vector<double> values = matrixConfig.getValues();

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
   throw "Not implemented yet";
}

void matrix_io::write_sparse_binary_bin(std::ostream& out, const MatrixConfig& matrixConfig)
{
   std::uint64_t nrow = matrixConfig.getNRow();
   std::uint64_t ncol = matrixConfig.getNCol();
   std::uint64_t nnz = matrixConfig.getNNZ();
   std::vector<std::uint32_t> rows = matrixConfig.getRows();
   std::vector<std::uint32_t> cols = matrixConfig.getCols();

   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row++; });
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col++; });

   out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
}