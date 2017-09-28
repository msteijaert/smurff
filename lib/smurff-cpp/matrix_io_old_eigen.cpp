#include "matrix_io_old_eigen.h"

const static Eigen::IOFormat csvFormat(6, Eigen::DontAlignCols, ",", "\n");

void writeToCSVfile(const std::string& filename, const Eigen::MatrixXd& matrix)
{
  std::ofstream file(filename.c_str());
  writeToCSVstream(file, matrix);
}

void writeToCSVstream(std::ostream& out, const Eigen::MatrixXd& matrix)
{
   out << matrix.rows() << std::endl;
   out << matrix.cols() << std::endl;
   out << matrix.format(csvFormat) << std::endl;
}

void readFromCSVfile(const std::string& filename, Eigen::MatrixXd &matrix)
{
    std::ifstream file(filename.c_str());
    readFromCSVstream(file, matrix);
}

void readFromCSVstream(std::istream& in, Eigen::MatrixXd& matrix)
{
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
}

void write_ddm(const std::string& filename, const Eigen::MatrixXd& matrix)
{
   std::ofstream out(filename,std::ios::out | std::ios::binary | std::ios::trunc);
   write_ddm(out, matrix);
   out.close();
}

void write_ddm(std::ostream& out, const Eigen::MatrixXd& matrix)
{
   std::uint64_t rows = matrix.rows();
   std::uint64_t cols = matrix.cols();
   out.write((char*) (&rows), sizeof(std::uint64_t));
   out.write((char*) (&cols), sizeof(std::uint64_t));
   out.write((char*) matrix.data(), rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );
}

void read_ddm(const std::string& filename, Eigen::MatrixXd& matrix)
{
    std::ifstream in(filename,std::ios::in | std::ios::binary);
    read_ddm(in, matrix);
    in.close();
}

void read_ddm(std::istream& in, Eigen::MatrixXd& matrix)
{
   std::uint64_t rows=0, cols=0;
   in.read((char*) (&rows),sizeof(std::uint64_t));
   in.read((char*) (&cols),sizeof(std::uint64_t));
   matrix.resize(rows, cols);
   in.read( (char *) matrix.data() , rows*cols*sizeof(double) );
}

void read_dense(const std::string& fname, Eigen::VectorXd& V)
{
   die_unless_file_exists(fname);
   DenseMatrixType denseMatrixType = ExtensionToDenseMatrixType(fname);
   std::ifstream in(fname, DenseMatrixTypeToOpenMode(denseMatrixType));
   read_dense(in, denseMatrixType, V);
}

void read_dense(std::istream& in, DenseMatrixType denseMatrixType, Eigen::VectorXd& V)
{
   Eigen::MatrixXd X;
   read_dense(in, denseMatrixType, X);
   V = X; // this will fail if X has more than one column
}

void read_dense(const std::string& fname, Eigen::MatrixXd& X)
{
   die_unless_file_exists(fname);
   DenseMatrixType denseMatrixType = ExtensionToDenseMatrixType(fname);
   std::ifstream in(fname, DenseMatrixTypeToOpenMode(denseMatrixType));
   read_dense(in, denseMatrixType, X);
}

void read_dense(std::istream& in, DenseMatrixType denseMatrixType, Eigen::MatrixXd& X)
{
   switch (denseMatrixType)
   {
      case DenseMatrixType::ddm:
         read_ddm(in, X);
         break;
      case DenseMatrixType::csv:
         readFromCSVstream(in, X);
         break;
      default:
         die("Unknown matrix type");
   }
}

void read_sparse(const std::string& fname, Eigen::SparseMatrix<double>& M)
{
   assert(is_sparse_fname(fname));
   SparseMatrixType sparseMatrixType = ExtensionToSparseMatrixType(fname);
   switch (sparseMatrixType)
   {
      case SparseMatrixType::sdm:
         {
            auto sdm_ptr = read_sdm(fname.c_str());
            M = sparse_to_eigen(*sdm_ptr);
            free_sdm(sdm_ptr);
            delete sdm_ptr;
            break;
         }
      case SparseMatrixType::sbm:
         {
            auto sbm_ptr = read_sbm(fname.c_str());
            M = sparse_to_eigen(*sbm_ptr);
            free_sbm(sbm_ptr);
            delete sbm_ptr;
            break;
         }
      case SparseMatrixType::mtx:
         {
            loadMarket(M, fname.c_str());
            break;
         }
      default:
         die("Unknown filename in read_sparse: " + fname);
   }
}

void write_dense(const std::string& fname, const Eigen::MatrixXd &X)
{
   assert(is_dense_fname(fname));
   DenseMatrixType denseMatrixType = ExtensionToDenseMatrixType(fname);
   std::ofstream out(fname, DenseMatrixTypeToOpenMode(denseMatrixType));
   write_dense(out, denseMatrixType, X);
}

void write_dense(std::ostream& out, DenseMatrixType denseMatrixType, const Eigen::MatrixXd &X)
{
   switch (denseMatrixType)
   {
      case DenseMatrixType::ddm:
         write_ddm(out, X);
         break;
      case DenseMatrixType::csv:
         writeToCSVstream(out, X);
         break;
      default:
         die("Unknown matrix type");
   }
}