#include <set>
#include <cstdlib>

#include <unsupported/Eigen/SparseExtra>

#include "utils.h"
#include "matrix_io.h"

const static Eigen::IOFormat csvFormat(6, Eigen::DontAlignCols, ",", "\n");

DenseMatrixType ExtensionToDenseMatrixType(const std::string& fname)
{
   std::string extension = fname.substr(fname.size() - 4);
   if (extension == ".ddm")
   {
      return DenseMatrixType::ddm;
   }
   else if (extension == ".csv")
   {
      return DenseMatrixType::csv;
   }
   else
   {
      die("Unknown file type: " + extension);
   }
   return DenseMatrixType::none;
}

std::ios_base::openmode DenseMatrixTypeToOpenMode(DenseMatrixType denseMatrixType)
{
   switch (denseMatrixType)
   {
      case DenseMatrixType::ddm:
         return std::ios::binary;
      case DenseMatrixType::csv:
         return std::ios_base::openmode();
      default:
         die("Unknown dense matrix type");
   }
   return std::ios_base::openmode();
}

SparseMatrixType ExtensionToSparseMatrixType(const std::string& fname)
{
   std::string extension = fname.substr(fname.find_last_of("."));
   if (extension == ".sdm")
   {
      return SparseMatrixType::sdm;
   }
   else if (extension == ".sbm")
   {
      return SparseMatrixType::sbm;
   }
   else if (extension == ".mtx" || extension == ".mm")
   {
       return SparseMatrixType::mtx;
   }
   else
   {
      die("Unknown file type: " + extension);
   }
   return SparseMatrixType::none;
}

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
   int nrow = atol(line.c_str());
   getline(in, line);
   int ncol = atol(line.c_str());
   matrix.resize(nrow, ncol);

   int row = 0;
   int col = 0;
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

smurff::MatrixConfig read_csv(const std::string& filename)
{
   std::ifstream file(filename.c_str());
   return read_csv(file);
}

smurff::MatrixConfig read_csv(std::istream& in)
{
   std::string line;

   // rows and cols
   getline(in, line);
   int nrow = stol(line);
   getline(in, line);
   int ncol = stol(line);
   int nnz = nrow * ncol;
   std::vector<double> values;
   values.resize(nnz);

   int row = 0;
   int col = 0;
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

std::unique_ptr<SparseFeat> load_bcsr(const char* filename)
{
   SparseBinaryMatrix* A = read_sbm(filename);
   SparseFeat* sf = new SparseFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols);
   free_sbm(A);
   std::unique_ptr<SparseFeat> sf_ptr(sf);
   return sf_ptr;
}

std::unique_ptr<SparseDoubleFeat> load_csr(const char* filename)
{
    struct SparseDoubleMatrix* A = read_sdm(filename);
    SparseDoubleFeat* sf = new SparseDoubleFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols, A->vals);
    delete A;
    std::unique_ptr<SparseDoubleFeat> sf_ptr(sf);
    return sf_ptr;
}

Eigen::MatrixXd sparse_to_dense(const SparseBinaryMatrix& in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = 1.;
    return out;
}

Eigen::MatrixXd sparse_to_dense(const SparseDoubleMatrix& in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = in.vals[i];
    return out;
}

static std::set<std::string> compact_matrix_fname_extensions = { ".sbm", ".sdm", ".ddm" };
static std::set<std::string> txt_matrix_fname_extensions = { ".mtx", ".mm", ".csv" };
static std::set<std::string> matrix_fname_extensions = { ".sbm", ".sdm", ".ddm", ".mtx", ".mm", ".csv" };
static std::set<std::string> sparse_fname_extensions = { ".sbm", ".sdm", ".mtx", ".mm" };

bool extension_in(const std::string& fname, const std::set<std::string>& extensions, bool die_if_not_found = false)
{
    std::string extension = fname.substr(fname.find_last_of("."));

    if (extensions.find(extension) != extensions.end())
      return true;

    if (die_if_not_found)
    {
          die("Unknown extension: " + extension + " of filename: " + fname);
    }

    return false;
}

bool is_matrix_fname(const std::string& fname)
{
    return extension_in(fname, matrix_fname_extensions);
}

bool is_sparse_fname(const std::string& fname)
{
    return extension_in(fname, sparse_fname_extensions);
}

bool is_sparse_binary_fname(const std::string& fname)
{
    return extension_in(fname, { ".sbm" });
}

bool is_dense_fname(const std::string& fname)
{
    return !is_sparse_fname(fname);
}

bool is_compact_fname(const std::string& fname)
{
    return file_exists(fname) && extension_in(fname, compact_matrix_fname_extensions);
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

smurff::MatrixConfig read_dense(const std::string& fname)
{
   die_unless_file_exists(fname);
   DenseMatrixType denseMatrixType = ExtensionToDenseMatrixType(fname);
   std::ifstream in(fname, DenseMatrixTypeToOpenMode(denseMatrixType));
   return read_dense(in, denseMatrixType);
}

smurff::MatrixConfig read_dense(std::istream& in, DenseMatrixType denseMatrixType)
{
   switch (denseMatrixType)
   {
      case DenseMatrixType::ddm:
         return read_ddm(in);
      case DenseMatrixType::csv:
         return read_csv(in);
      default:
         die("Unknown matrix type");
   }
   return smurff::MatrixConfig();
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

void write_ddm(const std::string& filename, const Eigen::MatrixXd& matrix)
{
   std::ofstream out(filename,std::ios::out | std::ios::binary | std::ios::trunc);
   write_ddm(out, matrix);
   out.close();
}

void write_ddm(std::ostream& out, const Eigen::MatrixXd& matrix)
{
   long rows = matrix.rows();
   long cols = matrix.cols();
   out.write((char*) (&rows), sizeof(long));
   out.write((char*) (&cols), sizeof(long));
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
   long rows=0, cols=0;
   in.read((char*) (&rows),sizeof(long));
   in.read((char*) (&cols),sizeof(long));
   matrix.resize(rows, cols);
   in.read( (char *) matrix.data() , rows*cols*sizeof(double) );
}

smurff::MatrixConfig read_ddm(const std::string& filename)
{
   std::ifstream in(filename,std::ios::in | std::ios::binary);
   return read_ddm(in);
}

smurff::MatrixConfig read_ddm(std::istream& in)
{
   long nrow;
   long ncol;
   in.read((char*) (&nrow),sizeof(long));
   in.read((char*) (&ncol),sizeof(long));
   int nnz = nrow * ncol;
   std::vector<double> values;
   values.resize(nnz);
   in.read( (char *) values.data(), nnz*sizeof(double) );

   smurff::MatrixConfig ret(nrow, ncol, values, smurff::NoiseConfig());
   return ret;
}

smurff::MatrixConfig read_mtx(const std::string& fname)
{
   std::ifstream fin(fname);
   return read_mtx(fin);
}

smurff::MatrixConfig read_mtx(std::istream& in)
{
   // Ignore headers and comments:
   while (in.peek() == '%') in.ignore(2048, '\n');

   // Read defining parameters:
   size_t nrow;
   size_t ncol;
   size_t nnz;
   in >> nrow >> ncol >> nnz;
   in.ignore(2048, '\n'); // skip to end of line

   std::vector<size_t> rows;
   std::vector<size_t> cols;
   std::vector<double> values;
   rows.resize(nnz);
   cols.resize(nnz);
   values.resize(nnz);

   // Read the data
   char line[2048];
   size_t r,c;
   double v;
   for (size_t l = 0; l < nnz; l++)
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

void free_sdm(SparseDoubleMatrix* sdm)
{
   free(sdm->rows);
   free(sdm->cols);
   free(sdm->vals);
}

smurff::MatrixConfig read_sparse(const std::string& fname)
{
   assert(is_sparse_fname(fname));
   SparseMatrixType sparseMatrixType = ExtensionToSparseMatrixType(fname);
   switch (sparseMatrixType)
   {
      case SparseMatrixType::sdm:
         {
            SparseDoubleMatrix* p = read_sdm(fname.c_str());
            smurff::MatrixConfig matrixConfig( p->nrow
                                             , p->ncol
                                             , std::vector<size_t>(p->rows, p->rows + p->nnz)
                                             , std::vector<size_t>(p->cols, p->cols + p->nnz)
                                             , std::vector<double>(p->vals, p->vals + p->nnz)
                                             , smurff::NoiseConfig()
                                             );
            free_sdm(p);
            free(p);
            return matrixConfig;
         }
      case SparseMatrixType::sbm:
         {
            SparseBinaryMatrix* p = read_sbm(fname.c_str());
            smurff::MatrixConfig matrixConfig( p->nrow
                                             , p->ncol
                                             , std::vector<size_t>(p->rows, p->rows + p->nnz)
                                             , std::vector<size_t>(p->cols, p->cols + p->nnz)
                                             , smurff::NoiseConfig()
                                             );
            free_sbm(p);
            free(p);
            return matrixConfig;
         }
      case SparseMatrixType::mtx:
         return read_mtx(fname);
      default:
         die("Unknown filename in read_sparse: " + fname);
   }
   return smurff::MatrixConfig();
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

smurff::MatrixConfig read_matrix(const std::string& fname)
{
   if (is_sparse_fname(fname))
      return read_sparse(fname);
   else
      return read_dense(fname);
}

template<>
Eigen::SparseMatrix<double> sparse_to_eigen<const smurff::MatrixConfig>(const smurff::MatrixConfig& matrixConfig)
{
   Eigen::SparseMatrix<double> out(matrixConfig.getNRow(), matrixConfig.getNCol());
   std::shared_ptr<std::vector<size_t> > rowsPtr = matrixConfig.getRowsPtr();
   std::shared_ptr<std::vector<size_t> > colsPtr = matrixConfig.getColsPtr();
   std::shared_ptr<std::vector<double> > valuesPtr = matrixConfig.getValuesPtr();

   std::vector<Eigen::Triplet<double> > eigenTriplets;
   for (size_t i = 0; i < matrixConfig.getNNZ(); i++)
   {
      size_t row = rowsPtr->operator[](i);
      size_t col = colsPtr->operator[](i);
      double val = matrixConfig.isBinary() ? 1.0 : valuesPtr->operator[](i);
      eigenTriplets.push_back(Eigen::Triplet<double>(row, col, val));
   }

   out.setFromTriplets(eigenTriplets.begin(), eigenTriplets.end());
   return out;
}

template<>
Eigen::SparseMatrix<double> sparse_to_eigen<smurff::MatrixConfig>(smurff::MatrixConfig& matrixConfig)
{
   return sparse_to_eigen<const smurff::MatrixConfig>(matrixConfig);
}
