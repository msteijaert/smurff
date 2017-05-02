#include <set>
#include <cstdlib>
#include "utils.h"
#include "matrix_io.h"

const static Eigen::IOFormat csvFormat(6, Eigen::DontAlignCols, ",", "\n");

void writeToCSVfile(std::string filename, Eigen::MatrixXd matrix) {
  std::ofstream file(filename.c_str());
  file << matrix.format(csvFormat);
}

void readFromCSVfile(std::string filename, Eigen::MatrixXd &matrix) {
    std::ifstream file(filename.c_str());
    std::string line;
    while (getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) matrix << strtod(cell.c_str(), NULL);
    }
}

std::unique_ptr<SparseFeat> load_bcsr(const char* filename) {
   SparseBinaryMatrix* A = read_sbm(filename);
   SparseFeat* sf = new SparseFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols);
   free_sbm(A);
   std::unique_ptr<SparseFeat> sf_ptr(sf);
   return sf_ptr;
}

std::unique_ptr<SparseDoubleFeat> load_csr(const char* filename) {
   struct SparseDoubleMatrix* A = read_sdm(filename);
   SparseDoubleFeat* sf = new SparseDoubleFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols, A->vals);
   delete A;
   std::unique_ptr<SparseDoubleFeat> sf_ptr(sf);
   return sf_ptr;
}

Eigen::MatrixXd sparse_to_dense(SparseBinaryMatrix &in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = 1.;
    return out;
}

Eigen::MatrixXd sparse_to_dense(SparseDoubleMatrix &in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = in.vals[i];
    return out;
}

static std::set<std::string> compact_matrix_file_extensions = { ".sbm", ".sdm", ".ddm" };
static std::set<std::string> txt_matrix_file_extensions = { ".mtx", ".mm", ".csv" };
static std::set<std::string> matrix_file_extensions = { ".sbm", ".sdm", ".ddm", ".mtx", ".mm", ".csv" };
static std::set<std::string> sparse_file_extensions = { ".sbm", ".sdm", ".mtx", ".mm" };

bool extension_in(std::string fname, const std::set<std::string> &extensions, bool = false);

bool is_matrix_file(std::string fname) {
    if (fname.size()  == 0) return false;
    if (!file_exists(fname)) return false;
    return extension_in(fname, matrix_file_extensions);
}

bool extension_in(std::string fname, const std::set<std::string> &extensions, bool die_if_not_found)
{
    std::string extension = fname.substr(fname.size() - 4);
    if (extensions.find(extension) != extensions.end()) return true;
    if (die_if_not_found) {
          die("Unknown extension: " + extension + " of filename: " + fname);
    }
    return false;
}

bool is_sparse_file(std::string fname) {
    return file_exists(fname) && extension_in(fname, sparse_file_extensions);
}

bool is_sparse_binary_file(std::string fname) {
    return file_exists(fname) && extension_in(fname, { ".sbm" });
}

bool is_dense_file(std::string fname) {
    return !is_sparse_file(fname);
}

bool is_compact_file(std::string fname) {
    return file_exists(fname) && extension_in(fname, compact_matrix_file_extensions);
}


