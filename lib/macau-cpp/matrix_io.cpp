#include <set>
#include <cstdlib>

#include <unsupported/Eigen/SparseExtra>

#include "utils.h"
#include "matrix_io.h"

const static Eigen::IOFormat csvFormat(6, Eigen::DontAlignCols, ",", "\n");

void writeToCSVfile(std::string filename, Eigen::MatrixXd matrix) {
  std::ofstream file(filename.c_str());
  file << matrix.rows() << std::endl;
  file << matrix.cols() << std::endl;
  file << matrix.format(csvFormat);
}

void readFromCSVfile(std::string filename, Eigen::MatrixXd &matrix) {
    std::ifstream file(filename.c_str());
    std::string line;
    
    // rows and cols
    getline(file, line); 
    int nrow = atol(line.c_str());
    getline(file, line); 
    int ncol = atol(line.c_str());
    matrix.resize(nrow, ncol);
    int pos = 0;

    while (getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            matrix.data()[pos++] = strtod(cell.c_str(), NULL);
        }
    }
    assert(pos == nrow*ncol);
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

static std::set<std::string> compact_matrix_fname_extensions = { ".sbm", ".sdm", ".ddm" };
static std::set<std::string> txt_matrix_fname_extensions = { ".mtx", ".mm", ".csv" };
static std::set<std::string> matrix_fname_extensions = { ".sbm", ".sdm", ".ddm", ".mtx", ".mm", ".csv" };
static std::set<std::string> sparse_fname_extensions = { ".sbm", ".sdm", ".mtx", ".mm" };

bool extension_in(std::string fname, const std::set<std::string> &extensions, bool = false);

bool is_matrix_fname(std::string fname) {
    return extension_in(fname, matrix_fname_extensions);
}

bool extension_in(std::string fname, const std::set<std::string> &extensions, bool die_if_not_found)
{
    std::string extension = fname.substr(fname.find_last_of("."));
    if (extensions.find(extension) != extensions.end()) return true;
    if (die_if_not_found) {
          die("Unknown extension: " + extension + " of filename: " + fname);
    }
    return false;
}

bool is_sparse_fname(std::string fname) {
    return extension_in(fname, sparse_fname_extensions);
}

bool is_sparse_binary_fname(std::string fname) {
    return extension_in(fname, { ".sbm" });
}

bool is_dense_fname(std::string fname) {
    return !is_sparse_fname(fname);
}

bool is_compact_fname(std::string fname) {
    return file_exists(fname) && extension_in(fname, compact_matrix_fname_extensions);
}

void read_dense(std::string fname, Eigen::VectorXd &V) {
   Eigen::MatrixXd X;
   read_dense(fname, X);
   V = X; // this will fail if X has more than one column
}
    
void read_dense(std::string fname, Eigen::MatrixXd &X) {
    die_unless_file_exists(fname);
    std::string extension = fname.substr(fname.size() - 4);
    if (extension == ".ddm") {
        read_ddm(fname.c_str(), X);
    } else if (extension == ".csv") {
        readFromCSVfile(fname, X);
    } else {
        die("Unknown filename in read_dense: " + fname);
    }
}

void write_dense(std::string fname, const Eigen::MatrixXd &X) {
    assert(is_dense_fname(fname));
    std::string extension = fname.substr(fname.size() - 4);
    if (extension == ".ddm") {
        write_ddm(fname.c_str(), X);
    } else if (extension == ".csv") {
        writeToCSVfile(fname, X);
    } else {
        die("Unknown filename in write_dense: " + fname);
    }
}

void write_ddm(std::string filename, const Eigen::MatrixXd& matrix) {
    std::ofstream out(filename,std::ios::out | std::ios::binary | std::ios::trunc);
    long rows = matrix.rows();
    long cols = matrix.cols();
    out.write((char*) (&rows), sizeof(long));
    out.write((char*) (&cols), sizeof(long));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );
    out.close();
}

void read_ddm(std::string filename, Eigen::MatrixXd &matrix) {
    std::ifstream in(filename,std::ios::in | std::ios::binary);
    long rows=0, cols=0;
    in.read((char*) (&rows),sizeof(long));
    in.read((char*) (&cols),sizeof(long));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(long) );
    in.close();
}

void read_sparse(std::string fname, Eigen::SparseMatrix<double> &M) {
    assert(is_sparse_fname(fname));
    std::string extension = fname.substr(fname.find_last_of("."));
    if (extension == ".sdm") {
        auto sdm_ptr = read_sdm(fname.c_str());
        M = to_eigen(*sdm_ptr);
        delete sdm_ptr;
    } else if (extension == ".sbm") {
        auto sbm_ptr = read_sbm(fname.c_str());
        M = to_eigen(*sbm_ptr);
        delete sbm_ptr;
    } else if (extension == ".mtx" || extension == ".mm") {
        loadMarket(M, fname.c_str());
    } else  {
        die("Unknown filename in read_sparse: " + fname);
    }
}
