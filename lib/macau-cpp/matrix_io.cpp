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

    int row = 0;
    int col = 0;
    while (getline(file, line)) {
        col = 0;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            matrix(row, col++) = strtod(cell.c_str(), NULL);
        }
        row++;
    }
    assert(row == nrow);
    assert(col == ncol);
}

Macau::MatrixConfig read_csv(std::string filename) {
    Macau::MatrixConfig ret;
    std::ifstream file(filename.c_str());
    std::string line;
    ret.dense = true;

    // rows and cols
    getline(file, line); 
    ret.nrow = atol(line.c_str());
    getline(file, line); 
    ret.ncol = atol(line.c_str());
    ret.nnz = ret.nrow * ret.ncol;
    ret.values = new double[ret.nnz];

    int row = 0;
    int col = 0;
    while (getline(file, line)) {
        col = 0;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            ret.values[row + (ret.nrow*col++)] = strtod(cell.c_str(), NULL);
        }
        row++;
    }
    assert(row == ret.nrow);
    assert(col == ret.ncol);

    return ret;
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
    
Macau::MatrixConfig read_dense(std::string fname) {
    die_unless_file_exists(fname);
    std::string extension = fname.substr(fname.size() - 4);
    if (extension == ".ddm") {
        return read_ddm(fname);
    } else if (extension == ".csv") {
        return read_csv(fname);
    } else {
        die("Unknown filename in read_dense: " + fname);
    }
    return Macau::MatrixConfig();
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

Macau::MatrixConfig read_ddm(std::string filename) {
    Macau::MatrixConfig ret;
    ret.dense = true;

    std::ifstream in(filename,std::ios::in | std::ios::binary);
    in.read((char*) (&ret.nrow),sizeof(long));
    in.read((char*) (&ret.ncol),sizeof(long));
    ret.nnz = ret.nrow * ret.ncol;
    ret.values = new double[ret.nnz];
    in.read( (char *) ret.values, ret.nnz*sizeof(double) );
    return ret;
}

Macau::MatrixConfig read_mtx(std::string fname) {
    Macau::MatrixConfig ret;
    ret.dense = false;
    std::ifstream fin(fname);

    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');

    // Read defining parameters:
    fin >> ret.nrow >> ret.ncol >> ret.nnz;

    ret.rows   = new int[ret.nnz];
    ret.cols   = new int[ret.nnz];
    ret.values = new double[ret.nnz];

    // Read the data
    for (int l = 0; l < ret.nnz; l++)
    {
        fin >> ret.rows[l] >> ret.cols[l] >> ret.values[l];
        ret.rows[l]--;
        ret.cols[l]--;
    }

    return ret;
}

Macau::MatrixConfig read_sparse(std::string fname) {
    assert(is_sparse_fname(fname));
    std::string extension = fname.substr(fname.find_last_of("."));
    if (extension == ".sdm") {
        auto p = read_sdm(fname.c_str());
        auto m = Macau::MatrixConfig(p->nrow, p->ncol, p->nnz, p->rows, p->cols, p->vals);
        delete p;
        return m;
    } else if (extension == ".sbm") {
        auto p = read_sbm(fname.c_str());
        auto m = Macau::MatrixConfig(p->nrow, p->ncol, p->nnz, p->rows, p->cols);
        delete p;
        return m;
    } else if (extension == ".mtx" || extension == ".mm") {
        return read_mtx(fname);
    } else  {
        die("Unknown filename in read_sparse: " + fname);
    }

    return Macau::MatrixConfig();
}

void read_sparse(std::string fname, Eigen::SparseMatrix<double> &M) {
    assert(is_sparse_fname(fname));
    std::string extension = fname.substr(fname.find_last_of("."));
    if (extension == ".sdm") {
        auto sdm_ptr = read_sdm(fname.c_str());
        M = sparse_to_eigen(*sdm_ptr);
        delete sdm_ptr;
    } else if (extension == ".sbm") {
        auto sbm_ptr = read_sbm(fname.c_str());
        M = sparse_to_eigen(*sbm_ptr);
        delete sbm_ptr;
    } else if (extension == ".mtx" || extension == ".mm") {
        loadMarket(M, fname.c_str());
    } else  {
        die("Unknown filename in read_sparse: " + fname);
    }
}

Macau::MatrixConfig read_matrix(std::string fname) {
    if (is_sparse_fname(fname)) return read_sparse(fname);
    else return read_dense(fname);
}
