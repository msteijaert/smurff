
#include "matrix_io.h"

struct sparse_vec_iterator {
    sparse_vec_iterator(int *rows, int *cols, uint64_t pos)
        : rows(rows), cols(cols), vals(0), fixed_val(1.0), pos(pos) {}
    sparse_vec_iterator(int *rows, int *cols, double *vals, uint64_t pos)
        : rows(rows), cols(cols), vals(vals), fixed_val(NAN), pos(pos) {}
    sparse_vec_iterator(int *rows, int *cols, double fixed_val, uint64_t pos)
        : rows(rows), cols(cols), vals(0), fixed_val(fixed_val), pos(pos) {}

    int *rows, *cols;
    double *vals; // can be null pointer -> use fixed value
    double fixed_val;
    int pos;
    bool operator!=(const sparse_vec_iterator &other) const { 
        assert(rows == other.rows);
        assert(cols == other.cols);
        assert(vals == other.vals);
        return pos != other.pos;
    }
    sparse_vec_iterator &operator++() { pos++; return *this; }
    typedef Eigen::Triplet<double> T;
    T v;
    T* operator->() {
        // also convert from 1-base to 0-base
        uint32_t row = rows[pos];
        uint32_t col = cols[pos];
        double val = vals ? vals[pos] : 1.0;
        v = T(row, col, val);
        return &v;
    }
};


void sparseFromIJV(Eigen::SparseMatrix<double> &X, int* rows, int* cols, double* values, int N) {
    sparse_vec_iterator begin(rows, cols, values, 0);
    sparse_vec_iterator end(rows, cols, values, N);
    X.setFromTriplets(begin, end);
}

void sparseFromIJ(Eigen::SparseMatrix<double> &X, int* rows, int* cols, int N) {
    sparse_vec_iterator begin(rows, cols, 1.0, 0);
    sparse_vec_iterator end  (rows, cols, 1.0, N);
    X.setFromTriplets(begin, end);
}

Eigen::SparseMatrix<double> to_eigen(SparseDoubleMatrix &Y) 
{
    Eigen::SparseMatrix<double> out(Y.nrow, Y.ncol);
    sparseFromIJV(out, Y.rows, Y.cols, Y.vals, Y.nnz);
    return out;
}

Eigen::SparseMatrix<double> to_eigen(SparseBinaryMatrix &Y) 
{
   Eigen::SparseMatrix<double> out(Y.nrow, Y.ncol);
   sparseFromIJ(out, Y.rows, Y.cols, Y.nnz);
   return out;
}

void writeToCSVfile(std::string filename, Eigen::MatrixXd matrix) {
  const static Eigen::IOFormat csvFormat(6, Eigen::DontAlignCols, ",", "\n");
  std::ofstream file(filename.c_str());
  file << matrix.format(csvFormat);
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
