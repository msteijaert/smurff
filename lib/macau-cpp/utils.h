#pragma once

#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>

#include <csr.h>
#include <dsparse.h>

template<typename T>
class thread_vector
{
    public:
        thread_vector(const T &t) : _m(thread_limit(), t), _i(t) {}
        template<typename F>
        T combine(F f) const {
            return std::accumulate(_m.begin(), _m.end(), _i, f);
        }
        T combine() const {
            return std::accumulate(_m.begin(), _m.end(), _i, std::plus<T>());
        }

        T &local() {
            return _m.at(thread_num()); 
        }
        void reset() {
            for(auto &t: _m) t = _i;
        }


    private:
        std::vector<T> _m;
        T _i;
};

#ifdef NDEBUG
#define SHOW(m)
#else
#define SHOW(m) std::cout << #m << ":\n" << m << std::endl;
#endif

class SparseFeat {
  public:
    BinaryCSR M;
    BinaryCSR Mt;

    SparseFeat() {}

    SparseFeat(int nrow, int ncol, long nnz, int* rows, int* cols) {
      new_bcsr(&M,  nnz, nrow, ncol, rows, cols);
      new_bcsr(&Mt, nnz, ncol, nrow, cols, rows);
    }
    virtual ~SparseFeat() {
      free_bcsr( & M);
      free_bcsr( & Mt);
    }
    int nfeat()    {return M.ncol;}
    int cols()     {return M.ncol;}
    int nsamples() {return M.nrow;}
    int rows()     {return M.nrow;}
};

class SparseDoubleFeat {
  public:
    CSR M;
    CSR Mt;

    SparseDoubleFeat() {}

    SparseDoubleFeat(int nrow, int ncol, long nnz, int* rows, int* cols, double* vals) {
      new_csr(&M,  nnz, nrow, ncol, rows, cols, vals);
      new_csr(&Mt, nnz, ncol, nrow, cols, rows, vals);
    }
    virtual ~SparseDoubleFeat() {
      free_csr( & M);
      free_csr( & Mt);
    }
    int nfeat()    {return M.ncol;}
    int cols()     {return M.ncol;}
    int nsamples() {return M.nrow;}
    int rows()     {return M.nrow;}
};

inline double tick() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); 
}

inline double clamp(double x, double min, double max) {
  return x < min ? min : (x > max ? max : x);
}

inline std::pair<double, double> getMinMax(const Eigen::SparseMatrix<double> &mat) { 
    double min = INFINITY;
    double max = -INFINITY;
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat,k); it; ++it) {
            double v = it.value();
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }
    return std::make_pair(min, max);
}

inline void split_work_mpi(int num_latent, int num_nodes, int* work) {
   double avg_work = num_latent / (double) num_nodes;
   int work_unit;
   if (2 <= avg_work) work_unit = 2;
   else work_unit = 1;

   int min_work  = work_unit * (int)floor(avg_work / work_unit);
   int work_left = num_latent;

   for (int i = 0; i < num_nodes; i++) {
      work[i]    = min_work;
      work_left -= min_work;
   }
   int i = 0;
   while (work_left > 0) {
      int take = std::min(work_left, work_unit);
      work[i]   += take;
      work_left -= take;
      i = (i + 1) % num_nodes;
   }
}


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


inline void sparseFromIJV(Eigen::SparseMatrix<double> &X, int* rows, int* cols, double* values, int N) {
    sparse_vec_iterator begin(rows, cols, values, 0);
    sparse_vec_iterator end(rows, cols, values, N);
    X.setFromTriplets(begin, end);
}

inline void sparseFromIJ(Eigen::SparseMatrix<double> &X, int* rows, int* cols, int N) {
    sparse_vec_iterator begin(rows, cols, 1.0, 0);
    sparse_vec_iterator end  (rows, cols, 1.0, N);
    X.setFromTriplets(begin, end);
}

inline Eigen::SparseMatrix<double> to_eigen(SparseDoubleMatrix &Y) 
{
    Eigen::SparseMatrix<double> out(Y.nrow, Y.ncol);
    sparseFromIJV(out, Y.rows, Y.cols, Y.vals, Y.nnz);
    return out;
}

inline Eigen::SparseMatrix<double> to_eigen(SparseBinaryMatrix &Y) 
{
   Eigen::SparseMatrix<double> out(Y.nrow, Y.ncol);
   sparseFromIJ(out, Y.rows, Y.cols, Y.nnz);
   return out;
}

template <typename Matrix>
inline bool is_binary(const Matrix &M) 
{
    auto *values = M.valuePtr();
    for(int i=0; i<M.nonZeros(); ++i) {
        if (values[i] != 1.0 && values[i] != 0.0) return false;
    }

    std::cout << "Detected binary matrix\n";

    return true;
}


inline double square(double x) { return x * x; }

inline void row_mean_var(Eigen::VectorXd & mean, Eigen::VectorXd & var, const Eigen::MatrixXd X) {
  const int N = X.cols();
  const int D = X.rows();

  mean.resize(D);
  var.resize(D);
  mean.setZero();
  var.setZero();

#pragma omp parallel
  {
    Eigen::VectorXd tmp(D);
    tmp.setZero();
#pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
      for (int d = 0; d < D; d++) {
        tmp(d) += X(d, i);
      }
    }
#pragma omp critical
    {
      mean += tmp;
    }
  }
  // computing mean
  mean /= N;

#pragma omp parallel
  {
    Eigen::VectorXd tmp(D);
    tmp.setZero();
#pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
      for (int d = 0; d < D; d++) {
        tmp(d) += square(X(d, i) - mean(d));
      }
    }
#pragma omp critical
    {
      var += tmp;
    }
  }
  var /= N;
}

inline void writeToCSVfile(std::string filename, Eigen::MatrixXd matrix) {
  const static Eigen::IOFormat csvFormat(6, Eigen::DontAlignCols, ",", "\n");
  std::ofstream file(filename.c_str());
  file << matrix.format(csvFormat);
}

inline std::string to_string_with_precision(const double a_value, const int n = 6)
{
    std::ostringstream out;
    out << std::setprecision(n) << a_value;
    return out.str();
}

inline bool file_exists(const char *fileName)
{
   std::ifstream infile(fileName);
   return infile.good();
}

inline bool file_exists(const std::string fileName)
{
   return file_exists(fileName.c_str());
}

inline std::unique_ptr<SparseFeat> load_bcsr(const char* filename) {
   SparseBinaryMatrix* A = read_sbm(filename);
   SparseFeat* sf = new SparseFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols);
   free_sbm(A);
   std::unique_ptr<SparseFeat> sf_ptr(sf);
   return sf_ptr;
}

inline std::unique_ptr<SparseDoubleFeat> load_csr(const char* filename) {
   struct SparseDoubleMatrix* A = read_sdm(filename);
   SparseDoubleFeat* sf = new SparseDoubleFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols, A->vals);
   delete A;
   std::unique_ptr<SparseDoubleFeat> sf_ptr(sf);
   return sf_ptr;
}


// assumes matrix (not tensor)
inline Eigen::MatrixXd to_coo(const Eigen::SparseMatrix<double> &Y) {
    Eigen::MatrixXd coords(Y.nonZeros(), 3);
#pragma omp parallel for schedule(dynamic, 2)
    for (int k = 0; k < Y.outerSize(); ++k) {
        int idx = Y.outerIndexPtr()[k];
        for (Eigen::SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
            coords(idx, 0) = it.row();
            coords(idx, 1) = it.col();
            coords(idx, 2) = it.value();
            idx++;
        }
    }
    return coords;
}

template<class Matrix>
void write_ddm(const char* filename, const Matrix& matrix){
    std::ofstream out(filename,std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}

template<class Matrix>
Matrix read_ddm(const char* filename) {
    Matrix matrix;
    std::ifstream in(filename,std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
    return matrix;
}

inline Eigen::MatrixXd sparse_to_dense(SparseBinaryMatrix &in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = 1.;
    return out;
}

inline Eigen::MatrixXd sparse_to_dense(SparseDoubleMatrix &in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = in.vals[i];
    return out;
}

typedef Eigen::VectorXd VectorNd;
typedef Eigen::MatrixXd MatrixNNd;
typedef Eigen::ArrayXd ArrayNd;

typedef Eigen::SparseMatrix<double> SparseMatrixD;


inline void die_unless_file_exists(std::string fname) {
    if ( fname.size() && ! file_exists(fname) ) {
        throw std::runtime_error(std::string("[ERROR]\nFile '") + fname +  "' not found.\n");
    }
}


