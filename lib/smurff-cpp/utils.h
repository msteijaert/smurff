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
#include <array>

#include <csr.h>
#include <dsparse.h>
#include "omp_util.h"

struct PVec {
    // c'tor
    PVec() : n(0) {}
    PVec(int n) : n(n), v({{0}}) { assert(n <= max); }
    PVec(int a, int b) : n(2), v({{a,b}}) {}

    // meta info
    int size() const { return n; }

    // const accessor
    const int &operator[](int p) const { return v[p]; }
    const int &at(int p) const { assert(p>=0 && p < n); return v[p]; }

    // non-const accessor
    int &operator[](int p) { return v[p]; }
    int &at(int p) { assert(p>=0 && p < n); return v[p]; }

    // operator+
    PVec operator+(const PVec &other) const {
        assert(n == other.n);
        PVec ret = *this;
        for(int i=0; i<n; ++i) { ret[i] += other[i]; }
        return ret;
    }
    //
    // operator-
    PVec operator-(const PVec &other) const {
        assert(n == other.n);
        PVec ret = *this;
        for(int i=0; i<n; ++i) { ret[i] -= other[i]; }
        return ret;
    }

    bool in(const PVec &start, const PVec &end) const {
        for(int i=0; i<n; ++i) {
            if (at(i) < start.at(i)) return false;
            if (at(i) >= end.at(i)) return false;
        }
        return true;
    }

    int dot() const {
        int ret = 1;
        for(int i=0; i<n; ++i) ret *= at(i);
        return ret;
    }


    std::ostream &info(std::ostream &os) const {
        os << "[ ";
        for(int i=0; i<n; ++i) {
            os << n;
            if (i != n-1) os << " x ";
        }
        os << " ]";
        return os;
    }

  private:
    static const unsigned int max = 2; // only matrices for the moment
    int n;
    std::array<int, max> v;
};


template<typename T>
class thread_vector
{
    public:
        thread_vector(const T &t = T()) : _m(thread_limit(), t), _i(t) {}
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
        template<typename F>
        T combine_and_reset(F f) const {
            T ret = combine(f);
            reset();
            return ret;
        }
        T combine_and_reset() {
            T ret = combine();
            reset();
            return ret;
        }
        void init(const T &t) {
            _i = t;
            reset();
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

inline void die(std::string message) {
    throw std::runtime_error(std::string("[ERROR]: ") + message +  "\n");
}

inline void not_implemented(std::string message) {
    throw std::runtime_error(std::string("[Not implemented]: '") + message +  "'\n");
}

inline void die_unless_file_exists(std::string fname) {
    if ( fname.size() && ! file_exists(fname) ) {
        throw std::runtime_error(std::string("[ERROR]\nFile '") + fname +  "' not found.\n");
    }
}

typedef Eigen::VectorXd VectorNd;
typedef Eigen::MatrixXd MatrixNNd;
typedef Eigen::ArrayXd ArrayNd;
typedef Eigen::ArrayXXd ArrayNNd;

typedef Eigen::SparseMatrix<double> SparseMatrixD;


