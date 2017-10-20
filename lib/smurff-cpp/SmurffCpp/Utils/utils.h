#pragma once

#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <array>
#include <map>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "omp_util.h"

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
        void init(const std::vector<T> &v) {
            _m = v;
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

template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    out << "[";
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last)
            out << ", ";
    }
    out << "]";
    return out;
}
