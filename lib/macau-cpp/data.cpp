#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <signal.h>

#include "model.h"
#include "utils.h"
#include "data.h"
#include "mvnormal.h"

using namespace std; 
using namespace Eigen;

namespace Macau {


////----- Data below


std::ostream &Data::info(std::ostream &os, std::string indent)
{
    os << indent << "Type: " << name << "\n";
    os << indent << "Noise: ";
    noise->info(os, "");
    return os;
}

FixedGaussianNoise &Data::setPrecision(double p) {
  auto *n = new FixedGaussianNoise(*this, p);
  noise.reset(n);
  return *n;
}


AdaptiveGaussianNoise &Data::setAdaptivePrecision(double sn_init, double sn_max) {
  auto *n = new AdaptiveGaussianNoise(*this, sn_init, sn_max);
  noise.reset(n);
  return *n;
}

ProbitNoise &Data::setProbit() {
  auto *n = new ProbitNoise(*this);
  noise.reset(n);
  return *n;
}


std::ostream &MatrixData::info(std::ostream &os, std::string indent)
{
    Data::info(os, indent);
    double train_fill_rate = 100. * nnz() / size();
    os << indent << "Size: " << nnz() << " [" << nrow() << " x " << ncol() << "] (" << train_fill_rate << "%)\n";
    return os;
}

MatrixData& MatricesData::add(int row, int col, const MatrixConfig &c, bool scarce) {
}

void MatricesData::get_pnm(const Model &model, int mode, int n, VectorNd &rr, MatrixNNd &MM) {

        

}
void MatricesData::update_pnm(const Model &,int) {
}

std::ostream &MatricesData::info(std::ostream &os, std::string indent) {
}

void             MatricesData::init()       {
}
int              MatricesData::nnz()  const {
}
int              MatricesData::size() const {
}
std::vector<int> MatricesData::dims() const {
}

template<typename YType>
void MatrixDataTempl<YType>::init_base()
{
    assert(nrow() > 0 && ncol() > 0);

//    if (pred.ncols > 0) {
//        assert(pred.nrows == nrow() && pred.ncols == ncol() && "Size of train must be equal to size of test");
//    }

    mean_rating = Y.sum() / Y.nonZeros();

    Yc.push_back(Y);
    Yc.push_back(Y.transpose());

    noise->init();
}

template<>
void MatrixDataTempl<SparseMatrixD>::init()
{
    init_base();
    Yc.at(0).coeffs() -= mean_rating;
    Yc.at(1).coeffs() -= mean_rating;
}


template<>
void MatrixDataTempl<Eigen::MatrixXd>::init()
{
    init_base();
    Yc.at(0).array() -= this->mean_rating;
    Yc.at(1).array() -= this->mean_rating;
}

template<>
double MatrixDataTempl<Eigen::MatrixXd>::var_total() const {
    auto &Y = Yc.at(0);
    double se = Y.array().square().sum();

    double var_total = se / Y.nonZeros();
    if (var_total <= 0.0 || std::isnan(var_total)) {
        // if var cannot be computed using 1.0
        var_total = 1.0;
    }

    return var_total;
}

template<>
double MatrixDataTempl<SparseMatrixD>::var_total() const {
    double se = 0.0;
    auto &Y = Yc.at(0);

#pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
    for (int k = 0; k < Y.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
            se += square(it.value());
        }
    }

    double var_total = se / Y.nonZeros();
    if (var_total <= 0.0 || std::isnan(var_total)) {
        // if var cannot be computed using 1.0
        var_total = 1.0;
    }

    return var_total;
}

// for the adaptive gaussian noise
template<>
double MatrixDataTempl<Eigen::MatrixXd>::sumsq(const Model &model) const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < this->ncol(); j++) {
        for (int i = 0; i < this->nrow(); i++) {
            double Yhat = model.predict({i,j});
            sumsq += square(Yhat - this->Y(i,j));
        }
    }

    return sumsq;
}

template<>
double MatrixDataTempl<SparseMatrixD>::sumsq(const Model &model) const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < Y.outerSize(); j++) {
        for (SparseMatrix<double>::InnerIterator it(Y, j); it; ++it) {
            int i = it.row();
            double Yhat = model.predict({i,j});
            sumsq += square(Yhat - it.value());
        }
    }

    return sumsq;
}

//
//-- ScarceMatrixData specific stuff

void ScarceMatrixData::get_pnm(const Model &model, int mode, int n, VectorXd &rr, MatrixXd &MM) {
    auto &Y = Yc.at(mode);
    const int num_latent = model.nlatent();
    const MatrixXd &Vf = model.V(mode);
    const int local_nnz = Y.col(n).nonZeros();
    const int total_nnz = Y.nonZeros();
    const double alpha = noise->getAlpha();

    bool in_parallel = (local_nnz >10000) || ((double)local_nnz > (double)total_nnz / 100.);
    if (in_parallel) {
        const int task_size = ceil(local_nnz / 100.0);
        auto from = Y.outerIndexPtr()[n];
        auto to = Y.outerIndexPtr()[n+1];
        thread_vector<VectorXd> rrs(VectorXd::Zero(num_latent));
        thread_vector<MatrixXd> MMs(MatrixXd::Zero(num_latent, num_latent)); 
        for(int j=from; j<to; j+=task_size) {
#pragma omp task shared(Y,Vf,rrs,MMs)
            {
                auto &my_rr = rrs.local();
                auto &my_MM = MMs.local();

                for(int i=j; i<std::min(j+task_size,to); ++i)
                {
                    auto val = Y.valuePtr()[i];
                    auto idx = Y.innerIndexPtr()[i];
                    const auto &col = Vf.col(idx);
                    my_rr.noalias() += col * val;
                    my_MM.triangularView<Eigen::Lower>() += col * col.transpose();
                }
                
                // make MM complete
                my_MM.triangularView<Upper>() = my_MM.transpose();

            }
        }
#pragma omp taskwait
        // accumulate + add noise
        MM += MMs.combine() * alpha;
        rr += rrs.combine() * alpha;
    } else {
        VectorXd my_rr = VectorXd::Zero(num_latent);
        MatrixXd my_MM = MatrixXd::Zero(num_latent, num_latent);
        for (SparseMatrix<double>::InnerIterator it(Y, n); it; ++it) {
            const auto &col = Vf.col(it.row());
            my_rr.noalias() += col * it.value();
            my_MM.triangularView<Lower>() += col * col.transpose();
        }

        // make MM complete
        my_MM.triangularView<Upper>() = my_MM.transpose();

        //add noise
        my_rr.array() *= alpha;
        my_MM.array() *= alpha;

        // add to global
        rr += my_rr;
        MM += my_MM;
    }
}

void ScarceBinaryMatrixData::get_pnm(const Model &model, int mode, int n, VectorXd &rr, MatrixXd &MM)
{
    // todo : check noise == probit noise
    auto u = model.U(mode).col(n);
    for (SparseMatrix<double>::InnerIterator it(Yc.at(mode), n); it; ++it) {
        const auto &col = model.V(mode).col(it.row());
        MM.noalias() += col * col.transpose();
        auto z = (2 * it.value() - 1) * fabs(col.dot(u) + bmrandn_single());
        rr.noalias() += col * z;
    }
}


//
//-- FullMatrixData specific stuff
//

template<class YType>
void FullMatrixData<YType>::get_pnm(const Model &model, int mode, int d, VectorXd &rr, MatrixXd &MM) {
    const double alpha = this->noise->getAlpha();
    auto &Y = this->Yc.at(mode);
    rr.noalias() += (model.V(mode) * Y.col(d)) * alpha;
    MM.noalias() += VV[mode] * alpha; 
}

template<class YType>
void FullMatrixData<YType>::update_pnm(const Model &model, int mode) {
    auto &Vf = model.V(mode);
    const int nl = model.nlatent();
    thread_vector<MatrixXd> VVs(MatrixXd::Zero(nl, nl));

#pragma omp parallel for schedule(dynamic, 8) shared(VVs)
    for(int n = 0; n < Vf.cols(); n++) {
        auto v = Vf.col(n);
        VVs.local() += v * v.transpose();
    }

    VV[mode] = VVs.combine();
}


template struct FullMatrixData<Eigen::MatrixXd>;
template struct FullMatrixData<Eigen::SparseMatrix<double>>;

} //end namespace Macau

#ifdef BENCH

#include "utils.h"

int main()
{
    const int N = 32 * 1024;
    const int K = 96;
    const int R = 20;
    {
        init_bmrng(1234);
        MatrixXd U(K,N);
        bmrandn(U);

        MatrixXd M(K,K) ;
        double start = tick();
        for(int i=0; i<R; ++i) {
            M.setZero();
            for(int j=0; j<N;++j) {
                const auto &col = U.col(j);
                M.noalias() += col * col.transpose();
            }
        }
        double stop = tick();
        std::cout << "norm U: " << U.norm() << std::endl;
        std::cout << "norm M: " << M.norm() << std::endl;
        std::cout << "MatrixXd: " << stop - start << std::endl;
    }

    {
        init_bmrng(1234);
        Matrix<double, K, Dynamic> U(K,N);
        U = nrandn(K,N);

        Matrix<double,K,K> M;
        double start = tick();
        for(int i=0; i<R; ++i) {
            M.setZero();
            for(int j=0; j<N;++j) {
                const auto &col = U.col(j);
                M.noalias() += col * col.transpose();
            }
        }
        double stop = tick();
        std::cout << "norm U: " << U.norm() << std::endl;
        std::cout << "norm M: " << M.norm() << std::endl;
        std::cout << "MatrixNNd: " << stop - start << std::endl;
    }

    return 0;
}
#endif
