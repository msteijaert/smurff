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
#include "truncnorm.h"

#include "Noiseless.h"

using namespace std;
using namespace Eigen;

namespace smurff {


////----- Data below
SubModel MatricesData::Block::submodel(const SubModel &model) const {
    return SubModel(model, start(), dim());
}

MatrixData &MatricesData::add(const PVec &p, std::unique_ptr<MatrixData> data) {
    blocks.push_back(Block(p, std::move(data)));
    return blocks.back().data();
}


void MatricesData::get_pnm(const SubModel &model, int mode, int pos, VectorNd &rr, MatrixNNd &MM) {
    int count = 0;
    apply(mode, pos, [&model, mode, pos, &rr, &MM, &count](const Block &b) {
        b.data().get_pnm(b.submodel(model), mode, pos - b.start(mode), rr, MM);
        count++;
    });
    assert(count>0);
}

void MatricesData::update(const SubModel &model) {
    for(auto &b : blocks) {
        b.data().noise().update(b.submodel(model));
    }
}

void MatricesData::update_pnm(const SubModel &model, int m) {
    for(auto &b : blocks) {
        b.data().update_pnm(b.submodel(model), m);
    }
}

std::ostream &MatricesData::info(std::ostream &os, std::string indent)
{
    MatrixData::info(os, indent);
    os << indent << "Sub-Matrices:\n";
    for(auto &p : blocks) {
        os << indent;
        p.pos().info(os);
        os << ":\n";
        p.data().info(os, indent + "  ");
        os << std::endl;
    }
    return os;
}

std::ostream &MatricesData::status(std::ostream &os, std::string indent) const
{
    os << indent << "Sub-Matrices:\n";
    for(auto &p : blocks) {
        os << indent << "  ";
        p.pos().info(os);
        os << ": " << p.data().noise().getStatus() << "\n";
    }
    return os;
}

void MatricesData::init_pre()
{
    mode_dim.resize(nmode());
    for(int n = 0; n<nmode(); ++n) {
        std::vector<int> S(blocks.size());
        int max_pos = -1;
        for(auto &blk : blocks) {
            int pos  = blk.pos(n);
            int size = blk.dim(n);
            assert(size > 0);
            assert(S.at(pos) == 0 || S.at(pos) == size);
            S.at(pos) = size;
            max_pos = std::max(max_pos, pos);
        }
        int off = 0;
        auto &O = mode_dim.at(n);
        O.resize(max_pos+1);
        for(int pos=0; pos<=max_pos; ++pos) {
            O.at(pos) = off;
            off += S[pos];
        }
        total_dim.at(n) = off;
        for(auto &blk : blocks) {
            int pos = blk.pos(n);
            blk._start.at(n) = O[pos];
        }

    }

    cwise_mean = sum() / (double)(size() - nna());

    // init sub-matrices
    for(auto &p : blocks) {
        p.data().init_pre();
        p.data().compute_mode_mean();
    }
}

void MatricesData::init_post()
{
    Data::init_post();

    // init sub-matrices
    for(auto &p : blocks) {
        p.data().init_post();
    }
}

void MatricesData::setCenterMode(std::string mode)
{
    Data::setCenterMode(mode);
    for(auto &p : blocks) p.data().setCenterMode(mode);
}


void MatricesData::center(double global_mean)
{
    // center sub-matrices
    assert(global_mean == cwise_mean);
    this->global_mean = global_mean;
    for(auto &p : blocks) p.data().center(cwise_mean);
}

double MatricesData::compute_mode_mean(int mode, int pos) {
    double sum = .0;
    int N = 0;
    int count = 0;

    apply(mode, pos, [&](const Block &b) {
        double local_mean = b.data().mean(mode, pos - b.start(mode));
        sum += local_mean * b.dim(mode);
        N += b.dim(mode);
        count++;
    });

    assert(N>0);

    return sum / N;
}

double MatricesData::offset_to_mean(const PVec &pos) const {
    const Block &b = find(pos);
    return b.data().offset_to_mean(pos - b.start());
}

double MatricesData::train_rmse(const SubModel &model) const {
    double sum = .0;
    int N = 0;
    int count = 0;

    for(auto &p : blocks) {
        auto &mtx = p.data();
        double local_rmse = mtx.train_rmse(p.submodel(model));
        sum += (local_rmse * local_rmse) * (mtx.size() - mtx.nna());
        N += (mtx.size() - mtx.nna());
        count++;
    }

    assert(N>0);

    return sqrt(sum / N);
}

template<typename YType>
void MatrixDataTempl<YType>::init_pre()
{
    assert(nrow() > 0 && ncol() > 0);

    Yc.push_back(Y);
    Yc.push_back(Y.transpose());


    cwise_mean = sum() / (size() - nna());
}

template<>
double MatrixDataTempl<Eigen::MatrixXd>::var_total() const {
    auto &Y = Yc.at(0);
    double se = Y.array().square().sum();
    double var = se / Y.nonZeros();
    if (var <= 0.0 || std::isnan(var)) {
        // if var cannot be computed using 1.0
        var = 1.0;
    }

    return var;
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

    double var = se / Y.nonZeros();
    if (var <= 0.0 || std::isnan(var)) {
        // if var cannot be computed using 1.0
        var = 1.0;
    }

    return var;
}

//macau
/*
double var_total(MatrixData & matrixData)
{
  double se = 0.0;
  double mean_value = matrixData.mean_value;

  #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
  for (int k = 0; k < matrixData.Y.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(matrixData.Y, k); it; ++it) {
      se += square(it.value() - mean_value);
    }
  }

  var_total = se / matrixData.Y.nonZeros();
  if (var_total <= 0.0 || std::isnan(var_total)) {
    // if var cannot be computed using 1.0
    var_total = 1.0;
  }

  return var_total;
}
*/

//macau
/*
double var_total(TensorData & data)
{
  double se = 0.0;
  double mean_value = data.mean_value;

  auto& sparseMode   = (*data.Y)[0];
  VectorXd & values  = sparseMode->values;

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
  for (int i = 0; i < values.size(); i++) {
    se += square(values(i) - mean_value);
  }
  var_total = se / values.size();
  if (var_total <= 0.0 || std::isnan(var_total)) {
    var_total = 1.0;
  }

  return var_total;
}
*/

// for the adaptive gaussian noise
template<>
double MatrixDataTempl<Eigen::MatrixXd>::sumsq(const SubModel &model) const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < this->ncol(); j++) {
        for (int i = 0; i < this->nrow(); i++) {
            double Yhat = model.dot({j,i}) + offset_to_mean({j,i});
            sumsq += square(Yhat - this->Y(i,j));
        }
    }

    return sumsq;
}

template<>
double MatrixDataTempl<SparseMatrixD>::sumsq(const SubModel &model) const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < Y.outerSize(); j++) {
        for (SparseMatrix<double>::InnerIterator it(Y, j); it; ++it) {
            int i = it.row();
            double Yhat = model.dot({j,i}) + offset_to_mean({j,i});
            sumsq += square(Yhat - it.value());
        }
    }

    return sumsq;
}

//macau
/*
double sumsq(MatrixData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
{
  double sumsq = 0.0;
  MatrixXd & U = *samples[0];
  MatrixXd & V = *samples[1];

  Eigen::SparseMatrix<double> & train = data.Y;
  double mean_value = data.mean_value;

  #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
  for (int j = 0; j < train.outerSize(); j++) {
    auto Vj = V.col(j);
    for (SparseMatrix<double>::InnerIterator it(train, j); it; ++it) {
      double Yhat = Vj.dot( U.col(it.row()) ) + mean_value;
      sumsq += square(Yhat - it.value());
    }
  }

  return sumsq;
}
*/

//macau
/*
double sumsq(TensorData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
{
  double sumsq = 0.0;
  double mean_value = data.mean_value;

  auto& sparseMode = (*data.Y)[0];
  auto& U = samples[0];

  const int nmodes = samples.size();
  const int num_latents = U->rows();

  #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
  for (int n = 0; n < data.dims(0); n++) {
    Eigen::VectorXd u = U->col(n);
    for (int j = sparseMode->row_ptr(n);
             j < sparseMode->row_ptr(n + 1);
             j++)
    {
      VectorXi idx = sparseMode->indices.row(j);
      // computing prediction from tensor
      double Yhat = mean_value;
      for (int d = 0; d < num_latents; d++) {
        double tmp = u(d);

        for (int m = 1; m < nmodes; m++) {
          tmp *= (*samples[m])(d, idx(m - 1));
        }
        Yhat += tmp;
      }
      sumsq += square(Yhat - sparseMode->values(j));
    }

  }

   return sumsq;
}
*/

template<typename YType>
double MatrixDataTempl<YType>::offset_to_mean(const PVec &pos) const {
         if (center_mode == CENTER_GLOBAL) return global_mean;
    else if (center_mode == CENTER_VIEW)   return cwise_mean;
    else if (center_mode == CENTER_ROWS)   return mean(1,pos.at(1));
    else if (center_mode == CENTER_COLS)   return mean(0,pos.at(0));
    else if (center_mode == CENTER_NONE)   return .0;
    assert(false);
    return .0;
}

//
//-- ScarceMatrixData specific stuff

void ScarceMatrixData::init_pre() {
    MatrixDataTempl<SparseMatrixD>::init_pre();

    // check no rows, nor cols withouth data
    for(unsigned i=0; i<Yc.size(); ++i) {
        auto &v = Yc[i];
        auto &count = num_empty[i];
        for (int j = 0; j < v.cols(); j++) {
            if (v.col(j).nonZeros() == 0) count++;
        }
    }
}

double ScarceMatrixData::compute_mode_mean(int m, int c)
{
    const auto &col = Yc.at(m).col(c);
    if (col.nonZeros() == 0) return cwise_mean;
    return col.sum() / col.nonZeros();
}

void ScarceMatrixData::center(double global_mean)
{
    assert(!centered);
    this->global_mean = global_mean;

    auto center_cols = [this](SparseMatrixD &Y, int m) {
        for (int k = 0; k < Y.outerSize(); ++k) {
            double v = mean(m, k);
            for (SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
                it.valueRef() -= v;
            }
        }
    };

    if (center_mode == CENTER_GLOBAL) {
        Yc.at(0).coeffs() -= global_mean;
        Yc.at(1).coeffs() -= global_mean;
    } else if (center_mode == CENTER_VIEW) {
        Yc.at(0).coeffs() -= cwise_mean;
        Yc.at(1).coeffs() -= cwise_mean;
    } else if (center_mode == CENTER_COLS) {
        center_cols(Yc.at(0), 0);
        Yc.at(1) = Yc.at(0).transpose();
    } else if (center_mode == CENTER_ROWS) {
        center_cols(Yc.at(1), 1);
        Yc.at(0) = Yc.at(1).transpose();
    }

    centered = true;
}

std::ostream &ScarceMatrixData::info(std::ostream &os, std::string indent)
{
    MatrixDataTempl<SparseMatrixD>::info(os, indent);
    if (num_empty[0]) os << indent << "  Warning: " << num_empty[0] << " empty cols\n";
    if (num_empty[1]) os << indent << "  Warning: " << num_empty[1] << " empty rows\n";
    return os;
}

void ScarceMatrixData::get_pnm(const SubModel &model, int mode, int n, VectorXd &rr, MatrixXd &MM) {
    auto &Y = Yc.at(mode);
    const int num_latent = model.nlatent();
    const auto &Vf = model.V(mode);
    const int local_nnz = Y.col(n).nonZeros();
    const int total_nnz = Y.nonZeros();
    const double alpha = noise().getAlpha();

    bool in_parallel = (local_nnz >10000) || ((double)local_nnz > (double)total_nnz / 100.);
    if (in_parallel) {
        const int task_size = ceil(local_nnz / 100.0);
        auto from = Y.outerIndexPtr()[n];
        auto to = Y.outerIndexPtr()[n+1];
        thread_vector<VectorXd> rrs(VectorXd::Zero(num_latent));
        thread_vector<MatrixXd> MMs(MatrixXd::Zero(num_latent, num_latent));
        for(int j=from; j<to; j+=task_size) {
#pragma omp task shared(model,Y,Vf,rrs,MMs)
            {
                auto &my_rr = rrs.local();
                auto &my_MM = MMs.local();

                for(int i=j; i<std::min(j+task_size,to); ++i)
                {
                    auto val = Y.valuePtr()[i];
                    auto idx = Y.innerIndexPtr()[i];
                    const auto &col = model.V(mode).col(idx);
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

double ScarceMatrixData::train_rmse(const SubModel &model) const {
    double se = 0.;
#pragma omp parallel for schedule(guided) reduction(+:se)
    for(int c=0; c<Y.cols();++c) {
        for (SparseMatrix<double>::InnerIterator it(Y, c); it; ++it) {
            se += square(it.value() - predict({(int)it.col(), (int)it.row()}, model));
        }
    }
    return sqrt( se / Y.nonZeros() );
}

void ScarceBinaryMatrixData::get_pnm(const SubModel &model, int mode, int n, VectorXd &rr, MatrixXd &MM)
{
    // todo : check noise == probit noise
    auto u = model.U(mode).col(n);
    for (SparseMatrix<double>::InnerIterator it(Yc.at(mode), n); it; ++it) {
        const auto &col = model.V(mode).col(it.row());
        MM.noalias() += col * col.transpose();
		double y = 2 * it.value() - 1;
		auto z = y * rand_truncnorm(y * col.dot(u), 1.0, 0.0);
        rr.noalias() += col * z;
    }
}


//
//-- FullMatrixData specific stuff
//

template<class YType>
void FullMatrixData<YType>::get_pnm(const SubModel &model, int mode, int d, VectorXd &rr, MatrixXd &MM) {
    const double alpha = this->noise().getAlpha();
    auto &Y = this->Yc.at(mode);
    rr.noalias() += (model.V(mode) * Y.col(d)) * alpha;
    MM.noalias() += VV[mode] * alpha;
}

template<class YType>
void FullMatrixData<YType>::update_pnm(const SubModel &model, int mode) {
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


template<class YType>
double FullMatrixData<YType>::compute_mode_mean(int m, int c)
{
    const auto &col = this->Yc.at(m).col(c);
    if (col.nonZeros() == 0) return this->cwise_mean;
    return col.sum() / this->Yc.at(m).rows();
}


template struct FullMatrixData<Eigen::MatrixXd>;
template struct FullMatrixData<Eigen::SparseMatrix<double>>;


void SparseMatrixData::center(double global_mean)
{
    this->global_mean = global_mean;
    if (center_mode == CENTER_GLOBAL) {
        Yc.at(0).coeffs() -= global_mean;
        Yc.at(1).coeffs() -= global_mean;
    } else if (center_mode == CENTER_VIEW) {
        Yc.at(0).coeffs() -= cwise_mean;
        Yc.at(1).coeffs() -= cwise_mean;
    } else if (center_mode == CENTER_COLS) {
        // you cannot col/row center fully know sparse matrices
        // without converting to dense
        assert(false);
    } else if (center_mode == CENTER_ROWS) {
        // you cannot col/row center fully know sparse matrices
        // without converting to dense
        assert(false);
    }
}

double SparseMatrixData::train_rmse(const SubModel &model) const {
    double se = 0.;
#pragma omp parallel for schedule(guided) reduction(+:se)
    for(int c=0; c<Y.cols();++c) {
        int r = 0;
        for (SparseMatrix<double>::InnerIterator it(Y, c); it; ++it) {
            while (r<it.row()) se += square(predict({c,r++}, model));
            se += square(it.value() - predict({c,r}, model));
        }
        for(;r<Y.rows();r++) se += square(predict({c,r}, model));
    }
    return sqrt( se / Y.rows() / Y.cols() );
}

void DenseMatrixData::center(double global_mean)
{
    this->global_mean = global_mean;
    if (center_mode == CENTER_GLOBAL) {
        Yc.at(0).array() -= global_mean;
        Yc.at(1).array() -= global_mean;
    } else if (center_mode == CENTER_VIEW) {
        Yc.at(0).array() -= cwise_mean;
        Yc.at(1).array() -= cwise_mean;
    } else if (center_mode == CENTER_COLS) {
        Yc.at(0).rowwise() -= mode_mean.at(0).transpose();
        Yc.at(1) = Yc.at(0).transpose();
    } else if (center_mode == CENTER_ROWS) {
        Yc.at(1).rowwise() -= mode_mean.at(1).transpose();
        Yc.at(0) = Yc.at(1).transpose();
    }
}

double DenseMatrixData::train_rmse(const SubModel &model) const
{
    double se = 0.;
#pragma omp parallel for schedule(guided) reduction(+:se)
    for(int c=0; c<Y.cols();++c) {
        for(int m=0; m<Y.rows(); ++m) {
            se += square(Y(m,c) - predict({c,m}, model));
        }
    }
    return sqrt( se / Y.rows() / Y.cols() );
}



} //end namespace smurff

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
