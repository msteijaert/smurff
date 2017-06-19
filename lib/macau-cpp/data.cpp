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

namespace smurff {


////----- Data below


std::ostream &Data::info(std::ostream &os, std::string indent)
{
    os << indent << "Type: " << name << "\n";
    os << indent << "Component-wise mean: " << cwise_mean << "\n";
    std::vector<std::string> center_names { "none", "global", "view", "cols", "rows" };
    os << indent << "Center: " << center_names.at(center_mode + 3) << "\n";
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

void Data::setCenterMode(std::string c)
{
    //-- centering model
         if (c == "none")   center_mode = CENTER_NONE;
    else if (c == "global") center_mode = CENTER_GLOBAL;
    else if (c == "view")   center_mode = CENTER_VIEW;
    else if (c == "rows")   center_mode = CENTER_ROWS;
    else if (c == "cols")   center_mode = CENTER_COLS;
    else assert(false);
}

double Data::predict(const PVec &pos, const SubModel &model) const
{
       return model.dot(pos) + offset_to_mean(pos);
}


std::ostream &MatrixData::info(std::ostream &os, std::string indent)
{
    Data::info(os, indent);
    double train_fill_rate = 100. * nnz() / size();
    os << indent << "Size: " << nnz() << " [" << nrow() << " x " << ncol() << "] (" << train_fill_rate << "%)\n";
    return os;
}

MatrixData& MatricesData::add(int row, int col, std::unique_ptr<MatrixData> c) {
    auto pos = std::make_pair(row, col);
    matrices[pos] = std::move(c);
    return *matrices[pos];
}


PVec MatricesData::bdims(int brow, int bcol) const
{
    return PVec(coldims.find(bcol)->second, rowdims.find(brow)->second);
}


PVec MatricesData::boffs(int brow, int bcol) const
{
    PVec ret(0,0);
    for(int i=0; i<brow; ++i) ret[1] += rowdims.find(i)->second;
    for(int i=0; i<bcol; ++i) ret[0] += coldims.find(i)->second;
    return ret;
}

SubModel MatricesData::submodel(const SubModel &model, int brow, int bcol) 
{
    return SubModel(model, boffs(brow, bcol), bdims(brow, bcol));
}


void MatricesData::get_pnm(const SubModel &model, int mode, int pos, VectorNd &rr, MatrixNNd &MM) {
    int count = 0;
    for(auto &p : matrices) {
        int brow = p.first.first;
        int bcol = p.first.second;

        auto off = boffs(brow, bcol);
        auto dim = bdims(brow, bcol);

        if (off[mode] > pos || off[mode] + dim[mode] <= pos) continue;

        p.second->get_pnm(submodel(model, brow, bcol), mode, pos - off[mode], rr, MM);
        count++;
    }
    assert(count>0);
}

void MatricesData::update_pnm(const SubModel &model, int m) {
    for(auto &p : matrices) {
        p.second->update_pnm(model, m);
    }
}

std::ostream &MatricesData::info(std::ostream &os, std::string indent)
{
    MatrixData::info(os, indent);
    os << indent << "Sub-Matrices:\n";
    for(auto &p : matrices) {
        os << indent <<  "[ " << p.first.first << "," << p.first.second << " ]:\n";
           p.second->info(os, indent + "  ");
           os << std::endl;
    }
    return os;
}

void MatricesData::init_base() 
{
    // FIXME: noise!
    for(auto &p : matrices) p.second->setPrecision(5.);

    // init coldims and 
    for(auto &p : matrices) {
        auto row = p.first.first;
        auto col = p.first.second;
        auto &m = p.second;

        if (coldims.find(col) != coldims.end()) {
            assert(coldims[col] = m->ncol());
        } else {
            coldims[col] = m->ncol();
        }

        if (rowdims.find(row) != rowdims.end()) {
            assert(rowdims[row] = m->nrow());
        } else {
            rowdims[row] = m->nrow();
        }
    }

    cwise_mean = sum() / (double)(size() - nna());

    // init sub-matrices
    for(auto &p : matrices) {
        p.second->init_base();
        p.second->compute_mode_mean();
    }
}

void MatricesData::setCenterMode(std::string mode) 
{
    Data::setCenterMode(mode);
    for(auto &p : matrices) p.second->setCenterMode(mode);
}


void MatricesData::center(double global_mean) 
{
    // center sub-matrices
    assert(global_mean == cwise_mean);
    this->global_mean = global_mean;
    for(auto &p : matrices) p.second->center(cwise_mean);
}

double MatricesData::compute_mode_mean(int mode, int pos) {
    double sum = .0;
    int N = 0;
    int count = 0;


    for(auto &p : matrices) {
        int brow = p.first.first;
        int bcol = p.first.second;

        auto off = boffs(brow, bcol);
        auto dim = bdims(brow, bcol);

        if (off[mode] > pos || off[mode] + dim[mode] <= pos) continue;

        double local_mean = p.second->mean(mode, pos - off[mode]);
        sum += local_mean * dim[mode];
        N += dim[mode];
        count++;
    }

    assert(N>0);

    return sum / N;
}

double MatricesData::offset_to_mean(const PVec &pos) const {
    for(auto &p : matrices) {
        int brow = p.first.first;
        int bcol = p.first.second;

        auto off = boffs(brow, bcol);
        auto dim = bdims(brow, bcol);

        if (off[0] > pos[0] || off[0] + dim[0] <= pos[0]) continue;
        if (off[1] > pos[1] || off[1] + dim[1] <= pos[1]) continue;

        return p.second->offset_to_mean({pos[0] - off[0], pos[1] - off[1]});
    }
    assert(false);
    return .0;
}

double MatricesData::train_rmse(const SubModel &model) const {
    double sum = .0;
    int N = 0;
    int count = 0;


    for(auto &p : matrices) {
        int brow = p.first.first;
        int bcol = p.first.second;

        auto off = boffs(brow, bcol);
        auto dim = bdims(brow, bcol);

        auto &mtx = *p.second;
        double local_rmse = mtx.train_rmse(SubModel(model, off, dim));
        sum += (local_rmse * local_rmse) * (mtx.size() - mtx.nna());
        N += (mtx.size() - mtx.nna());
        count++;
    }

    assert(N>0);

    return sqrt(sum / N);
}

template<typename YType>
void MatrixDataTempl<YType>::init_base()
{
    assert(nrow() > 0 && ncol() > 0);

    Yc.push_back(Y);
    Yc.push_back(Y.transpose());

    noise->init();
    cwise_mean = sum() / (size() - nna());
}

void Data::init()
{
    init_base();

    //compute global mean & mode-wise means
    compute_mode_mean();
    center(cwise_mean);
}

void Data::compute_mode_mean()
{
    assert(!mean_computed);
    mode_mean.resize(nmode());
    for(int m=0; m<nmode(); ++m) {
        auto &M = mode_mean.at(m);
        const auto d = dims().at(m);
        M.resize(d);
        for(int n=0; n<d; n++) M(n) = compute_mode_mean(m, n);
    }

    mean_computed = true;
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
double MatrixDataTempl<Eigen::MatrixXd>::sumsq(const SubModel &model) const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < this->ncol(); j++) {
        for (int i = 0; i < this->nrow(); i++) {
            double Yhat = model.dot({i,j}) + offset_to_mean({i,j});
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
            double Yhat = model.dot({i,j}) + offset_to_mean({i,j});
            sumsq += square(Yhat - it.value());
        }
    }

    return sumsq;
}

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

void ScarceMatrixData::init_base() {
    MatrixDataTempl<SparseMatrixD>::init_base();

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
    const double alpha = noise->getAlpha();

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
        auto z = (2 * it.value() - 1) * fabs(col.dot(u) + bmrandn_single());
        rr.noalias() += col * z;
    }
}


//
//-- FullMatrixData specific stuff
//

template<class YType>
void FullMatrixData<YType>::get_pnm(const SubModel &model, int mode, int d, VectorXd &rr, MatrixXd &MM) {
    const double alpha = this->noise->getAlpha();
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
