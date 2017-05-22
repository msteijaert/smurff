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

#include "utils.h"
#include "model.h"
#include "mvnormal.h"

using namespace std; 
using namespace Eigen;

namespace Macau {

int Model::num_latent = -1;

void Result::set(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
    for(int i=0; i<N; ++i) {
        predictions.push_back({rows[i], cols[i], values[i]});
    }
    this->nrows = nrows;
    this->ncols = ncols;
    init();
}
 
void Result::set(SparseDoubleMatrix &Y) {
    for(unsigned i=0; i<Y.nnz; ++i) {
        predictions.push_back({Y.rows[i], Y.cols[i], Y.vals[i]});
    }
    nrows = Y.nrow;
    ncols = Y.ncol;
    init();
}

void Result::set(SparseMatrixD Y) {
    for (int k = 0; k < Y.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
            predictions.push_back({(int)it.row(), (int)it.col(), it.value()});
        }
    }
    nrows = Y.rows();
    ncols = Y.cols();
    init();
}

void Result::init() {
    total_pos = 0;
    if (classify) {
        for(auto &t : predictions) {
            int is_positive = t.val > threshold;
            total_pos += is_positive;
        }
    }
}

//--- output model to files
void Result::save(std::string prefix) {
    if (predictions.empty()) return;
    std::string fname_pred = prefix + "-predictions.csv";
    std::ofstream predfile;
    predfile.open(fname_pred);
    predfile << "row,col,y,pred_1samp,pred_avg,var,std\n";
    for ( auto &t : predictions) {
        predfile
                << to_string( t.row  )
         << "," << to_string( t.col  )
         << "," << to_string( t.val  )
         << "," << to_string( t.pred )
         << "," << to_string( t.pred_avg )
         << "," << to_string( t.var )
         << "," << to_string( t.stds )
         << "\n";
    }
    predfile.close();

}

void Model::save(std::string prefix, std::string suffix) {
    int i = 0;
    for(auto &U : factors) {
        write_dense(prefix + "-U" + std::to_string(i++) + "-latents" + suffix, U);
    }
}

void Model::restore(std::string prefix, std::string suffix) {
    int i = 0;
    for(auto &U : factors) {
        read_dense(prefix + "-U" + std::to_string(i++) + "-latents" + suffix, U);
    }
}

///--- update RMSE and AUC

double Result::modemean_rmse(int mode, const Model &model) {
    const unsigned N = predictions.size();
    double se = 0.;
    double pred;
    for(auto t : predictions) {
        int n = mode == 0 ? t.col : t.row;
        pred = mode >= 0 ? model.mean(mode, n) : .0;
        pred += model.mean_rating;
        se += square(t.val - pred);
    }
    return sqrt( se / N );
}

void Result::update(const Model &model, bool burnin)
{
    if (predictions.size() == 0) return;
    const unsigned N = predictions.size();

    if (burnin) {
        double se = 0.0;
#pragma omp parallel for schedule(guided) reduction(+:se)
        for(unsigned k=0; k<predictions.size(); ++k) {
            auto &t = predictions[k];
            t.pred = model.predict(t.row, t.col);
            se += square(t.val - t.pred);
        }
        burnin_iter++;
        rmse = sqrt( se / N );
    } else {
        double se = 0.0, se_avg = 0.0;
#pragma omp parallel for schedule(guided) reduction(+:se, se_avg)
        for(unsigned k=0; k<predictions.size(); ++k) {
            auto &t = predictions[k];
            const double pred = model.predict(t.row, t.col);
            se += square(t.val - pred);
            double delta = pred - t.pred_avg;
            double pred_avg = (t.pred_avg + delta / (sample_iter + 1));
            t.var += delta * (pred - pred_avg);
            const double inorm = 1.0 / sample_iter;
            t.stds = sqrt(t.var * inorm);
            t.pred_avg = pred_avg;
            t.pred = pred;
            se_avg += square(t.val - pred_avg);
        }
        sample_iter++;
        rmse = sqrt( se / N );
        rmse_avg = sqrt( se_avg / N );
    }

    update_auc();
}


void Result::update_auc()
{
    if (!classify) return;
    std::sort(predictions.begin(), predictions.end(),
            [this](const Item &a, const Item &b) { return a.pred < b.pred;});

    int num_positive = 0;
    int num_negative = 0;
    auc = .0;
    for(auto &t : predictions) {
        int is_positive = t.val > threshold;
        int is_negative = !is_positive; 
        num_positive += is_positive;
        num_negative += is_negative;
        auc += is_positive * num_negative;
    }

    auc /= num_positive;
    auc /= num_negative;
}

std::ostream &Model::info(std::ostream &os, std::string indent)
{
    os << indent << "Type: " << name << "\n";
    os << indent << "Num-latents: " << num_latent << "\n";
    double train_fill_rate = 100. * Ynnz() / Yrows() / Ycols();
    os << indent << "Train data: " << Ynnz() << " [" << Yrows() << " x " << Ycols() << "] (" << train_fill_rate << "%)\n";
    return os;
}

std::ostream &Result::info(std::ostream &os, std::string indent, const Model &model)
{
    if (predictions.size()) {
        double test_fill_rate = 100. * predictions.size() / nrows / ncols;
        os << indent << "Test data: " << predictions.size() << " [" << nrows << " x " << ncols << "] (" << test_fill_rate << "%)\n";
        os << indent << "RMSE using globalmean: " << modemean_rmse(-1,model) << endl;
        os << indent << "RMSE using colmean: " << modemean_rmse(0,model) << endl;
        os << indent << "RMSE using rowmean: " << modemean_rmse(1,model) << endl;
     } else {
        os << indent << "Test data: -\n";
    }
    if (classify) {
        double pos = 100. * (double)total_pos / (double)predictions.size();
        os << indent << "Binary classification threshold: " << threshold << "\n";
        os << indent << "  " << pos << "% positives in test data\n";
    }
    return os;
}

template<typename YType>
void MF<YType>::init_base()
{
    assert(Yrows() > 0 && Ycols() > 0);

//    if (pred.ncols > 0) {
//        assert(pred.nrows == Yrows() && pred.ncols == Ycols() && "Size of train must be equal to size of test");
//    }

    mean_rating = Y.sum() / Y.nonZeros();

    U(0).resize(num_latent, Y.cols());
    U(1).resize(num_latent, Y.rows());

    bmrandn(U(0));
    bmrandn(U(1));

    Yc.push_back(Y);
    Yc.push_back(Y.transpose());
}

template<>
void MF<SparseMatrixD>::init()
{
    init_base();
    Yc.at(0).coeffs() -= mean_rating;
    Yc.at(1).coeffs() -= mean_rating;
    name = name + " [with NAs]";
}


template<>
void MF<Eigen::MatrixXd>::init()
{
    init_base();
    Yc.at(0).array() -= this->mean_rating;
    Yc.at(1).array() -= this->mean_rating;
    name = "Dense" + name;
}

template<>
double MF<Eigen::MatrixXd>::var_total() const {
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
double MF<SparseMatrixD>::var_total() const {
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
double MF<Eigen::MatrixXd>::sumsq() const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < this->Y.cols(); j++) {
        auto Vj = this->U(0).col(j);
        for (int i = 0; i < this->Y.rows(); i++) {
            double Yhat = Vj.dot( this->U(1).col(i) ) + this->mean_rating;
            sumsq += square(Yhat - this->Y(i,j));
        }
    }

    return sumsq;
}

template<>
double MF<SparseMatrixD>::sumsq() const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < Y.outerSize(); j++) {
        auto Uj = col(0, j);
        for (SparseMatrix<double>::InnerIterator it(Y, j); it; ++it) {
            double Yhat = Uj.dot( col(1, it.row()) ) + mean_rating;
            sumsq += square(Yhat - it.value());
        }
    }

    return sumsq;
}

//
//-- SparseMF specific stuff
//
//
//

void SparseMF::init() {
    MF<SparseMatrixD>::init();

    // check no rows, nor cols withouth data
    for(unsigned i=0; i<Yc.size(); ++i) {
        auto &v = Yc[i];
        auto &count = num_empty[i]; 
        for (int j = 0; j < v.cols(); j++) {
            if (v.col(j).nonZeros() == 0) count++; 
        }
    }
}

std::ostream &SparseMF::info(std::ostream &os, std::string indent) {
    MF<SparseMatrixD>::info(os, indent);
    if (num_empty[0]) os << indent << "  Warning: " << num_empty[0] << " empty cols\n"; 
    if (num_empty[1]) os << indent << "  Warning: " << num_empty[0] << " empty rows\n"; 
    return os;
}

struct pnm_perf_item {
    int local_nnz, total_nnz;
    bool in_parallel;
    double start, stop;
};

static std::vector<pnm_perf_item> pnm_perf;

void SparseMF::get_pnm(int f, int n, VectorXd &rr, MatrixXd &MM) {
    auto &Y = Yc.at(f);
    MatrixXd &Vf = V(f);
    const int local_nnz = Y.col(n).nonZeros();
    const int total_nnz = Y.nonZeros();
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
            }
        }
#pragma omp taskwait
        MM += MMs.combine();
        rr += rrs.combine();
    } else {
        for (SparseMatrix<double>::InnerIterator it(Y, n); it; ++it) {
            const auto &col = Vf.col(it.row());
            rr.noalias() += col * it.value();
            MM.triangularView<Lower>() += col * col.transpose();
        }
    }

    MM.triangularView<Upper>() = MM.transpose();
}

void SparseMF::update_pnm(int f) {
    return;
    if (pnm_perf.size()) {
        printf("==========\n"); 
        for(auto &item: pnm_perf) printf("%d;%d;%d;%f;%f\n", item.local_nnz, item.total_nnz, item.in_parallel, item.start, item.stop);
        pnm_perf.clear();
    }

    auto &Y = Yc.at(f);

    int bin = 1;
    int count = 0;
    int total = 0;
    while (count < Y.cols()) {
        count = 0;
        for(int i=0; i<Y.cols();++i) if (Y.col(i).nonZeros() < bin) count++;
        auto bin_count = count - total;
        auto bin_nnz = bin_count * bin;
        auto bin_percent = (100. * bin_nnz) / Y.nonZeros();
            printf("fac: %d\t%5d < bin < %5d;\t#samples: %4d;\t%5d < #nnz < %5d;\t %.1f < %%nnz < %.1f\n",
                    f, bin/2, bin, bin_count, bin_nnz/2, bin_nnz, bin_percent/2, bin_percent);
        bin *= 2;
        total = count;
    }

    std::cout << "Total samples: " << Y.cols() << std::endl;
    std::cout << "Total nnz: " << Y.nonZeros() << std::endl;
}

void SparseBinaryMF::get_pnm(int f, int n, VectorXd &rr, MatrixXd &MM)
{
    auto u = U(f).col(n);
    for (SparseMatrix<double>::InnerIterator it(Yc.at(f), n); it; ++it) {
        const auto &col = V(f).col(it.row());
        MM.noalias() += col * col.transpose();
        auto z = (2 * it.value() - 1) * fabs(col.dot(u) + bmrandn_single());
        rr.noalias() += col * z;
    }
}


//
//-- DenseMF specific stuff
//

template<class YType>
void DenseMF<YType>::get_pnm(int f, int d, VectorXd &rr, MatrixXd &MM) {
    auto &Y = this->Yc.at(f);
    rr.noalias() += (this->V(f) * Y.col(d));
    MM.noalias() += VV.at(f); 
}

template<class YType>
void DenseMF<YType>::update_pnm(int f) {
    auto &Vf = this->V(f);
    thread_vector<MatrixXd> VVs(MatrixXd::Zero(this->num_latent, this->num_latent));

#pragma omp parallel for schedule(dynamic, 8) shared(VVs)
    for(int n = 0; n < Vf.cols(); n++) {
        auto v = Vf.col(n);
        VVs.local() += v * v.transpose();
    }

    VV.at(f) = VVs.combine();
}

template struct MF<SparseMatrix<double>>;
template struct MF<MatrixXd>;
template struct DenseMF<Eigen::MatrixXd>;
template struct DenseMF<SparseMatrixD>;

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
