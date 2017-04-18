#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "utils.h"

namespace Macau {

struct Factors {
    static int num_latent;

    //-- c'tor
    Factors(int nl, int num_fac = 2)
    {
        assert(num_fac == 2); 
        assert(num_latent == -1 || num_latent == nl);
        num_latent = nl;
        factors.resize(num_fac);
    }


    const Eigen::MatrixXd &U(int f) const { return factors.at(f); }
    Eigen::MatrixXd &U(int f) { return factors.at(f); }
    Eigen::MatrixXd &V(int f) { return factors.at((f+1)%2); }
    Eigen::MatrixXd::ConstColXpr col(int f, int i) const { return U(f).col(i); }

    int num_fac() const { return factors.size(); }


    // Ytest related
    struct YTestItem {
        int row, col;
        double val, pred, var, stds;
    };
    std::vector<YTestItem> Ytest;
    int Ytestrows, Ytestcols;
    void update_predictions(int iter, int burnin);
    double rmse_avg = NAN, rmse = NAN, auc = NAN; 
    const int num_bins = 10000;
    int total_pos;
    std::vector<unsigned> num_pos;
    std::vector<unsigned> num_neg;
    std::vector<double> bin_bounds;

    void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void setRelationDataTest(SparseDoubleMatrix &Y);
    void setRelationDataTest(Eigen::SparseMatrix<double> Y);
    void setThreshold(double t) { threshold = t; classify = true; } 

    std::pair<double,double> getRMSE(int iter, int burnin);

    // helper functions for noise
    virtual double sumsq() const = 0;
    virtual double var_total() const = 0;

    // helper functions for priors
    virtual void get_pnm(int,int,VectorNd &, MatrixNNd &) = 0;
    virtual void update_pnm(int) = 0;
 
    //-- output to file
    void saveGlobalParams(std::string);
    void savePredictions(std::string, int iter, int burnin);
    void saveModel(std::string, int iter, int burnin);
    std::ostream &printInitStatus(std::ostream &os, std::string indent);

    // virtual functions Y-related
    double mean_rating = .0;
    virtual void init() = 0;
    virtual int Yrows()    const = 0;
    virtual int Ycols()    const = 0;
    virtual int Ynnz ()    const = 0;

    std::string name;

  private:
    void init_predictions();
    int last_iter = -1;
    bool classify = false;
    double threshold;
    std::vector<Eigen::MatrixXd> factors;
    // AUC related
    std::vector<unsigned int> permutation;
};

template<typename YType>
struct MF : public Factors {
    //-- c'tor
    MF(int num_latent, int num_fac = 2)
        : Factors(num_latent, num_fac) { }

    void init_base();
    void init() override;

    int Yrows()   const override { return Y.rows(); }
    int Ycols()   const override { return Y.cols(); }
    int Ynnz()    const override { return Y.nonZeros(); }

    void setRelationData(YType Y);
    void setRelationData(SparseDoubleMatrix &Y);
    void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols);

    double var_total() const override;
    double sumsq() const override;

    YType Y;
    std::vector<YType> Yc; // centered version
};

struct SparseMF : public MF<SparseMatrixD> {
    //-- c'tor
    SparseMF(int num_latent, int num_fac = 2)
        : MF<SparseMatrixD>(num_latent, num_fac)
    {
        name = "SparseMF";
    }

    void get_pnm(int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(int) override;
};

struct SparseBinaryMF : public MF<SparseMatrixD> {
    //-- c'tor
    SparseBinaryMF(int num_latent, int num_fac = 2)
        : MF<SparseMatrixD>(num_latent, num_fac)
    {
        name = "SparseBinaryMF (Probit Noise Sampler)";
    }

    void get_pnm(int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(int) override {}
};

template<class YType>
struct DenseMF : public MF<YType> {
    //-- c'tor
    DenseMF(int num_latent, int num_fac = 2)
        : MF<YType>(num_latent, num_fac) 
    {
        VV.resize(num_fac);
        this->name = "DenseMF";
    }

    void get_pnm(int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(int) override;

  private:
    std::vector<MatrixNNd> VV;
};

typedef DenseMF<Eigen::MatrixXd> DenseDenseMF;
typedef DenseMF<SparseMatrixD> SparseDenseMF;

}

#endif
