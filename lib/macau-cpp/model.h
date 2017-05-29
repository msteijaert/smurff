#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "matrix_io.h"
#include "utils.h"

namespace Macau {

struct Model;

struct Result {
    //-- test set
    struct Item {
        int row, col;
        double val, pred, pred_avg, var, stds;
    };
    std::vector<Item> predictions;
    int nrows, ncols;
    void set(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void set(SparseDoubleMatrix &Y);
    void set(Eigen::SparseMatrix<double> Y);


    //-- prediction metrics
    double colmean_rmse(const Model &);
    double globalmean_rmse(const Model &);
    void update(const Model &, bool burnin);
    double rmse_avg = NAN;
    double rmse = NAN;
    double auc = NAN; 
    int sample_iter = 0;
    int burnin_iter = 0;

    // general
    void save(std::string fname_prefix);
    void init();
    std::ostream &info(std::ostream &os, std::string indent, const Model &model);

    //-- for binary classification
    int total_pos;
    bool classify = false;
    double threshold;
    void update_auc();
    void setThreshold(double t) { threshold = t; classify = true; } 
};

struct Model {
    static int num_latent;

    //-- c'tor
    Model(int nl, int num_fac = 2) {
        assert(num_fac == 2); 
        assert(num_latent == -1 || num_latent == nl);
        num_latent = nl;
        factors.resize(num_fac);
    }

    const Eigen::MatrixXd &U(int f) const { return factors.at(f); }
    Eigen::MatrixXd &U(int f) { return factors.at(f); }
    Eigen::MatrixXd &V(int f) { return factors.at((f+1)%2); }
    Eigen::MatrixXd::ConstColXpr col(int f, int i) const { return U(f).col(i); }
    double predict(int r, int c) const  {
        return col(0,c).dot(col(1,r)) + offset_to_mean(r, c);
    }

    int num_fac() const { return factors.size(); }

    // helper functions for noise
    virtual double sumsq() const = 0;
    virtual double var_total() const = 0;

    // helper functions for priors
    virtual void get_pnm(int,int,VectorNd &, MatrixNNd &) = 0;
    virtual void update_pnm(int) = 0;
 
    //-- output to file
    void save(std::string, std::string);
    void restore(std::string, std::string);
    virtual std::ostream &info(std::ostream &os, std::string indent);

    // virtual functions Y-related
    virtual void init() = 0;
    virtual int Yrows()    const = 0;
    virtual int Ycols()    const = 0;
    virtual int Ynnz ()    const = 0;
    virtual double  offset_to_mean(int row, int col) const = 0;

    // col/row-wise mean for a simple predictor
    virtual double colmean(int) const = 0;
    double global_mean = .0;

    std::string name;
    enum { CENTER_NONE = 0, CENTER_GLOBAL, CENTER_COLS, CENTER_ROWS } center;
  protected:
    Eigen::VectorXd mean_vec;

  private:
    std::vector<Eigen::MatrixXd> factors;
};

template<typename YType>
struct MF : public Model {
    //-- c'tor
    MF(int num_latent, int num_fac = 2)
        : Model(num_latent, num_fac) { }

    void init_base();
    void init() override;

    int Yrows()   const override { return Y.rows(); }
    int Ycols()   const override { return Y.cols(); }
    int Ynnz()    const override { return Y.nonZeros(); }
    double  offset_to_mean(int row, int col) const override;

    YType center_cols(Eigen::VectorXd &);
    double colmean(int c) const override { 
        auto &col = Y.col(c);
        if (col.nonZeros() == 0) return global_mean;
        return col.sum() / col.nonZeros();
    }

    void setRelationData(YType Y) { 
        this->Y = Y;
    }
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
    void init() override;

    std::ostream &info(std::ostream &os, std::string indent) override;

    void get_pnm(int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(int) override;
  private:
    int num_empty[2] = {0,0}; 

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
