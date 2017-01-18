#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

struct SparseDoubleMatrix;

struct Factors {
    int num_latent;

    //-- c'tor
    Factors(int num_latent, int num_fac = 2)
        : num_latent(num_latent), iter(0)
    {
        assert(num_fac == 2); 
        factors.resize(num_fac);
    }

    virtual void init();

    const Eigen::MatrixXd &U(int f) const { return factors.at(f); }
    Eigen::MatrixXd &U(int f) { return factors.at(f); }
    Eigen::MatrixXd::ConstColXpr col(int f, int i) const { return U(f).col(i); }
    int num_fac() const { return factors.size(); }

    std::vector<Eigen::MatrixXd> factors;

    Eigen::SparseMatrix<double> Ytest;
    Eigen::VectorXd predictions, predictions_var, test_vector;

    double iter;
    double rmse = NAN, rmse_avg = NAN;
    double auc = NAN, auc_avg = NAN;

    void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void setRelationDataTest(SparseDoubleMatrix &Y);
    void setRelationDataTest(Eigen::SparseMatrix<double> Y);

    void update_rmse(bool burnin);
    void update_auc(bool burnin);

    Eigen::VectorXd getStds(int iter);
    Eigen::MatrixXd getTestData();

    // Y-related
    double mean_rating = .0;
    virtual int Yrows() const = 0;
    virtual int Ycols() const = 0;
};

typedef Eigen::SparseMatrix<double> SparseMatrixD;

struct SparseMF : public Factors {
    //-- c'tor
    SparseMF(int num_latent, int num_fac = 2)
        : Factors(num_latent, num_fac) {}

    void init() override;

    void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void setRelationData(SparseDoubleMatrix& Y);

    int Yrows() const override { return Y.rows(); }
    int Ycols() const override { return Y.cols(); }

    SparseMatrixD Y;
};

struct DenseMF : public Factors {
    //-- c'tor
    DenseMF(int num_latent, int num_fac = 2)
        : Factors(num_latent, num_fac),
          UtU(num_fac, Eigen::MatrixXd::Zero(num_latent, num_latent)),
          UU(num_fac, Eigen::MatrixXd::Zero(num_latent, num_latent))
    {
        Ut.resize(num_fac);
    }

    void init() override;

    std::vector<Eigen::MatrixXd> Ut;
    std::vector<Eigen::MatrixXd> UtU;
    std::vector<Eigen::MatrixXd> UU;

    int Yrows() const override { return Y.rows(); }
    int Ycols() const override { return Y.cols(); }
    void setRelationData(Eigen::MatrixXd Y);

    Eigen::MatrixXd Y;
};

#endif /* MODEL_H */
