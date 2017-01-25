#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

struct SparseDoubleMatrix;
typedef Eigen::SparseMatrix<double> SparseMatrixD;

struct Factors {
    int num_latent;

    //-- c'tor
    Factors(int num_latent, int num_fac = 2)
        : num_latent(num_latent)
    {
        assert(num_fac == 2); 
        factors.resize(num_fac);
    }


    const Eigen::MatrixXd &U(int f) const { return factors.at(f); }
    Eigen::MatrixXd &U(int f) { return factors.at(f); }
    Eigen::MatrixXd::ConstColXpr col(int f, int i) const { return U(f).col(i); }
    int num_fac() const { return factors.size(); }

    std::vector<Eigen::MatrixXd> factors;

    Eigen::SparseMatrix<double> Ytest;

    void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void setRelationDataTest(SparseDoubleMatrix &Y);
    void setRelationDataTest(Eigen::SparseMatrix<double> Y);

    std::pair<double,double> getRMSE(int iter, int burnin);
    const Eigen::VectorXd &getPredictions(int iter, int burnin);
    const Eigen::VectorXd &getPredictionsVar(int iter, int burnin);
    const Eigen::VectorXd &getStds(int iter, int burnin);

    double auc();

    virtual double sumsq() const = 0;
    virtual double var_total() const = 0;

    //-- output to file
    void saveGlobalParams(std::string);
    void savePredictions(std::string, int iter, int burnin);
    void saveModel(std::string, int iter, int burnin);

    // virtual functions Y-related
    double mean_rating = .0;
    virtual int Yrows()    const = 0;
    virtual int Ycols()    const = 0;
    virtual int Ynnz ()    const = 0;

  private:
    void init_predictions();
    void update_predictions(int iter, int burnin);
    double rmse_avg = NAN, rmse = NAN; 
    int last_iter = -1;
    Eigen::VectorXd predictions, predictions_var, stds;
};

template<typename YType>
struct MF : public Factors {
    //-- c'tor
    MF(int num_latent, int num_fac = 2) : Factors(num_latent, num_fac) {}

    void init();

    int Yrows()   const override { return Y.rows(); }
    int Ycols()   const override { return Y.cols(); }
    int Ynnz()    const override { return Y.nonZeros(); }

    void setRelationData(YType Y);

    YType Y;
};

struct SparseMF : public MF<SparseMatrixD> {
    //-- c'tor
    SparseMF(int num_latent, int num_fac = 2)
        : MF<SparseMatrixD>(num_latent, num_fac) {}

    void setRelationData(SparseDoubleMatrix &Y);
    void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols);

    double var_total() const override;
    double sumsq() const override;
};

struct DenseMF : public MF<Eigen::MatrixXd> {
    //-- c'tor
    DenseMF(int num_latent, int num_fac = 2)
        : MF<Eigen::MatrixXd>(num_latent, num_fac) {}

    double var_total() const override;
    double sumsq() const override;
};

#endif
