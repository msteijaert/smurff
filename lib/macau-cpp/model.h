#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

struct SparseDoubleMatrix;

template<typename YType>
struct IFactor {
    //-- c'tor
    IFactor(int D) : num_latent(D) {}
    void init();

    int num_latent;
    double mean_rating = .0; 

    YType Y;

    // get vector
    Eigen::MatrixXd::ConstColXpr col(int i) const { return U.col(i); }
    // set vector
    template<typename T>
    Eigen::MatrixXd::ConstColXpr col(int i, const T &v) {
        U.col(i) = v;
        return U.col(i); 
    }
    
    int num() const { return U.cols(); }

    Eigen::MatrixXd U;
};


template<typename YType>
struct MF {
    typedef IFactor<YType> F;

    //-- c'tor
    MF(int num_latent, int num_fac = 2)      {
        assert(num_fac == 2); 
        for(int i=0; i<num_fac; ++i) factors.push_back(IFactor<YType>(num_latent));
    }

    const IFactor<YType> &fac(int f) const { return factors.at(f); }
    IFactor<YType> &fac(int f) { return factors.at(f); }
    int num_latent() const { return fac(0).num_latent; }
    double mean_rating() const { return fac(0).mean_rating; }
    YType &Y() { return fac(0).Y; }
    YType &Yt() { return fac(1).Y; }

    Eigen::MatrixXd::ConstColXpr col(int f, int i) const { return fac(f).col(i); }
    Eigen::MatrixXd &U(int f) { return fac(f).U; }

    std::vector<IFactor<YType>> factors;

    void init();
};

typedef IFactor<Eigen::SparseMatrix<double>> Factor;

struct SparseMF : MF<Eigen::SparseMatrix<double>> {
      //-- c'tor
      SparseMF(int num_latent, int num_fac = 2)
          : MF<Eigen::SparseMatrix<double>>(num_latent, num_fac), iter(0) {}

      Eigen::SparseMatrix<double> Ytest;
      Eigen::VectorXd predictions, predictions_var, test_vector;

      double iter;
      double rmse, rmse_avg;
      double auc, auc_avg;

      void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols);
      void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols);
      void setRelationData(SparseDoubleMatrix& Y);
      void setRelationDataTest(SparseDoubleMatrix &Y);

      void update_rmse(bool burnin);
      void update_auc(bool burnin);

      Eigen::VectorXd getStds(int iter);
      Eigen::MatrixXd getTestData();

      void init();
};

struct DenseFactor : public IFactor<Eigen::MatrixXd> {
      Eigen::MatrixXd UU, noiseUU, CovF, CovL;
};

struct DenseMF : MF<Eigen::MatrixXd> {
      //-- c'tor
      DenseMF(int num_latent, int num_fac = 2)
          : MF<Eigen::MatrixXd>(num_latent, num_fac) {}


};

#endif /* MODEL_H */
