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

    void init();
};

typedef Eigen::SparseMatrix<double> SparseMatrixD;

template<typename YType>
struct RelationData {
    double mean_rating = .0;
    
    void setRelationData(YType& Yin) {
        Y = Yin; 
        mean_rating = Y.sum() / Y.nonZeros();
        
    }

    YType Y;
};


struct SparseMF : public Factors, public RelationData<SparseMatrixD> {
      //-- c'tor
      SparseMF(int num_latent, int num_fac = 2)
          : Factors(num_latent, num_fac), iter(0) {}

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

struct DenseMF : public Factors, public RelationData<Eigen::MatrixXd> {
    //-- c'tor
    DenseMF(int num_latent, int num_fac = 2)
        : Factors(num_latent, num_fac) {}
};

#endif /* MODEL_H */
