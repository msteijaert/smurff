#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

struct MFactor {
      //-- c'tor
      MFactor(int D) : num_latent(D) {}
      void init();

      int num_latent;
      double mean_rating = .0; 

      Eigen::SparseMatrix<double> Y;
      Eigen::MatrixXd U;

      Eigen::MatrixXd::ConstColXpr col(int i) const { return U.col(i); }
};

struct MFactors {
      //-- c'tor
      MFactors(int num_latent, int num_fac = 2)
          : iter(0)
      {
          assert(num_fac == 2); 
          for(int i=0; i<num_fac; ++i) factors.push_back(MFactor(num_latent));
      }

      Eigen::SparseMatrix<double> Ytest;
      Eigen::VectorXd predictions, predictions_var;

      double iter;
      double rmse, rmse_avg;
      double auc, auc_avg;

      int num_latent() const { return factors.at(0).num_latent; }
      double mean_rating() const { return factors.at(0).mean_rating; }
      Eigen::SparseMatrix<double> &Y() { return factors.at(0).Y; }
      Eigen::SparseMatrix<double> &Yt() { return factors.at(1).Y; }
      Eigen::MatrixXd::ConstColXpr col(int f, int i) const { return factors.at(f).col(i); }
      Eigen::MatrixXd &U(int f) { return factors.at(f).U; }

      std::vector<MFactor> factors;

      void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols);
      void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols);

      void update_rmse(bool burnin);
      void update_auc(bool burnin);

      Eigen::VectorXd getStds(int iter);
      Eigen::MatrixXd getTestData();

      void init();
};

#endif /* MODEL_H */
