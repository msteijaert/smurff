#pragma once

#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// forward declarations
class ILatentPrior;
class ProbitNoise;
class AdaptiveGaussianNoise;
class FixedGaussianNoise;

class IData {
  public:
    virtual void setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) = 0;
    virtual void setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) = 0;
    virtual void setTrain(int* idx, int nmodes, double* values, int nnz, int* dims) = 0;
    virtual void setTest(int* idx, int nmodes, double* values, int nnz, int* dims) = 0;

    virtual Eigen::VectorXi getDims() = 0;
    virtual int getTestNonzeros() = 0;
    virtual Eigen::MatrixXd getTestData() = 0;
    virtual double getMeanValue() = 0;
    virtual ~IData() {};
};

//////   Matrix data    /////
class MatrixData : public IData 
{
  public:
   Eigen::SparseMatrix<double> Y; 
   Eigen::SparseMatrix<double> Yt;
   Eigen::SparseMatrix<double> Ytest;
   double mean_value = .0; 
   Eigen::VectorXi dims;
   
   MatrixData() {}

   Eigen::VectorXi getDims() override { return dims; }
   int getTestNonzeros() override { return Ytest.nonZeros(); }
   double getMeanValue() override { return mean_value; }

   void setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override;
   void setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override;

   void setTrain(int* idx, int nmodes, double* values, int nnz, int* dims) override;
   void setTest(int* idx, int nmodes, double* values, int nnz, int* dims) override;

   Eigen::MatrixXd getTestData() override;
};
