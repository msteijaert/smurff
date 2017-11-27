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



//////   Tensor data   //////

//this is a tensor slice where one dimention is fixed (this is not a matrix)
class SparseMode 
{
public:
   int num_modes; // number of modes
   Eigen::VectorXi row_ptr; // vector of offsets (in values) to each dimention
   Eigen::MatrixXi indices; // [nnz x nmodes - 1] matrix of coordinates
   Eigen::VectorXd values; // vector of values
   int nnz;
   int mode;

public:
   SparseMode() 
      :  num_modes(0), nnz(0), mode(0) 
   {
      
   }

   // idx - [nnz x nmodes] matrix of coordinates
   // vals - vector of values
   // mode - dimention index
   // mode_size - dimention size
   SparseMode(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size) 
   {
      init(idx, vals, mode, mode_size); 
   }

   void init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size);

   int nonZeros() 
   { 
      return nnz; 
   }

   int modeSize() 
   { 
      return row_ptr.size() - 1;
   }
};

class TensorData : public IData 
{
  public:
    Eigen::MatrixXi dims; // train data dimention sizes
    double mean_value = .0; //mean value of values vector
    std::vector< std::unique_ptr<SparseMode> >* Y; // this is a vector of dimention slices
    SparseMode Ytest; // slice of first dimention
    int N; //number of dimentions

  public:
    TensorData(int Nmodes) 
      : N(Nmodes) 
    {
        Y = new std::vector< std::unique_ptr<SparseMode> >(); 
    }

    // convert to vector of sparse views
    void setTrain(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d);

    // sparse tensor constructor
    void setTrain(int* columns, int nmodes, double* values, int nnz, int* dims) override;

    // sparse matrix constructor
    void setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override;

    //---

    // convert to sparse view (only take first dimention)
    void setTest(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d);

    // sparse tensor constructor
    void setTest(int* columns, int nmodes, double* values, int nnz, int* dims) override;

    // sparse matrix constructor
    void setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override;

    // get train data dimention sizes
    Eigen::VectorXi getDims() override 
    { 
       return dims;
    };
    
   //get test data nnz
   int getTestNonzeros() override 
   {
      return Ytest.nonZeros(); 
   };
   
   //transforms sparse view into matrix with 3 components K | I | V
   Eigen::MatrixXd getTestData() override;

   //mean value of values vector
   double getMeanValue() override 
   { 
      return mean_value; 
   };
};

//convert array of coordinates to [nnz x nmodes] matrix
Eigen::MatrixXi toMatrix(int* columns, int nrows, int ncols);

//convert two coordinate arrays to [N x 2] matrix
Eigen::MatrixXi toMatrix(int* col1, int* col2, int nrows);

//convert array of values to vector
Eigen::VectorXd toVector(double* vals, int size);

//convert array of values to vector
Eigen::VectorXi toVector(int* vals, int size);
