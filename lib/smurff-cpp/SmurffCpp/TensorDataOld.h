#pragma once

#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "sparsetensor.h"
#include "SparseModeOld.h"

//////   Tensor data   //////

class TensorDataOld : public IData 
{
  public:
    Eigen::MatrixXi dims; // train data dimention sizes
    double mean_value = .0; //mean value of values vector
    std::vector< std::unique_ptr<SparseModeOld> >* Y; // this is a vector of dimention slices
    SparseModeOld Ytest; // slice of first dimention
    int N; //number of dimentions

  public:
    TensorDataOld(int Nmodes) 
      : N(Nmodes) 
    {
        Y = new std::vector< std::unique_ptr<SparseModeOld> >(); 
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