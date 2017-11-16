#include "TensorDataOld.h"
#include "TensorUtilsOld.h"

using namespace Eigen;

////////  TensorData  ///////

//idx - [nnz x nmodes] matrix of coordinates
//vals - vector of values
//d - vector of dimentions
void TensorDataOld::setTrain(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d) 
{
  if (idx.rows() != vals.size()) 
  {
    throw std::runtime_error("setTrain(): idx.rows() must equal vals.size()");
  }

  dims = d;
  mean_value = vals.mean();

  for (int mode = 0; mode < N; mode++) 
  {
    Y->push_back( std::unique_ptr<SparseModeOld>(new SparseModeOld(idx, vals, mode, dims(mode))) );
  }
}

// sparse tensor constructor
// size(dims) == nmodes
// size(values) == nnz
// size(columns) = nnz * nmodes
void TensorDataOld::setTrain(int* columns, int nmodes, double* values, int nnz, int* dims) 
{
  auto idx  = toMatrix(columns, nnz, nmodes); //combine coordinates into [nnz x nmodes] matrix
  auto vals = toVector(values, nnz); // convert values to vector
  auto d    = toVector(dims, nmodes); // dimentions to vector
  setTrain(idx, vals, d);
}

// sparse matrix constructor 
// size(rows) == size(cols) == size(values) == nnz
// nrows, ncols - dimentions
void TensorDataOld::setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) 
{
   auto idx  = toMatrix(rows, cols, nnz); // combine coordinates into [nnz x 2] matrix
   auto vals = toVector(values, nnz); // convert values to vector
 
   VectorXi d(2);
   d << nrows, ncols; // create vector of dimentions
 
   setTrain(idx, vals, d);
}

//====
  
//idx - [nnz x nmodes] matrix of coordinates
//vals - vector of values
//d - vector of dimentions
void TensorDataOld::setTest(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d) 
{
  if (idx.rows() != vals.size()) 
  {
    throw std::runtime_error("setTest(): idx.rows() must equal vals.size()");
  }

  if ((d - dims).norm() != 0) 
  {
    throw std::runtime_error("setTest(): train and test Tensor sizes are not equal.");
  }

  Ytest = SparseModeOld(idx, vals, 0, d(0));
}

// sparse tensor constructor
// size(dims) == nmodes
// size(values) == nnz
// size(columns) = nnz * nmodes
// columns - vector of all coordinates
// values - vector of values
void TensorDataOld::setTest(int* columns, int nmodes, double* values, int nnz, int* dims) 
{
   auto idx  = toMatrix(columns, nnz, nmodes); //combine coordinates into [nnz x nmodes] matrix
   auto vals = toVector(values, nnz); // convert values to vector
   auto d    = toVector(dims, nmodes); // dimentions to vector
   setTest(idx, vals, d);
}

// sparse matrix constructor
// size(rows) == size(cols) == size(values) == nnz
// nrows, ncols - dimentions
// rows - row coordinates
// cols - col coordinates
// values - vector of values
void TensorDataOld::setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) 
{
  auto idx  = toMatrix(rows, cols, nnz); // combine coordinates into [nnz x 2] matrix
  auto vals = toVector(values, nnz); // convert values to vector

  VectorXi d(2);
  d << nrows, ncols; // create vector of dimentions

  setTest(idx, vals, d);
}

//transforms sparse view into matrix with 3 components K | I | V
// K - vector of dimention indexes (from Ytest.modeSize())
// I - matrix of coordinates (Ytest.indices)
// V - vector of values (Ytest.values)
Eigen::MatrixXd TensorDataOld::getTestData() 
{
  MatrixXd coords( getTestNonzeros(), N + 1);
  #pragma omp parallel for schedule(dynamic, 2)
  for (int k = 0; k < Ytest.modeSize(); ++k) // iterate through each dimention
  {
    for (int idx = Ytest.row_ptr[k]; idx < Ytest.row_ptr[k+1]; idx++)  //iterate through pairs of dimention sizes
    {
      coords(idx, 0) = (double)k;
      for (int col = 0; col < Ytest.indices.cols(); col++)  //iterate through each dimention coordinates column
      {
        coords(idx, col + 1) = (double)Ytest.indices(idx, col);
      }
      coords(idx, N) = Ytest.values(idx);
    }
  }
  return coords;
}