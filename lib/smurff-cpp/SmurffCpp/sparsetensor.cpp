#include <memory>
#include <iostream>

#include <Eigen/Dense>

#include "sparsetensor.h"
#include "TensorUtilsOld.h"

#include <Priors/ILatentPrior.h>

#include <SmurffCpp/Utils/Error.h>

using namespace Eigen;

//from macau bpmfutils
inline void sparseFromIJV(Eigen::SparseMatrix<double> &X, int* rows, int* cols, double* values, int N) 
{
   typedef Eigen::Triplet<double> T;
   std::vector<T> tripletList;
   tripletList.reserve(N);

   for (int n = 0; n < N; n++) 
   {
     tripletList.push_back(T(rows[n], cols[n], values[n]));
   }

   X.setFromTriplets(tripletList.begin(), tripletList.end());
}
 
 //from macau bpmfutils
inline void sparseFromIJV(Eigen::SparseMatrix<double> &X, Eigen::MatrixXi &idx, Eigen::VectorXd &values) 
{
   if (idx.rows() != values.size()) 
   {
      THROWERROR("sparseFromIJV: idx.rows() must equal values.size().");
   }

   if (idx.cols() != 2) 
   {
      THROWERROR("sparseFromIJV: idx.cols() must be equal to 2.");
   }

   typedef Eigen::Triplet<double> T;
   std::vector<T> tripletList;
   int N = values.size();
   tripletList.reserve(N);

   for (int n = 0; n < N; n++) 
   {
     tripletList.push_back(T(idx(n, 0), idx(n, 1), values(n)));
   }
   
   X.setFromTriplets(tripletList.begin(), tripletList.end());
}

Eigen::MatrixXd MatrixData::getTestData() 
{
  MatrixXd coords( getTestNonzeros(), 3);
  #pragma omp parallel for schedule(dynamic, 2)
  for (int k = 0; k < Ytest.outerSize(); ++k) 
  {
    int idx = Ytest.outerIndexPtr()[k];
    for (SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) 
    {
      coords(idx, 0) = it.row();
      coords(idx, 1) = it.col();
      coords(idx, 2) = it.value();
      idx++;
    }
  }
  return coords;
}

void MatrixData::setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) 
{
	Y.resize(nrows, ncols);
	sparseFromIJV(Y, rows, cols, values, nnz);
	Yt = Y.transpose();
	mean_value = Y.sum() / Y.nonZeros();
	dims.resize(2);
	dims << Y.rows(), Y.cols();
}

void MatrixData::setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) 
{
	Ytest.resize(nrows, ncols);
	sparseFromIJV(Ytest, rows, cols, values, nnz);
}

void MatrixData::setTrain(int* columns, int nmodes, double* values, int nnz, int* d) 
{
  if (nmodes != 2) 
  {
   THROWERROR("MatrixData: tensor training input not supported.");
  }

  auto idx  = toMatrix(columns, nnz, nmodes);
  auto vals = toVector(values, nnz);

  Y.resize(d[0], d[1]);
  sparseFromIJV(Y, idx, vals);

  Yt = Y.transpose();
  mean_value = Y.sum() / Y.nonZeros();
  dims.resize(2);
  dims << Y.rows(), Y.cols();
}
    
void MatrixData::setTest(int* columns, int nmodes, double* values, int nnz, int* d) 
{
  if (nmodes != 2) 
  {
   THROWERROR("MatrixData: tensor training input not supported.");
  }
  auto idx  = toMatrix(columns, nnz, nmodes);
  auto vals = toVector(values, nnz);

  Ytest.resize(d[0], d[1]);
  sparseFromIJV(Ytest, idx, vals);
}

