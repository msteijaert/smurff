#include "TensorUtilsOld.h"

using namespace Eigen;

// util functions

//convert two coordinate arrays to [N x 2] matrix
Eigen::MatrixXi toMatrix(int* col1, int* col2, int nrows) {
   Eigen::MatrixXi idx(nrows, 2);
   for (int row = 0; row < nrows; row++) {
     idx(row, 0) = col1[row];
     idx(row, 1) = col2[row];
   }
   return idx;
 }
 
 //convert array of coordinates to [nnz x nmodes] matrix
 Eigen::MatrixXi toMatrix(int* columns, int nrows, int ncols) {
   Eigen::MatrixXi idx(nrows, ncols);
   for (int row = 0; row < nrows; row++) {
     for (int col = 0; col < ncols; col++) {
       idx(row, col) = columns[col * nrows + row];
     }
   }
   return idx;
 }
 
 //convert array of values to vector
 Eigen::VectorXd toVector(double* vals, int size) {
   Eigen::VectorXd v(size);
   for (int i = 0; i < size; i++) {
     v(i) = vals[i];
   }
   return v;
 }
 
 //convert array of values to vector
 Eigen::VectorXi toVector(int* vals, int size) {
   Eigen::VectorXi v(size);
   for (int i = 0; i < size; i++) {
     v(i) = vals[i];
   }
   return v;
 }
 