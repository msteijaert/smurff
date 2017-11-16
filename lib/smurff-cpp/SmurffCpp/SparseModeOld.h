#pragma once

#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

//this is a tensor slice where one dimention is fixed (this is not a matrix)
class SparseModeOld
{
public:
   int num_modes; // number of modes
   Eigen::VectorXi row_ptr; // vector of offsets (in values, in indices) to each dimention
   Eigen::MatrixXi indices; // [nnz x (nmodes - 1)] matrix of coordinates
   Eigen::VectorXd values; // vector of values
   int nnz;
   int mode;

public:
   SparseModeOld() 
      :  num_modes(0), nnz(0), mode(0) 
   {
      
   }

   // idx - [nnz x nmodes] matrix of coordinates
   // vals - vector of values
   // mode - index of dimention to fix
   // mode_size - size of dimention to fix
   SparseModeOld(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size) 
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