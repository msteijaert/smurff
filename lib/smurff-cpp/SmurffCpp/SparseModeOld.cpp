#include "SparseModeOld.h"

using namespace Eigen;

// idx - [nnz x nmodes] matrix of coordinates
// vals - vector of values
// m - index of dimention to fix
// mode_size - size of dimention to fix
void SparseModeOld::init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int m, int mode_size) 
{
  if (idx.rows() != vals.size()) 
  {
    throw std::runtime_error("idx.rows() must equal vals.size()");
  }

  num_modes = idx.cols();
  row_ptr.resize(mode_size + 1); //mode_size + 1 because this vector will hold commulative sum
  row_ptr.setZero();
  values.resize(vals.size());
  indices.resize(idx.rows(), idx.cols() - 1);

  mode = m;
  auto rows = idx.col(mode);
  const int nrow = mode_size;
  nnz  = idx.rows();

  // compute number of non-zero entries per each element for the mode
  // (compute number of non-zero elements for each coordinate in specific dimention)
  for (int i = 0; i < nnz; i++) 
  {
    if (rows(i) >= mode_size) 
    {
      throw std::runtime_error("SparseMode: mode value larger than mode_size");
    }

    row_ptr(rows(i))++;
  }

  // cumsum counts
  // (compute commulative sum of nnz for each coordinate in specific dimention)
  for (int row = 0, cumsum = 0; row < nrow; row++) 
  {
    int temp     = row_ptr(row);
    row_ptr(row) = cumsum;
    cumsum      += temp;
  }
  row_ptr(nrow) = nnz; //last element should be equal to nnz

  // transform index matrix into index matrix with one fixed dimention ?

  // writing idx and vals to indices and values
  for (int i = 0; i < nnz; i++) 
  {
    int row  = rows(i); // coordinate in fixed dimention
    int dest = row_ptr(row); // commulative nnz for the coordinate

    for (int j = 0, nj = 0; j < idx.cols(); j++) 
    {
      if (j == mode) //skip fixed dimention
         continue;

      indices(dest, nj) = idx(i, j);
      nj++;
    }

    //A->cols[dest] = cols[i];
    values(dest) = vals(i);
    row_ptr(row)++; //update commulative nnz vector
  }

  // restore commulative nnz vector ?

  // fixing row_ptr
  for (int row = 0, prev = 0; row <= nrow; row++) 
  {
    int temp     = row_ptr(row);
    row_ptr(row) = prev;
    prev         = temp;
  }
}