#pragma once

#include "LibFastSparseDependency.h"

#include "ISideInfo.h"

#include <memory>

namespace smurff {

class SparseFeat
{
public:
   BinaryCSR M;
   BinaryCSR Mt;

   SparseFeat() {}

   SparseFeat(int nrow, int ncol, long nnz, int* rows, int* cols)
   {
      new_bcsr(&M, nnz, nrow, ncol, rows, cols);
      new_bcsr(&Mt, nnz, ncol, nrow, cols, rows);
   }

   virtual ~SparseFeat()
   {
      free_bcsr(&M);
      free_bcsr(&Mt);
   }

   int nfeat() const
   {
      return M.ncol;
   }

   int cols() const
   {
      return M.ncol;
   }

   int nsamples() const
   {
      return M.nrow;
   }

   int rows() const
   {
      return M.nrow;
   }
};

class SparseFeatSideInfo : public ISideInfo
{
private:
   std::shared_ptr<SparseFeat> m_side_info;

public:
   SparseFeatSideInfo(std::shared_ptr<SparseFeat> side_info)
      : m_side_info(side_info)
   {
   }

public:
   int cols() const override
   {
      return m_side_info->cols();
   }

   int rows() const override
   {
      return m_side_info->rows();
   }

public:
   std::ostream& print(std::ostream &os) const override
   {
      os << "SparseBinary [" << m_side_info->rows() << ", " << m_side_info->cols() << "]" << std::endl;
      return os;
   }

   bool is_dense() const
   {
      return false;
   }
};
}