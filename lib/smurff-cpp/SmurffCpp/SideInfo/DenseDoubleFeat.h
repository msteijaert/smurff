#pragma once

#include "LibFastSparseDependency.h"

#include <Eigen/Dense>

#include "ISideInfo.h"

#include <memory>

namespace smurff {

   class DenseDoubleFeatSideInfo : public ISideInfo
   {
   private:
      std::shared_ptr<Eigen::MatrixXd> m_side_info;

   public:
      DenseDoubleFeatSideInfo(std::shared_ptr<Eigen::MatrixXd> side_info)
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
         os << "DenseDouble [" << m_side_info->rows() << ", " << m_side_info->cols() << "]" << std::endl;
         return os;
      }

      bool is_dense() const
      {
         return true;
      }
   };

}