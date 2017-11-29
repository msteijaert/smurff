#include "MacauPrior.hpp"

namespace smurff
{
   // specialization for dense matrices --> always direct method
   template<>
   void MacauPrior<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::sample_beta_cg()
   {
      not_implemented("Dense Matrix requires direct method");
   }
}