#include "ScarceBinaryMatrixData.h"

#include <SmurffCpp/Utils/TruncNorm.h>

#include <SmurffCpp/VMatrixExprIterator.hpp>
#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

using namespace smurff;

ScarceBinaryMatrixData::ScarceBinaryMatrixData(Eigen::SparseMatrix<double>& Y)
   : ScarceMatrixData(Y)
{
   name = "ScarceBinaryMatrixData [containing 0,1,NA]";
}

void ScarceBinaryMatrixData::get_pnm(const SubModel& model, uint32_t mode, int n, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
   auto Vf = *model.CVbegin(mode);
   // todo : check noise == probit noise
   
   for (Eigen::SparseMatrix<double>::InnerIterator it(Y(mode), n); it; ++it)
   {
      // FIXME!
      int pos0 = it.row();
      int pos1 = it.col();
      if (mode == 0) std::swap(pos0, pos1);

      const auto &col = Vf.col(it.row());
      MM.noalias() += col * col.transpose();
      rr.noalias() += col * it.value() * noise()->getAlpha(model.predict({pos0, pos1}), it.value());
   }
}

void ScarceBinaryMatrixData::update_pnm(const SubModel& model, uint32_t mode)
{
   //can not cache VV because of scarceness
}

std::uint64_t ScarceBinaryMatrixData::nna() const
{
   return size() - nnz(); //nrows * ncols - nnz
}
