#include "ScarceBinaryMatrixData.h"
#include "truncnorm.h"

using namespace smurff;

ScarceBinaryMatrixData::ScarceBinaryMatrixData(Eigen::SparseMatrix<double>& Y)
   : ScarceMatrixData(Y)
{
      name = "ScarceBinaryMatrixData [containing 0,1,NA]";
}

void ScarceBinaryMatrixData::get_pnm(const SubModel& model, int mode, int n, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
    // todo : check noise == probit noise
    auto u = model.U(mode).col(n);
    for (Eigen::SparseMatrix<double>::InnerIterator it(getYcPtr()->at(mode), n); it; ++it)
    {
        const auto &col = model.V(mode).col(it.row());
        MM.noalias() += col * col.transpose();
		double y = 2 * it.value() - 1;
		auto z = y * rand_truncnorm(y * col.dot(u), 1.0, 0.0);
        rr.noalias() += col * z;
    }
}

void ScarceBinaryMatrixData::update_pnm(const SubModel& model,int mode)
{
}

int ScarceBinaryMatrixData::nna() const
{
   return size() - nnz();
}