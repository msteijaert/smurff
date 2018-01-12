#include "SparseMatrixData.h"

using namespace smurff;
using namespace Eigen;

SparseMatrixData::SparseMatrixData(SparseMatrix<double> Y)
   : FullMatrixData<SparseMatrix<double>>(Y)
{
   this->name = "SparseMatrixData [fully known]";
}

void SparseMatrixData::getMuLambda(const SubModel& model, uint32_t mode, int d, VectorXd& rr, MatrixXd& MM) const
{
    const auto& Y = this->Y(mode);
    auto Vf = *model.CVbegin(mode);
    auto &ns = *noise();

    for (SparseMatrix<double>::InnerIterator it(Y, d); it; ++it) 
    {
        const auto &col = Vf.col(it.row());
        auto p = pos(mode, d, it.row());
        double noisy_val = ns.sample(model, p, it.value());
        rr.noalias() += col * noisy_val; // rr = rr + (V[m] * y[d]) * alpha
    }

    MM.noalias() += ns.getAlpha() * VV[mode]; // MM = MM + VV[m]
}

double SparseMatrixData::train_rmse(const SubModel& model) const
{
   return std::sqrt(sumsq(model) / this->size());
}

double SparseMatrixData::var_total() const
{
   double cwise_mean = this->sum() / this->size();
   double se = 0.0;

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
   for(int c = 0; c < Y().cols(); ++c)
   {
      int r = 0;
      for (SparseMatrix<double>::InnerIterator it(Y(), c); it; ++it)
      {
         for(; r < it.row(); r++) //handle implicit zeroes
            se += std::pow(cwise_mean, 2);

         se += std::pow(it.value() - cwise_mean, 2);
         r++;
      }

      for(; r < Y().rows(); r++) //handle implicit zeroes
         se += std::pow(cwise_mean, 2);
   }

   double var = se / this->size();
   if (var <= 0.0 || std::isnan(var))
   {
      // if var cannot be computed using 1.0
      var = 1.0;
   }

   return var;
}

double SparseMatrixData::sumsq(const SubModel& model) const
{
   double sumsq = 0.0;
   
   #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
   for(int c = 0; c < Y().cols(); ++c)
   {
      int r = 0;
      for (SparseMatrix<double>::InnerIterator it(Y(), c); it; ++it)
      {
         for(; r < it.row(); r++) //handle implicit zeroes
            sumsq += std::pow(model.predict({r, c}), 2);

         sumsq += std::pow(model.predict({r, c}) - it.value(), 2);
         r++;
      }

      for(; r < Y().rows(); r++) //handle implicit zeroes
         sumsq += std::pow(model.predict({r, c}), 2);
   }

   return sumsq;
}
