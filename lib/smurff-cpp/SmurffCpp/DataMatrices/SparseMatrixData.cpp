#include "SparseMatrixData.h"

using namespace smurff;

SparseMatrixData::SparseMatrixData(Eigen::SparseMatrix<float> Y)
   : FullMatrixData<Eigen::SparseMatrix<float>>(Y)
{
   this->name = "SparseMatrixData [fully known]";
}

void SparseMatrixData::getMuLambda(const SubModel& model, uint32_t mode, int d, Eigen::VectorXf& rr, Eigen::MatrixXf& MM) const
{
    const auto& Y = this->Y(mode);
    auto Vf = *model.CVbegin(mode);
    auto &ns = noise();

    for (Eigen::SparseMatrix<float>::InnerIterator it(Y, d); it; ++it) 
    {
        const auto &col = Vf.col(it.row());
        auto p = pos(mode, d, it.row());
        float noisy_val = ns.sample(model, p, it.value());
        rr.noalias() += col * noisy_val; // rr = rr + (V[m] * y[d]) * alpha
    }

    MM.noalias() += ns.getAlpha() * VV[mode]; // MM = MM + VV[m]
}

float SparseMatrixData::train_rmse(const SubModel& model) const
{
   return std::sqrt(sumsq(model) / this->size());
}

float SparseMatrixData::var_total() const
{
   const float cwise_mean = this->sum() / this->size();
   const float cwise_mean_squared = std::pow(cwise_mean, 2);
   float se = 0.0;

   #pragma omp parallel for schedule(guided) reduction(+:se)
   for(int c = 0; c < Y().cols(); ++c)
   {
      int r = 0;
      for (Eigen::SparseMatrix<float>::InnerIterator it(Y(), c); it; ++it)
      {
         se += (r - it.row()) * cwise_mean_squared; // handle implicit zeroes
         se += std::pow(it.value() - cwise_mean, 2);
         r = it.row() + 1;
      }

      se += (r - Y().rows()) * cwise_mean_squared; // handle implicit zeroes
   }

   float var = se / this->size();
   if (var <= 0.0 || std::isnan(var))
   {
      // if var cannot be computed using 1.0
      var = 1.0;
   }

   return var;
}

float SparseMatrixData::sumsq(const SubModel& model) const
{
   float sumsq = 0.0;
   
   #pragma omp parallel for schedule(guided) reduction(+:sumsq)
   for(int c = 0; c < Y().cols(); ++c)
   {
      int r = 0;
      for (Eigen::SparseMatrix<float>::InnerIterator it(Y(), c); it; ++it)
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
