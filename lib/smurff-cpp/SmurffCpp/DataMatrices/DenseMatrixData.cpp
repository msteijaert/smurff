#include "DenseMatrixData.h"

using namespace smurff;

DenseMatrixData::DenseMatrixData(Eigen::MatrixXf Y)
   : FullMatrixData<Eigen::MatrixXf>(Y)
{
    this->name = "DenseMatrixData [fully known]";
}

//d is an index of column in U matrix
void DenseMatrixData::getMuLambda(const SubModel& model, uint32_t mode, int d, Eigen::VectorXf& rr, Eigen::MatrixXf& MM) const
{
    auto &Y = this->Y(mode).col(d);
    auto Vf = *model.CVbegin(mode);
    auto &ns = noise();

    for(int r = 0; r<Y.rows(); ++r) 
    {
        const auto &col = Vf.col(r);
        PVec<> pos = this->pos(mode, d, r);
        float noisy_val = ns.sample(model, pos, Y(r));
        rr.noalias() += col * noisy_val; // rr = rr + (V[m] * noisy_y[d]) 
    }

    MM.noalias() += ns.getAlpha() * VV[mode]; // MM = MM + VV[m]
}

float DenseMatrixData::train_rmse(const SubModel& model) const
{
   return std::sqrt(sumsq(model) / this->size());
}

float DenseMatrixData::var_total() const
{
   float cwise_mean = this->sum() / this->size();
   float se = (Y().array() - cwise_mean).square().sum();
   
   float var = se / this->size();
   if (var <= 0.0 || std::isnan(var))
   {
      // if var cannot be computed using 1.0
      var = 1.0;
   }

   return var;
}

// for the adaptive gaussian noise
float DenseMatrixData::sumsq(const SubModel& model) const
{
   float sumsq = 0.0;

   #pragma omp parallel for schedule(guided) reduction(+:sumsq)
   for (int j = 0; j < this->ncol(); j++) 
   {
      for (int i = 0; i < this->nrow(); i++) 
      {
         sumsq += std::pow(model.predict({i,j}) - this->Y()(i,j), 2);
      }
   }

   return sumsq;
}
