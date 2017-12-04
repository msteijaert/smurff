#include "DenseMatrixData.h"

using namespace smurff;

DenseMatrixData::DenseMatrixData(Eigen::MatrixXd Y)
   : FullMatrixData<Eigen::MatrixXd>(Y)
{
    this->name = "DenseMatrixData [fully known]";
}

double DenseMatrixData::train_rmse(const SubModel& model) const
{
   double se = 0.;
   #pragma omp parallel for schedule(guided) reduction(+:se)
   for(int c = 0; c < Y().cols(); ++c) 
   {
      for(int m = 0; m < Y().rows(); ++m) 
      {
         se += square(Y()(m,c) - model.predict({m,c}));
      }
   }
   return sqrt(se / this->size());
}

double DenseMatrixData::var_total() const
{
   double cwise_mean = this->sum() / this->size();
   double se = (Y().array() - cwise_mean).square().sum();
   
   double var = se / this->size();
   if (var <= 0.0 || std::isnan(var))
   {
      // if var cannot be computed using 1.0
      var = 1.0;
   }

   return var;
}

// for the adaptive gaussian noise
double DenseMatrixData::sumsq(const SubModel& model) const
{
   double sumsq = 0.0;

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
   for (int j = 0; j < this->ncol(); j++) 
   {
      for (int i = 0; i < this->nrow(); i++) 
      {
         sumsq += square(model.predict({i,j}) - this->Y()(i,j));
      }
   }

   return sumsq;
}
