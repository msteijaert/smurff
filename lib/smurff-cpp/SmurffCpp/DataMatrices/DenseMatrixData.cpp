#include "DenseMatrixData.h"

using namespace smurff;

DenseMatrixData::DenseMatrixData(Eigen::MatrixXd Y)
   : FullMatrixData<Eigen::MatrixXd>(Y)
{
    this->name = "DenseMatrixData [fully known]";
}

double DenseMatrixData::train_rmse(const SubModel& model) const
{
   return sqrt(sumsq(model) / this->size());
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
