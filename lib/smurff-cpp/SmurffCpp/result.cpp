#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/ConstVMatrixIterator.hpp>
#include <SmurffCpp/Utils/utils.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>

using namespace std;
using namespace Eigen;

namespace smurff {

//Y - test sparse matrix
void Result::set(std::shared_ptr<TensorConfig> Y)
{
   if(Y->isDense())
      throw std::runtime_error("test data should be sparse");

   if(Y->getNModes() != 2)
      throw "Invalid number of dimensions. Tensor can not be converted to matrix.";

   std::shared_ptr<std::vector<std::uint32_t> > columnsPtr = Y->getColumnsPtr();
   std::shared_ptr<std::vector<double> > valuesPtr = Y->getValuesPtr();

   for(std::uint64_t i = 0; i < Y->getNNZ(); i++)
   {
      double val = Y->isBinary() ? 1.0 : valuesPtr->operator[](i);
      std::uint32_t row = columnsPtr->operator[](i);
      std::uint32_t col = columnsPtr->operator[](i + Y->getNNZ());
      
      predictions.push_back({row, col, val});
   }

   m_nrows = Y->getDims()[0];
   m_ncols = Y->getDims()[1];

   init();
}

void Result::init()
{
   total_pos = 0;
   if (classify)
   {
      for(auto &t : predictions)
      {
         int is_positive = t.val > threshold;
         total_pos += is_positive;
      }
   }
}

double Result::rmse_using_globalmean(double mean)
{
   double se = 0.;
   for(auto t : predictions)
      se += square(t.val - mean);
   return sqrt( se / predictions.size() );
}

double Result::rmse_using_modemean(std::shared_ptr<Data> data, int mode)
{
   const unsigned N = predictions.size();
   double se = 0.;
   for(auto t : predictions)
   {
      int n = mode == 0 ? t.row : t.col;
      double pred = data->getModeMeanItem(mode, n);
      se += square(t.val - pred);
   }
   return sqrt( se / N );
}

//--- output model to files
void Result::save(std::string prefix)
{
   if (predictions.empty())
      return;

   std::string fname_pred = prefix + "-predictions.csv";
   std::ofstream predfile;
   predfile.open(fname_pred);
   predfile << "row,col,y,pred_1samp,pred_avg,var,std\n";

   for ( auto &t : predictions)
   {
      predfile
             << to_string( t.row  )
      << "," << to_string( t.col  )
      << "," << to_string( t.val  )
      << "," << to_string( t.pred_1sample )
      << "," << to_string( t.pred_avg )
      << "," << to_string( t.var )
      << "," << to_string( t.stds )
      << "\n";
   }

   predfile.close();
}

///--- update RMSE and AUC

//model - holds samples (U matrices)
//data - Y train matrix
void Result::update(std::shared_ptr<const Model> model, std::shared_ptr<Data> data, bool burnin)
{
   if (predictions.size() == 0)
      return;

   const unsigned N = predictions.size();

   if (burnin)
   {
      double se_1sample = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample)
      for(unsigned k = 0; k < predictions.size(); ++k)
      {
         auto &t = predictions[k];
         t.pred_1sample = model->predict({(int)t.row, (int)t.col}, data); //dot product of i'th columns in each U matrix
         se_1sample += square(t.val - t.pred_1sample);
      }

      burnin_iter++;
      rmse_1sample = sqrt(se_1sample / N);

      if (classify)
      {
         auc_1sample = calc_auc(predictions, threshold,
               [](const Item &a, const Item &b) { return a.pred_1sample < b.pred_1sample;});
      }
   }
   else
   {
      double se_1sample = 0.0, se_avg = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample, se_avg)
      for(size_t k = 0; k < predictions.size(); ++k)
      {
         auto &t = predictions[k];
         const double pred = model->predict({(int)t.row, (int)t.col}, data); //dot product of i'th columns in each U matrix
         
         se_1sample += square(t.val - pred);
         double delta = pred - t.pred_avg;
         double pred_avg = (t.pred_avg + delta / (sample_iter + 1));
         t.var += delta * (pred - pred_avg);
         const double inorm = 1.0 / sample_iter;
         t.stds = sqrt(t.var * inorm);
         t.pred_avg = pred_avg;
         t.pred_1sample = pred;
         se_avg += square(t.val - pred_avg);
      }

      sample_iter++;
      rmse_1sample = sqrt(se_1sample / N);
      rmse_avg = sqrt(se_avg / N);

      if (classify)
      {
         auc_1sample = calc_auc(predictions, threshold,
               [](const Item &a, const Item &b) { return a.pred_1sample < b.pred_1sample;});

         auc_avg = calc_auc(predictions, threshold,
               [](const Item &a, const Item &b) { return a.pred_avg < b.pred_avg;});
      }
   }
}

//macau ProbitNoise eval
/*
eval_rmse(MatrixData & data, 
          const int n, 
          Eigen::VectorXd & predictions, 
          Eigen::VectorXd & predictions_var,
          std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
{
 const unsigned N = data.Ytest.nonZeros();
  Eigen::VectorXd pred(N);
  Eigen::VectorXd test(N);
  Eigen::MatrixXd & rows = *samples[0];
  Eigen::MatrixXd & cols = *samples[1];

// #pragma omp parallel for schedule(dynamic,8) reduction(+:se, se_avg) <- dark magic :)
  for (int k = 0; k < data.Ytest.outerSize(); ++k) {
    int idx = data.Ytest.outerIndexPtr()[k];
    for (Eigen::SparseMatrix<double>::InnerIterator it(data.Ytest,k); it; ++it) {
     pred[idx] = nCDF(cols.col(it.col()).dot(rows.col(it.row())));
     test[idx] = it.value();

      // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
      double pred_avg;
      if (n == 0) {
        pred_avg = pred[idx];
      } else {
        double delta = pred[idx] - predictions[idx];
        pred_avg = (predictions[idx] + delta / (n + 1));
        predictions_var[idx] += delta * (pred[idx] - pred_avg);
      }
      predictions[idx++] = pred_avg;

   }
  }
  auc_test_onesample = auc(pred,test);
  auc_test = auc(predictions, test);
}
*/

//macau tensor eval

//sview - sparse view that is constructed from YTest by fixing mode 0 and taking nrows as mode_size
//Nepoch - similar to (iter < config.burnin)

typedef Eigen::Matrix<std::uint32_t, Eigen::Dynamic, 1 > VectorXui32;

std::pair<double,double> eval_rmse_tensor(std::shared_ptr<const Model> model,
                                          std::shared_ptr<SparseMode> sview,
                                          const int Nepoch,
                                          Eigen::VectorXd & predictions,
                                          Eigen::VectorXd & predictions_var,
                                          double mean_value
                                          )
{
   // No test data, returning NaN's
   if (sview->getNNZ() == 0) 
      return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

   if (sview->getNNZ() != predictions.size()) 
      throw std::runtime_error("Ytest.size() and predictions.size() must be equal.");

   if (sview->getNPlanes() != model->U(0)->cols()) 
      throw std::runtime_error("U.cols() and sview size must be equal.");

   double se = 0.0;
   double se_avg = 0.0;

   const unsigned NNZ = sview->getNNZ();

   std::shared_ptr<const Eigen::MatrixXd> U0 = model->U(0);
   
   #pragma omp parallel for schedule(dynamic, 2) reduction(+:se, se_avg)
   for(int n = 0; n < U0->cols(); n++)
   {
      const Eigen::VectorXd& u0_col = U0->col(n); //get column from U

      for (std::uint64_t j = sview->beginPlane(n); j < sview->endPlane(n); j++) //go through hyperplane in tensor rotation
      {
         double pred = 0;

         for (int nl = 0; nl < U0->rows(); nl++) // go through each latent vector
         {
            double tmp = u0_col(nl); //value from n'th column of U0

            ConstVMatrixIterator<Eigen::MatrixXd> V = model->CVbegin(0); //get V matrices for mode      
            for (std::uint64_t m = 0; m < sview->getNCoords(); m++, ++V) //go through each coordinate of value
            {
               const Eigen::VectorXd& vm_col = V->col(sview->getIndices()(j, m)); //get m'th column from V
               tmp *= vm_col(nl);
            }

            pred += tmp;
         }

         double pred_avg = 0;
         if (Nepoch == 0) 
         {
            pred_avg = pred;
         } 
         else 
         {
            double delta = pred - predictions(j);
            pred_avg = (predictions(j) + delta / (Nepoch + 1));
            predictions_var(j) += delta * (pred - pred_avg);
         }

         se     += square(sview->getValues()[j] - pred);
         se_avg += square(sview->getValues()[j] - pred_avg);
         predictions(j) = pred_avg;
      }
   }

   const double rmse = sqrt(se / NNZ);
   const double rmse_avg = sqrt(se_avg / NNZ);
   return std::make_pair(rmse, rmse_avg);
}

std::ostream &Result::info(std::ostream &os, std::string indent, std::shared_ptr<Data> data)
{
   if (predictions.size())
   {
      double test_fill_rate = 100. * predictions.size() / m_nrows / m_ncols;
      os << indent << "Test data: " << predictions.size() << " [" << m_nrows << " x " << m_ncols << "] (" << test_fill_rate << "%)\n";
      os << indent << "RMSE using globalmean: " << rmse_using_globalmean(data->getGlobalMean()) << endl;
      os << indent << "RMSE using rowmean: " << rmse_using_modemean(data,0) << endl;
      os << indent << "RMSE using colmean: " << rmse_using_modemean(data,1) << endl;
   }
   else
   {
    os << indent << "Test data: -\n";
   }

   if (classify)
   {
      double pos = 100. * (double)total_pos / (double)predictions.size();
      os << indent << "Binary classification threshold: " << threshold << "\n";
      os << indent << "  " << pos << "% positives in test data\n";
   }

   return os;
}

} // end namespace
