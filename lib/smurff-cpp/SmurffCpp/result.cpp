#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
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

#include <SmurffCpp/Utils/Error.h>

using namespace std;
using namespace Eigen;

namespace smurff {

//Y - test sparse matrix
void Result::set(std::shared_ptr<TensorConfig> Y)
{
   if(Y->isDense())
      THROWERROR("test data should be sparse");

   std::shared_ptr<std::vector<std::uint32_t> > columnsPtr = Y->getColumnsPtr();
   std::shared_ptr<std::vector<double> > valuesPtr = Y->getValuesPtr();

   std::vector<int> coords(Y->getNModes());
   for(std::uint64_t i = 0; i < Y->getNNZ(); i++)
   {
      for(std::uint64_t m = 0; m < Y->getNModes(); m++)
         coords[m] = static_cast<int>(columnsPtr->operator[](Y->getNNZ() * m + i));

      m_predictions.push_back({smurff::PVec<>(coords), valuesPtr->operator[](i)});
   }

   m_dims = Y->getDims();

   init();
}

void Result::init()
{
   total_pos = 0;
   if (classify)
   {
      for(auto &t : m_predictions)
      {
         int is_positive = t.val > threshold;
         total_pos += is_positive;
      }
   }
}

//--- output model to files
void Result::save(std::string prefix)
{
   if (m_predictions.empty())
      return;

   std::string fname_pred = prefix + "-predictions.csv";
   std::ofstream predfile;
   predfile.open(fname_pred);

   for(size_t d = 0; d < m_dims.size(); d++)
      predfile << "c" << d << ",";

   predfile << "y,pred_1samp,pred_avg,var,std" << std::endl;

   for (auto &t : m_predictions)
   {
      t.coords.save(predfile)
         << "," << to_string(t.val)
         << "," << to_string(t.pred_1sample)
         << "," << to_string(t.pred_avg)
         << "," << to_string(t.var)
         << "," << to_string(t.stds)
         << std::endl;
   }

   predfile.close();
}

///--- update RMSE and AUC

//model - holds samples (U matrices)
void Result::update(std::shared_ptr<const Model> model, bool burnin)
{
   if (m_predictions.size() == 0)
      return;

   const size_t NNZ = m_predictions.size();

   if (burnin)
   {
      double se_1sample = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample)
      for(size_t k = 0; k < m_predictions.size(); ++k)
      {
         auto &t = m_predictions[k];
         t.pred_1sample = model->predict(t.coords); //dot product of i'th columns in each U matrix
         se_1sample += square(t.val - t.pred_1sample);
      }

      burnin_iter++;
      rmse_1sample = sqrt(se_1sample / NNZ);

      if (classify)
      {
         auc_1sample = calc_auc(m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_1sample < b.pred_1sample;});
      }
   }
   else
   {
      double se_1sample = 0.0;
      double se_avg = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample, se_avg)
      for(size_t k = 0; k < m_predictions.size(); ++k)
      {
         auto &t = m_predictions[k];
         const double pred = model->predict(t.coords); //dot product of i'th columns in each U matrix
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
      rmse_1sample = sqrt(se_1sample / NNZ);
      rmse_avg = sqrt(se_avg / NNZ);

      if (classify)
      {
         auc_1sample = calc_auc(m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_1sample < b.pred_1sample;});

         auc_avg = calc_auc(m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_avg < b.pred_avg;});
      }
   }
}

std::ostream &Result::info(std::ostream &os, std::string indent)
{
   if (m_predictions.size())
   {
      std::uint64_t dtotal = 1;
      for(size_t d = 0; d < m_dims.size(); d++)
         dtotal *= m_dims[d];

      double test_fill_rate = 100. * m_predictions.size() / dtotal;

      os << indent << "Test data: " << m_predictions.size();

      os << " [";
      for(size_t d = 0; d < m_dims.size(); d++)
      {
         if(d == m_dims.size() - 1)
            os << m_dims[d];
         else
            os << m_dims[d] << " x ";
      }
      os << "]";

      os << " (" << test_fill_rate << "%)" << std::endl;
   }
   else
   {
      os << indent << "Test data: -" << std::endl;
   }

   if (classify)
   {
      double pos = 100. * (double)total_pos / (double)m_predictions.size();
      os << indent << "Binary classification threshold: " << threshold << std::endl;
      os << indent << "  " << pos << "% positives in test data" << std::endl;
   }

   return os;
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

} // end namespace
