#pragma once

#include <memory>

#include <SmurffCpp/Utils/utils.h>
#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/DataTensors/SparseMode.h>

namespace smurff {

class Model;
class Data;

template<typename Item, typename Compare>
double calc_auc(const std::vector<Item> &predictions,
                double threshold,
                const Compare &compare)
{
    auto sorted_predictions = predictions;
    std::sort(sorted_predictions.begin(), sorted_predictions.end(), compare);

    int num_positive = 0;
    int num_negative = 0;
    double auc = .0;

    for(auto &t : sorted_predictions)
    {
        int is_positive = t.val > threshold;
        int is_negative = !is_positive;
        num_positive += is_positive;
        num_negative += is_negative;
        auc += is_positive * num_negative;
    }

    auc /= num_positive;
    auc /= num_negative;
    return auc;
}

template<typename Item>
double calc_auc(const std::vector<Item> &predictions, double threshold)
{
   return calc_auc(predictions, threshold, [](const Item &a, const Item &b) { return a.pred < b.pred;});
}

struct Result
{
   //-- test set
   struct Item
   {
      std::uint32_t row;
      std::uint32_t col;

      double val;
      double pred_1sample;
      double pred_avg;
      double var;
      double stds;
   };

   //sparse representation of test matrix
   std::vector<Item> predictions;

   //number of rows and columns in test matrix
   std::uint64_t m_nrows;
   std::uint64_t m_ncols;

   //Y - test sparse matrix
   void set(std::shared_ptr<TensorConfig> Y);

   //-- prediction metrics
   void update(std::shared_ptr<const Model> model, std::shared_ptr<Data> data,  bool burnin);
   double rmse_avg = NAN;
   double rmse_1sample = NAN;
   double auc_avg = NAN;
   double auc_1sample = NAN;
   int sample_iter = 0;
   int burnin_iter = 0;

   double rmse_using_globalmean(double);
   double rmse_using_modemean( std::shared_ptr<Data> data, int mode);

   // general

public:
   void save(std::string fname_prefix);

private:
   void init();

public:
   std::ostream &info(std::ostream &os, std::string indent, std::shared_ptr<Data> data);

   //-- for binary classification
   int total_pos = -1;
   bool classify = false;
   double threshold;

   void setThreshold(double t)
   {
      threshold = t; classify = true;
   }
};

}; // end namespace smurff