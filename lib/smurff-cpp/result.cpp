#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>

#include "Data.h"
#include "model.h"
#include "utils.h"
#include "result.h"

using namespace std;
using namespace Eigen;

namespace smurff {

void Result::set(SparseMatrixD Y) {
    for (int k = 0; k < Y.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
            predictions.push_back({(int)it.row(), (int)it.col(), it.value()});
        }
    }
    nrows = Y.rows();
    ncols = Y.cols();
    init();
}

void Result::init() {
    total_pos = 0;
    if (classify) {
        for(auto &t : predictions) {
            int is_positive = t.val > threshold;
            total_pos += is_positive;
        }
    }
}


double Result::rmse_using_globalmean(double mean) {
    double se = 0.;
    for(auto t : predictions) se += square(t.val - mean);
    return sqrt( se / predictions.size() );
}

double Result::rmse_using_modemean(const Data &data, int mode) {
     const unsigned N = predictions.size();
     double se = 0.;
     for(auto t : predictions) {
        int n = mode == 0 ? t.col : t.row;
        double pred = data.mean(mode, n);
        se += square(t.val - pred);
     }
     return sqrt( se / N );
}

//--- output model to files
void Result::save(std::string prefix) {
    if (predictions.empty()) return;
    std::string fname_pred = prefix + "-predictions.csv";
    std::ofstream predfile;
    predfile.open(fname_pred);
    predfile << "row,col,y,pred_1samp,pred_avg,var,std\n";
    for ( auto &t : predictions) {
        predfile
                << to_string( t.row  )
         << "," << to_string( t.col  )
         << "," << to_string( t.val  )
         << "," << to_string( t.pred )
         << "," << to_string( t.pred_avg )
         << "," << to_string( t.var )
         << "," << to_string( t.stds )
         << "\n";
    }
    predfile.close();

}

///--- update RMSE and AUC

void Result::update(const Model &model, const Data &data,  bool burnin)
{
    if (predictions.size() == 0) return;
    const unsigned N = predictions.size();

    if (burnin) {
        double se = 0.0;
        #pragma omp parallel for schedule(guided) reduction(+:se)
        for(unsigned k=0; k<predictions.size(); ++k) {
            auto &t = predictions[k];
            t.pred = model.predict({t.col, t.row}, data);
            se += square(t.val - t.pred);
        }
        burnin_iter++;
        rmse = sqrt( se / N );
    } else {
        double se = 0.0, se_avg = 0.0;
        #pragma omp parallel for schedule(guided) reduction(+:se, se_avg)
        for(unsigned k=0; k<predictions.size(); ++k) {
            auto &t = predictions[k];
            const double pred = model.predict({t.col, t.row}, data);
            se += square(t.val - pred);
            double delta = pred - t.pred_avg;
            double pred_avg = (t.pred_avg + delta / (sample_iter + 1));
            t.var += delta * (pred - pred_avg);
            const double inorm = 1.0 / sample_iter;
            t.stds = sqrt(t.var * inorm);
            t.pred_avg = pred_avg;
            t.pred = pred;
            se_avg += square(t.val - pred_avg);
        }
        sample_iter++;
        rmse = sqrt( se / N );
        rmse_avg = sqrt( se_avg / N );
    }

    update_auc();
}

//macau ProbitNoise eval
/*
eval_rmse(MatrixData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
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
/*
std::pair<double,double> eval_rmse_tensor(
		SparseMode & sparseMode,
		const int Nepoch,
		Eigen::VectorXd & predictions,
		Eigen::VectorXd & predictions_var,
		std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
		double mean_value)
{
  auto& U = samples[0];

  const int nmodes = samples.size();
  const int num_latents = U->rows();

  const unsigned N = sparseMode.values.size();
  double se = 0.0, se_avg = 0.0;

  if (N == 0) {
    // No test data, returning NaN's
    return std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN());
  }

  if (N != predictions.size()) {
    throw std::runtime_error("Ytest.size() and predictions.size() must be equal.");
  }
	if (sparseMode.row_ptr.size() - 1 != U->cols()) {
    throw std::runtime_error("U.cols() and sparseMode size must be equal.");
	}

#pragma omp parallel for schedule(dynamic, 2) reduction(+:se, se_avg)
  for (int n = 0; n < U->cols(); n++) {
    Eigen::VectorXd u = U->col(n);
    for (int j = sparseMode.row_ptr(n);
             j < sparseMode.row_ptr(n + 1);
             j++)
    {
      VectorXi idx = sparseMode.indices.row(j);
      double pred = mean_value;
      for (int d = 0; d < num_latents; d++) {
        double tmp = u(d);

        for (int m = 1; m < nmodes; m++) {
          tmp *= (*samples[m])(d, idx(m - 1));
        }
        pred += tmp;
      }

      double pred_avg;
      if (Nepoch == 0) {
        pred_avg = pred;
      } else {
        double delta = pred - predictions(j);
        pred_avg = (predictions(j) + delta / (Nepoch + 1));
        predictions_var(j) += delta * (pred - pred_avg);
      }
      se     += square(sparseMode.values(j) - pred);
      se_avg += square(sparseMode.values(j) - pred_avg);
      predictions(j) = pred_avg;
    }
  }
  const double rmse = sqrt(se / N);
  const double rmse_avg = sqrt(se_avg / N);
  return std::make_pair(rmse, rmse_avg);
}
*/

void Result::update_auc()
{
    if (!classify) return;
    std::sort(predictions.begin(), predictions.end(),
            [this](const Item &a, const Item &b) { return a.pred < b.pred;});

    int num_positive = 0;
    int num_negative = 0;
    auc = .0;
    for(auto &t : predictions) {
        int is_positive = t.val > threshold;
        int is_negative = !is_positive;
        num_positive += is_positive;
        num_negative += is_negative;
        auc += is_positive * num_negative;
    }

    auc /= num_positive;
    auc /= num_negative;
}

/*
inline double auc(Eigen::VectorXd & pred, Eigen::VectorXd & test)
{
	Eigen::VectorXd stack_x(pred.size());
	Eigen::VectorXd stack_y(pred.size());
	double auc = 0.0;

	if (pred.size() == 0) {
		return NAN;
	}

	std::vector<unsigned int> permutation( pred.size() );
	for(unsigned int i = 0; i < pred.size(); i++) {
		permutation[i] = i;
	}
	std::sort(permutation.begin(), permutation.end(), [&pred](unsigned int a, unsigned int b) { return pred[a] < pred[b];});

	double NP = test.sum();
	double NN = test.size() - NP;
	//Build stack_x and stack_y
	stack_x[0] = test[permutation[0]];
	stack_y[0] = 1-stack_x[0];
	for(int i=1; i < pred.size(); i++) {
		stack_x[i] = stack_x[i-1] + test[permutation[i]];
		stack_y[i] = stack_y[i-1] + 1 - test[permutation[i]];
	}

	for(int i=0; i < pred.size() - 1; i++) {
		auc += (stack_x(i+1) - stack_x(i)) * stack_y(i+1); //TODO:Make it Eigen
	}

	return auc / (NP*NN);
}
*/

std::ostream &Result::info(std::ostream &os, std::string indent, const Data &data)
{
    if (predictions.size()) {
        double test_fill_rate = 100. * predictions.size() / nrows / ncols;
        os << indent << "Test data: " << predictions.size() << " [" << nrows << " x " << ncols << "] (" << test_fill_rate << "%)\n";
        os << indent << "RMSE using globalmean: " << rmse_using_globalmean(data.global_mean) << endl;
        os << indent << "RMSE using colmean: " << rmse_using_modemean(data,0) << endl;
        os << indent << "RMSE using rowmean: " << rmse_using_modemean(data,1) << endl;
    } else {
        os << indent << "Test data: -\n";
    }
    if (classify) {
        double pos = 100. * (double)total_pos / (double)predictions.size();
        os << indent << "Binary classification threshold: " << threshold << "\n";
        os << indent << "  " << pos << "% positives in test data\n";
    }
    return os;
}

} // end namespace
