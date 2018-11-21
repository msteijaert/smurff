#include "ScarceMatrixData.h"

#include <SmurffCpp/VMatrixExprIterator.hpp>
#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

#include <SmurffCpp/Utils/ThreadVector.hpp>

using namespace smurff;

ScarceMatrixData::ScarceMatrixData(Eigen::SparseMatrix<float> Y)
   : MatrixDataTempl<Eigen::SparseMatrix<float> >(Y)
{
   name = "ScarceMatrixData [with NAs]";
}

void ScarceMatrixData::init_pre()
{
   MatrixDataTempl<Eigen::SparseMatrix<float> >::init_pre();

   // check no rows, nor cols withouth data
   for(std::uint64_t mode = 0; mode < nmode(); ++mode)
   {
      auto& m = this->Y(mode);
      auto& count = num_empty[mode];
      for (int j = 0; j < m.cols(); j++)
      {
         if (m.col(j).nonZeros() == 0) 
            count++;
      }
   }
}

float ScarceMatrixData::train_rmse(const SubModel& model) const 
{
   return std::sqrt(sumsq(model) / this->nnz());
}

std::ostream& ScarceMatrixData::info(std::ostream& os, std::string indent)
{
    MatrixDataTempl<Eigen::SparseMatrix<float> >::info(os, indent);
    if (num_empty[0]) os << indent << "  Warning: " << num_empty[0] << " empty rows\n";
    if (num_empty[1]) os << indent << "  Warning: " << num_empty[1] << " empty cols\n";
    return os;
}

void ScarceMatrixData::getMuLambda(const SubModel& model, std::uint32_t mode, int n, Eigen::VectorXf& rr, Eigen::MatrixXf& MM) const
{
   auto &Y = this->Y(mode);
   const int num_latent = model.nlatent();
   const std::int64_t local_nnz = Y.col(n).nonZeros();
   const std::int64_t total_nnz = Y.nonZeros();
   auto from = Y.outerIndexPtr()[n];
   auto to = Y.outerIndexPtr()[n+1];

   auto getMuLambdaBasic = [&model, this, mode, n](int from, int to, Eigen::VectorXf& rr, Eigen::MatrixXf& MM) -> void
   {
       auto &Y = this->Y(mode);
       auto Vf = *model.CVbegin(mode);
       auto &ns = noise();

       for(int i = from; i < to; ++i)
       {
           auto val = Y.valuePtr()[i];
           auto idx = Y.innerIndexPtr()[i];
           const auto &col = Vf.col(idx);
           auto pos = this->pos(mode, n, idx);
           float noisy_val = ns.sample(model, pos, val);
           rr.noalias() += col * noisy_val;
           MM.triangularView<Eigen::Lower>() +=  ns.getAlpha() * col * col.transpose();
       }

       // make MM complete
       MM.triangularView<Eigen::Upper>() = MM.transpose();
    };

   

   bool in_parallel = (local_nnz >10000) || ((float)local_nnz > (float)total_nnz / 100.);
   if (in_parallel) 
   {
       const int task_size = ceil(local_nnz / 100.0);
       thread_vector<Eigen::VectorXf> rrs(Eigen::VectorXf::Zero(num_latent));
       thread_vector<Eigen::MatrixXf> MMs(Eigen::MatrixXf::Zero(num_latent, num_latent));

       for(int j = from; j < to; j += task_size) 
       {
           #pragma omp task shared(rrs, MMs)
           getMuLambdaBasic(j, std::min(j + task_size, to), rrs.local(), MMs.local());
       }
       #pragma omp taskwait
       
       // accumulate 
       MM += MMs.combine();
       rr += rrs.combine();
   } 
   else 
   {
      Eigen::VectorXf my_rr = Eigen::VectorXf::Zero(num_latent);
      Eigen::MatrixXf my_MM = Eigen::MatrixXf::Zero(num_latent, num_latent);

      getMuLambdaBasic(from, to, my_rr, my_MM);

      // add to global
      rr += my_rr;
      MM += my_MM;
   }
}

void ScarceMatrixData::update_pnm(const SubModel &, std::uint32_t mode)
{
   //can not cache VV because of scarceness
}

std::uint64_t ScarceMatrixData::nna() const
{
   return size() - this->nnz(); //nrows * ncols - nnz
}

float ScarceMatrixData::var_total() const
{
   float cwise_mean = this->sum() / this->nnz();
   float se = 0.0;

   #pragma omp parallel for schedule(guided) reduction(+:se)
   for (int k = 0; k < Y().outerSize(); ++k)
   {
      for (Eigen::SparseMatrix<float>::InnerIterator it(Y(), k); it; ++it)
      {
         se += std::pow(it.value() - cwise_mean, 2);
      }
   }

   float var = se / this->nnz();
   if (var <= 0.0 || std::isnan(var))
   {
      // if var cannot be computed using 1.0
      var = 1.0;
   }

   return var;
}

float ScarceMatrixData::sumsq(const SubModel& model) const
{
   float sumsq = 0.0;

   #pragma omp parallel for schedule(guided) reduction(+:sumsq)
   for (int j = 0; j < Y().outerSize(); j++) 
   {
      for (Eigen::SparseMatrix<float>::InnerIterator it(Y(), j); it; ++it) 
      {
         sumsq += std::pow(model.predict({static_cast<int>(it.row()), static_cast<int>(it.col())})- it.value(), 2);
      }
   }

   return sumsq;
}
