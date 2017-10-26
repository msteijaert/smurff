#include "ScarceMatrixData.h"

using namespace smurff;

ScarceMatrixData::ScarceMatrixData(Eigen::SparseMatrix<double> Y)
   : MatrixDataTempl<Eigen::SparseMatrix<double> >(Y)
{
   name = "ScarceMatrixData [with NAs]";
}

void ScarceMatrixData::init_pre()
{
   MatrixDataTempl<Eigen::SparseMatrix<double> >::init_pre();

   // check no rows, nor cols withouth data
   for(unsigned i=0; i<getCenter<Eigen::SparseMatrix<double>>()->getYcPtr()->size(); ++i)
   {
      auto &v = getCenter<Eigen::SparseMatrix<double>>()->getYcPtr()->operator[](i);
      auto &count = num_empty[i];
      for (int j = 0; j < v.cols(); j++)
      {
         if (v.col(j).nonZeros() == 0) count++;
      }
   }
}

double ScarceMatrixData::train_rmse(const SubModel& model) const 
{
   double se = 0.;
   #pragma omp parallel for schedule(guided) reduction(+:se)
   for(int c=0; c<Y.cols();++c) 
   {
       for (Eigen::SparseMatrix<double>::InnerIterator it(Y, c); it; ++it) 
       {
           se += square(it.value() - predict({(int)it.row(), (int)it.col()}, model));
       }
   }
   return sqrt( se / Y.nonZeros() );
}

std::ostream& ScarceMatrixData::info(std::ostream& os, std::string indent)
{
    MatrixDataTempl<Eigen::SparseMatrix<double> >::info(os, indent);
    if (num_empty[0]) os << indent << "  Warning: " << num_empty[0] << " empty rows\n";
    if (num_empty[1]) os << indent << "  Warning: " << num_empty[1] << " empty cols\n";
    return os;
}

void ScarceMatrixData::get_pnm(const SubModel& model, int mode, int n, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
   auto &Y = getCenter<Eigen::SparseMatrix<double>>()->getYcPtr()->at(mode);
   const int num_latent = model.nlatent();
   const auto &Vf = model.V(mode);
   const int local_nnz = Y.col(n).nonZeros();
   const int total_nnz = Y.nonZeros();
   const double alpha = noise().getAlpha();

   bool in_parallel = (local_nnz >10000) || ((double)local_nnz > (double)total_nnz / 100.);
   if (in_parallel) {
       const int task_size = ceil(local_nnz / 100.0);
       auto from = Y.outerIndexPtr()[n];
       auto to = Y.outerIndexPtr()[n+1];
       thread_vector<Eigen::VectorXd> rrs(Eigen::VectorXd::Zero(num_latent));
       thread_vector<Eigen::MatrixXd> MMs(Eigen::MatrixXd::Zero(num_latent, num_latent));
       for(int j=from; j<to; j+=task_size) 
       {
           #pragma omp task shared(model,Y,Vf,rrs,MMs)
           {
               auto &my_rr = rrs.local();
               auto &my_MM = MMs.local();

               for(int i=j; i<std::min(j+task_size,to); ++i)
               {
                   auto val = Y.valuePtr()[i];
                   auto idx = Y.innerIndexPtr()[i];
                   const auto &col = model.V(mode).col(idx);
                   my_rr.noalias() += col * val;
                   my_MM.triangularView<Eigen::Lower>() += col * col.transpose();
               }

               // make MM complete
               my_MM.triangularView<Eigen::Upper>() = my_MM.transpose();

           }
       }
       #pragma omp taskwait
       
       // accumulate + add noise
       MM += MMs.combine() * alpha;
       rr += rrs.combine() * alpha;
   } else {
      Eigen::VectorXd my_rr = Eigen::VectorXd::Zero(num_latent);
      Eigen::MatrixXd my_MM = Eigen::MatrixXd::Zero(num_latent, num_latent);
      for (Eigen::SparseMatrix<double>::InnerIterator it(Y, n); it; ++it) {
         const auto &col = Vf.col(it.row());
         my_rr.noalias() += col * it.value();
         my_MM.triangularView<Eigen::Lower>() += col * col.transpose();
      }

      // make MM complete
      my_MM.triangularView<Eigen::Upper>() = my_MM.transpose();

      //add noise
      my_rr.array() *= alpha;
      my_MM.array() *= alpha;

      // add to global
      rr += my_rr;
      MM += my_MM;
   }
}

void ScarceMatrixData::update_pnm(const SubModel &,int)
{
}

int ScarceMatrixData::nna() const
{
   return size() - nnz();
}
