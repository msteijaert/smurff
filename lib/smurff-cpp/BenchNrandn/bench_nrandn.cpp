#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/utils.h>

using namespace std;
using namespace Eigen;

int main()
{
   const int N = 2 * 1024;
   const int R = 10;
   {
      smurff::init_bmrng(1234);
      MatrixXd U;
      double start = tick();
      for(int i=0; i<R; ++i) 
      {
         U = smurff::nrandn(N,N);
      }
      double stop = tick();
      std::cout << "norm: " << U.norm() << std::endl;
      std::cout << "nullary: " << stop - start << std::endl;
   }
   {
      smurff::init_bmrng(1234);
      MatrixXd U(N,N);
      double start = tick();
      for(int i=0; i<R; ++i) 
      {
         smurff::bmrandn(U);
      }
      double stop = tick();
      std::cout << "norm: " << U.norm() << std::endl;
      std::cout << "inplace omp: " << stop - start << std::endl;
   }
   {
      smurff::init_bmrng(1234);
      MatrixXd U(N,N);
      double start = tick();
      for(int i=0; i<R; ++i) 
      {
         smurff::bmrandn_single(U);
      }
      double stop = tick();
      std::cout << "norm: " << U.norm() << std::endl;
      std::cout << "inplace single: " << stop - start << std::endl;
   }
   return 0;
}