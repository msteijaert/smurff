#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/utils.h>
#include <SmurffCpp/Utils/counters.h>

using namespace std;
using namespace Eigen;

int main()
{
   smurff::init_bmrng(1234);
   {
      MatrixXd U(32,32 * 1024);
      U.setOnes();

      VectorXd mu(32);
      mu.setZero();

      MatrixXd T(32,32);
      T.setIdentity(32,32);
      T.array() /= 4;

      VectorXd mu_out;
      MatrixXd T_out;

      cout << "MVNORMAL\n" << endl;
      MatrixXd out = smurff::MvNormal(T, mu, 10);
      cout << "mu:\n" << mu << endl;
      cout << "T:\n" << T << endl;
      cout << "out:\n" << out << endl;
   }

   return 0;
}