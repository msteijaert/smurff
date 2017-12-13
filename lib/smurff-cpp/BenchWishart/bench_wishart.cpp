#include <iostream>

#include <SmurffCpp/Utils/Distribution.h>
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

      double kappa = 2;

      MatrixXd T(32,32);
      T.setIdentity(32,32);
      T.array() /= 4;

      int nu = 3;

      VectorXd mu_out;
      MatrixXd T_out;

      cout << "COND NORMAL WISHART\n" << endl;

      tie(mu_out, T_out) = smurff::CondNormalWishart(U, mu, kappa, T, nu);

      cout << "mu_out:\n" << mu_out << endl;
      cout << "T_out:\n" << T_out << endl;

      cout << "\n-----\n\n";

      cout << "NORMAL WISHART\n" << endl;

      tie(mu_out, T_out) = smurff::NormalWishart(mu, kappa, T, nu);
      cout << "mu_out:\n" << mu_out << endl;
      cout << "T_out:\n" << T_out << endl;
   }

   return 0;
}