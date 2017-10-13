#include <Utils/Distribution.h>
#include <Utils/utils.h>

#if defined(TEST) || defined (BENCH)

using namespace std;
using namespace Eigen;

int main()
{
#ifdef BENCH_NRANDN
    const int N = 2 * 1024;
    const int R = 10;
    {
        smurff::init_bmrng(1234);
        MatrixXd U;
        double start = tick();
        for(int i=0; i<R; ++i) {
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
        for(int i=0; i<R; ++i) {
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
        for(int i=0; i<R; ++i) {
           smurff::bmrandn_single(U);
        }
        double stop = tick();
        std::cout << "norm: " << U.norm() << std::endl;
        std::cout << "inplace single: " << stop - start << std::endl;
    }
    return 0;

#else
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

#if defined(BENCH_COND_NORMALWISHART)
        cout << "COND NORMAL WISHART\n" << endl;

        tie(mu_out, T_out) = smurff::CondNormalWishart(U, mu, kappa, T, nu);

        cout << "mu_out:\n" << mu_out << endl;
        cout << "T_out:\n" << T_out << endl;

        cout << "\n-----\n\n";
#endif

#if defined(BENCH_NORMAL_WISHART)
        cout << "NORMAL WISHART\n" << endl;

        tie(mu_out, T_out) = smurff::NormalWishart(mu, kappa, T, nu);
        cout << "mu_out:\n" << mu_out << endl;
        cout << "T_out:\n" << T_out << endl;
#endif

#if defined(BENCH_MVNORMAL)
        cout << "MVNORMAL\n" << endl;
        MatrixXd out = smurff::MvNormal(T, mu, 10);
        cout << "mu:\n" << mu << endl;
        cout << "T:\n" << T << endl;
        cout << "out:\n" << out << endl;
#endif
    }

#endif
}

#endif