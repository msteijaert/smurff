#include "catch.hpp"

#include <SmurffCpp/SideInfo/SparseDoubleFeatSideInfo.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/linop.h>

using namespace smurff;

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE( "SparseDoubleFeatSideInfo/At_mul_A", "[At_mul_A] for SparseDoubleFeatSideInfo" )
{
    std::uint32_t rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
    std::uint32_t cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
    double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
    SparseDoubleFeatSideInfo si = SparseDoubleFeatSideInfo(6, 4, 9, rows, cols, vals);

    Eigen::MatrixXd AA(4, 4);
    si.At_mul_A(AA);
    REQUIRE( AA(0,0) == Approx(4.3801) );
    REQUIRE( AA(1,1) == Approx(2.4485) );
    REQUIRE( AA(2,2) == Approx(8.6420) );
    REQUIRE( AA(3,3) == Approx(5.9572) );
    
    REQUIRE( AA(1,0) == 0 );
    REQUIRE( AA(2,0) == Approx(3.8282) );
    REQUIRE( AA(3,0) == 0 );
    
    REQUIRE( AA(2,1) == 0 );
    REQUIRE( AA(3,1) == Approx(0.0714) );
    
    REQUIRE( AA(3,2) == 0 );
}

TEST_CASE( "SparseDoubleFeatSideInfo/A_mul_B", "[A_mul_B] for SparseDoubleFeatSideInfo" )
{
    std::uint32_t rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
    std::uint32_t cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
    double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
    SparseDoubleFeatSideInfo si = SparseDoubleFeatSideInfo(6, 4, 9, rows, cols, vals);

    Eigen::MatrixXd X(4, 6);
    X << 0., 0.6, 0., 0., 0., -0.82,
        0., 0., 0., 1.19, 0., 0.06,
        -0.76, 0., 1.48, 0., 1.95, 0.,
        2.54, 0., 0., 0., 0., 2.44;

    Eigen::MatrixXd AB = si.A_mul_B(X);
    REQUIRE( AB(0,0) == 0 );
    REQUIRE( AB(1,1) == 0 );
    REQUIRE( AB(2,2) == Approx(4.953) );
    REQUIRE( AB(3,3) == Approx(5.9536) );

    REQUIRE( AB(1,0) == Approx(-0.9044) );
    REQUIRE( AB(2,0) == Approx(3.8025) );
    REQUIRE( AB(3,0) == 0 );

    REQUIRE( AB(0,1) == Approx(-0.492) );
    REQUIRE( AB(0,2) == 0 );
    REQUIRE( AB(0,3) == Approx(-2.0008) );

    REQUIRE( AB(2,1) == Approx(1.3052) );
    REQUIRE( AB(3,1) == Approx(1.524) );

    REQUIRE( AB(1,2) == Approx(1.7612) );
    REQUIRE( AB(1,3) == Approx(0.1464) );

    REQUIRE( AB(3,2) == 0);

    REQUIRE( AB(2,3) == Approx(0.0888) );
}

TEST_CASE( "SparseDoubleFeatSideInfo/At_mul_Bt", "[At_mul_Bt] for SparseDoubleFeatSideInfo" )
{
    std::uint32_t rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
    std::uint32_t cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
    double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
    SparseDoubleFeatSideInfo si = SparseDoubleFeatSideInfo(6, 4, 9, rows, cols, vals);

    Eigen::MatrixXd X(4, 6);
    X << 0., 0.6, 0., 0., 0., -0.82,
        0., 0., 0., 1.19, 0., 0.06,
        -0.76, 0., 1.48, 0., 1.95, 0.,
        2.54, 0., 0., 0., 0., 2.44;
    
    Eigen::VectorXd Y(4);

    si.At_mul_Bt(Y, 0, X);

    REQUIRE( Y(0) == 0 );
    REQUIRE( Y(1) == Approx(-0.9044));
    REQUIRE( Y(2) == Approx(3.8025) );
    REQUIRE( Y(3) == 0 );
}

TEST_CASE( "SparseDoubleFeatSideInfo/add_Acol_mul_bt", "[add_Acol_mul_bt] for SparseDoubleFeatSideInfo" )
{
    std::uint32_t rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
    std::uint32_t cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
    double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
    SparseDoubleFeatSideInfo si = SparseDoubleFeatSideInfo(6, 4, 9, rows, cols, vals);
    
    Eigen::MatrixXd Z(4, 6);
    Z << 0., 0.6, 0., 0., 0., -0.82,
        0., 0., 0., 1.19, 0., 0.06,
        -0.76, 0., 1.48, 0., 1.95, 0.,
        2.54, 0., 0., 0., 0., 2.44;
    
    Eigen::VectorXd b(4);
    b << 1.4, 0., -0.46, 0.13;

    si.add_Acol_mul_bt(Z, 2, b);

    REQUIRE( Z(0,0) == 0 );
    REQUIRE( Z(0,1) == Approx(0.6) );
    REQUIRE( Z(0,2) == 0 );
    REQUIRE( Z(0,3) == Approx(2.072) );
    REQUIRE( Z(0,4) == Approx(3.556) );
    REQUIRE( Z(0,5) == Approx(-0.82) );

    REQUIRE( Z(1,0) == 0 );
    REQUIRE( Z(1,1) == 0 );
    REQUIRE( Z(1,2) == 0 );
    REQUIRE( Z(1,3) == Approx(1.19) );
    REQUIRE( Z(1,4) == 0 );
    REQUIRE( Z(1,5) == Approx(0.06) );

    REQUIRE( Z(2,0) == Approx(-0.76) );
    REQUIRE( Z(2,1) == 0 );
    REQUIRE( Z(2,2) == Approx(1.48) );
    REQUIRE( Z(2,3) == Approx(-0.6808) );
    REQUIRE( Z(2,4) == Approx(0.7816) );
    REQUIRE( Z(2,5) == 0 );

    REQUIRE( Z(3,0) == Approx(2.54) );
    REQUIRE( Z(3,1) == 0 );
    REQUIRE( Z(3,2) == 0 );
    REQUIRE( Z(3,3) == Approx(0.1924) );
    REQUIRE( Z(3,4) == Approx(0.3302) );
    REQUIRE( Z(3,5) == Approx(2.44) );
}

TEST_CASE( "SparseDoubleFeatSideInfo/col_square_sum", "[col_square_sum] for SparseDoubleFeatSideInfo" )
{
    std::uint32_t rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
    std::uint32_t cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
    double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
    SparseDoubleFeatSideInfo si = SparseDoubleFeatSideInfo(6, 4, 9, rows, cols, vals);

    Eigen::VectorXd out = si.col_square_sum();

    REQUIRE( out(0) == Approx(4.3801) );
    REQUIRE( out(1) == Approx(2.4485) );
    REQUIRE( out(2) == Approx(8.642) );
    REQUIRE( out(3) == Approx(5.9572) );
}

TEST_CASE( "SparseDoubleFeatSideInfo/compute_uhat", "[compute_uhat] for SparseDoubleFeatSideInfo" )
{
    std::uint32_t rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
    std::uint32_t cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
    double vals[9] = {0.6, -0.76, 1.48, 1.19, 2.44, 1.95, -0.82, 0.06, 2.54};
    SparseDoubleFeatSideInfo si = SparseDoubleFeatSideInfo(6, 4, 9, rows, cols, vals);
    
    Eigen::MatrixXd beta(6,4);
    beta << 1.4, 0., 0.76, 1.34,
            -2.32, 0.12, -1.3, 0.,
            0.45, 0.19, -1.87, 2.34,
            2.12, -1.43, -0.98, -2.71,
            0., 0., 1.10, 2.13,
            0.56, -1.3, 0, 0;

    Eigen::MatrixXd true_uhat(6,6);
    true_uhat << 0, 0, 0.0804, 0.0608, 4.6604, 3.2696,
                0.072, -0.0984, 0.1428, -0.1608, -7.826, 0,
                0.114, -0.1558, 0.3665, -3.1096, -3.8723, 5.7096,
                -0.858, 1.1726, -1.8643, -3.0616, 1.6448, -6.6124,
                0, 0, 0.1278, 1.628, 2.794, 5.1972,
                -0.78, 1.066, -1.547, -0.4256, 1.092, 0;

    Eigen::MatrixXd out(6,6);
    si.compute_uhat(out, beta);
    
    for (int i = 0; i < true_uhat.rows(); i++) {
        for (int j = 0; j < true_uhat.cols(); j++) {
            REQUIRE( out(i,j) == Approx(true_uhat(i,j)) );
        }
    }
}

TEST_CASE( "SparseDoubleFeatSideInfo/AtA_mul_B", "[AtA_mul_B] for SparseDoubleFeatSideInfo" )
{
    Eigen::MatrixXd out(6,6);
    const uint32_t rows[30] =  { 
                        0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                        3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5
                    };
    const uint32_t cols[30] =  {
                        2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5,
                        0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 0, 1, 2, 3, 4
                    };
    double vals[30] =   { 
                            0.0804, 0.0608, 4.6604, 3.2696,
                            0.072, -0.0984, 0.1428, -0.1608, -7.826,
                            0.114, -0.1558, 0.3665, -3.1096, -3.8723, 5.7096, 
                            -0.858, 1.1726, -1.8643, -3.0616, 1.6448, -6.6124,
                            0.1278, 1.628, 2.794, 5.1972, 
                            -0.78, 1.066, -1.547, -0.4256, 1.092 
                        };
    
    SparseDoubleFeatSideInfo A(6, 6, 30, rows, cols, vals);

    double reg = 0.76;
    Eigen::MatrixXd B(6,6);
    B <<    1.4, 0., 0.76, 1.34, 0.98, -1.98,
            -2.32, 0.12, -1.3, 0., 6.54, -3.12,
            0.45, 0.19, -1.87, 2.34, 0., 0.54,
            2.12, -1.43, -0.98, -2.71, 0., 2.65,
            0., 0., 1.10, 2.13, 0., 0.,
            0.56, -1.3, 0., 0., -0.43, -3.21; 

    Eigen::MatrixXd inner(6,6);
    smurff::linop::AtA_mul_Bx<6>(out, A, reg, B, inner);

    Eigen::MatrixXd true_out(6,6);
    true_out << -7.10631, 11.1661, -20.3844, 28.4183, 121.97, -194.977,
                -49.9681, 65.9712, -106.738, 34.3392, 748.891, -414.892,
                4.73854, -5.86421, 8.77816, 49.4195, 39.4612, 60.5787,
                14.0955, -18.1487, 30.9668, -26.6171, -49.6666, 284.69,
                8.66668, -11.8445, 19.024, 54.2326, 19.6878, 40.6309,
                -15.286, 20.4846, -39.7644, -35.1639, -44.7608, -352.293;

    for (int i = 0; i < true_out.rows(); i++) {
        for (int j = 0; j < true_out.cols(); j++) {
            REQUIRE( out(i,j) == Approx(true_out(i,j)) );
        }
    }
}