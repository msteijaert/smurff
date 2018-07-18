#include "catch.hpp"

#include <SmurffCpp/SideInfo/SparseDoubleFeatSideInfo.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/linop.h>

using namespace smurff;

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE( "SparseDoubleFeatSideInfo/At_mul_A", "[At_mul_A] for SparseDoubleFeatSideInfo" )
{
    int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
    int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
    double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
    auto side_info_ptr = std::make_shared<SparseDoubleFeat>(6, 4, 9, rows, cols, vals);
    SparseDoubleFeatSideInfo si = SparseDoubleFeatSideInfo(side_info_ptr);

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