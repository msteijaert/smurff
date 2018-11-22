#include "catch.hpp"

#include "viennacl/matrix.hpp"

#include <Eigen/Core>

#include <SmurffCpp/Utils/Error.h>

TEST_CASE("RowMajor/ColMajor", "[viennacl]")
{
   typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenRowmajor;
   EigenRowmajor eigen_rowmajor(3, 2);
   eigen_rowmajor << 1, 2, 3, 4, 5, 6;
   SHOW(eigen_rowmajor);

   typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> EigenColmajor;
   EigenColmajor eigen_colmajor(3, 2);
   eigen_colmajor << 1, 2, 3, 4, 5, 6;
   SHOW(eigen_colmajor);

   Eigen::Map<EigenColmajor> eigen_implicit_transpose_rowmajor(eigen_rowmajor.data(), 2, 3);
   SHOW(eigen_implicit_transpose_rowmajor);

   Eigen::Map<EigenRowmajor> eigen_implicit_transpose_colmajor(eigen_colmajor.data(), 2, 3);
   SHOW(eigen_implicit_transpose_colmajor);

   using namespace viennacl;

   matrix<double, row_major> vcl_rowmajor_from_eigen_rowmajor(3, 2);
   copy(eigen_rowmajor, vcl_rowmajor_from_eigen_rowmajor);
   SHOW(vcl_rowmajor_from_eigen_rowmajor);

   matrix<double, row_major> vcl_rowmajor_from_eigen_colmajor(3, 2);
   copy(eigen_colmajor, vcl_rowmajor_from_eigen_colmajor);
   SHOW(vcl_rowmajor_from_eigen_colmajor);

   matrix<double, column_major> vcl_colmajor_from_eigen_rowmajor(3, 2);
   copy(eigen_rowmajor, vcl_colmajor_from_eigen_rowmajor);
   SHOW(vcl_colmajor_from_eigen_rowmajor);

   matrix<double, column_major> vcl_colmajor_from_eigen_colmajor(3, 2);
   copy(eigen_colmajor, vcl_colmajor_from_eigen_colmajor);
   SHOW(vcl_colmajor_from_eigen_colmajor);

   matrix<double, row_major> vcl_rowmajor_from_eigen_colmajor_transposed(2, 3);
   copy(eigen_implicit_transpose_colmajor, vcl_rowmajor_from_eigen_colmajor_transposed);
   SHOW(vcl_rowmajor_from_eigen_colmajor_transposed);
}
