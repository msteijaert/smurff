#include <numeric>

#include "TensorUtils.h"

#include <SmurffCpp/Utils/Error.h>

Eigen::MatrixXd smurff::tensor_utils::dense_to_eigen(const smurff::TensorConfig& tensorConfig)
{
   if(!tensorConfig.isDense())
   {
      THROWERROR("tensor config should be dense");
   }

   if(tensorConfig.getNModes() != 2)
   {
      THROWERROR("Invalid number of dimensions. Tensor can not be converted to matrix.");
   }

   return Eigen::Map<const Eigen::MatrixXd>(tensorConfig.getValues().data(), tensorConfig.getDims()[0], tensorConfig.getDims()[1]);
}

Eigen::MatrixXd smurff::tensor_utils::dense_to_eigen(smurff::TensorConfig& tensorConfig)
{
   const smurff::TensorConfig& tc = tensorConfig;
   return smurff::tensor_utils::dense_to_eigen(tc);
}

template<>
Eigen::SparseMatrix<double> smurff::tensor_utils::sparse_to_eigen<const smurff::TensorConfig>(const smurff::TensorConfig& tensorConfig)
{
   if(tensorConfig.isDense())
   {
      THROWERROR("tensor config should be sparse");
   }

   if(tensorConfig.getNModes() != 2)
   {
      THROWERROR("Invalid number of dimensions. Tensor can not be converted to matrix.");
   }

   const auto &columns = tensorConfig.getColumns();
   const auto &values = tensorConfig.getValues();

   Eigen::SparseMatrix<double> out(tensorConfig.getDims()[0], tensorConfig.getDims()[1]);

   std::vector<Eigen::Triplet<double> > triplets;
   for(std::uint64_t i = 0; i < tensorConfig.getNNZ(); i++)
   {
      double val = values[i];
      std::uint32_t row = columns[i];
      std::uint32_t col = columns[i + tensorConfig.getNNZ()];
      triplets.push_back(Eigen::Triplet<double>(row, col, val));
   }

   out.setFromTriplets(triplets.begin(), triplets.end());

   return out;
}

template<>
Eigen::SparseMatrix<double> smurff::tensor_utils::sparse_to_eigen<smurff::TensorConfig>(smurff::TensorConfig& tensorConfig)
{
   return smurff::tensor_utils::sparse_to_eigen<const smurff::TensorConfig>(tensorConfig);
}

smurff::MatrixConfig smurff::tensor_utils::tensor_to_matrix(const smurff::TensorConfig& tensorConfig)
{
   if(tensorConfig.getNModes() != 2)
   {
      THROWERROR("Invalid number of dimentions. Tensor can not be converted to matrix.");
   }

   if(tensorConfig.isDense())
   {
      return smurff::MatrixConfig(tensorConfig.getDims()[0], tensorConfig.getDims()[1],
         tensorConfig.getValues(),
         tensorConfig.getNoiseConfig());
   }
   else if(tensorConfig.isBinary())
   {
      return smurff::MatrixConfig(tensorConfig.getDims()[0], tensorConfig.getDims()[1],
         tensorConfig.getColumns(),
         tensorConfig.getNoiseConfig(),
         tensorConfig.isScarce());
   }
   else
   {
      return smurff::MatrixConfig(tensorConfig.getDims()[0], tensorConfig.getDims()[1],
         tensorConfig.getColumns(), tensorConfig.getValues(),
         tensorConfig.getNoiseConfig(),
         tensorConfig.isScarce());
   }
}

std::ostream& smurff::tensor_utils::operator << (std::ostream& os, const TensorConfig& tc)
{
   const std::vector<double>& values = tc.getValues();
   const std::vector<std::uint32_t>& columns = tc.getColumns();

   os << "columns: " << std::endl;
   for(std::uint64_t i = 0; i < columns.size(); i++)
      os << columns[i] << ", ";
   os << std::endl;

   os << "values: " << std::endl;
   for(std::uint64_t i = 0; i < values.size(); i++)
      os << values[i] << ", ";
   os << std::endl;

   if(tc.getNModes() == 2)
   {
      os << "dims: " << tc.getDims()[0] << " " << tc.getDims()[1] << std::endl;

      Eigen::SparseMatrix<double> X(tc.getDims()[0], tc.getDims()[1]);

      std::vector<Eigen::Triplet<double> > triplets;
      for(std::uint64_t i = 0; i < tc.getNNZ(); i++)
         triplets.push_back(Eigen::Triplet<double>(columns[i], columns[i + tc.getNNZ()], values[i]));

      os << "NTriplets: " << triplets.size() << std::endl;

      X.setFromTriplets(triplets.begin(), triplets.end());

      os << X << std::endl;
   }

   return os;
}

Eigen::MatrixXd smurff::tensor_utils::slice( const TensorConfig& tensorConfig
                                           , const std::array<std::uint64_t, 2>& fixedDims
                                           , const std::unordered_map<std::uint64_t, std::uint32_t>& dimCoords)
{
   if (fixedDims[0] == fixedDims[1])
   {
      THROWERROR("fixedDims should contain 2 unique dimension numbers");
   }

   for (const std::uint64_t& fd : fixedDims)
   {
      if (fd > tensorConfig.getNModes() - 1)
      {
         THROWERROR("fixedDims should contain only valid for tensorConfig dimension numbers");
      }
   }

   if (dimCoords.size() != (tensorConfig.getNModes() -  2))
   {
      THROWERROR("dimsCoords.size() should be the same as tensorConfig.getNModes() - 2");
   }

   for (const std::unordered_map<std::uint64_t, std::uint32_t>::value_type& dc : dimCoords)
   {
      if (dc.first == fixedDims[0] || dc.first == fixedDims[1])
      {
         THROWERROR("dimCoords and fixedDims should not intersect");
      }

      if (dc.first >= tensorConfig.getNModes())
      {
         THROWERROR("dimCoords should contain only valid for tensorConfig dimension numbers");
      }

      if (dc.second >= tensorConfig.getDims()[dc.first])
      {
         THROWERROR("dimCoords should contain valid coord values for corresponding dimensions");
      }
   }

   std::unordered_map<std::uint64_t, std::vector<std::uint32_t>::const_iterator> dimColumns;
   for (const std::unordered_map<std::uint64_t, std::uint32_t>::value_type& dc : dimCoords)
   {
      std::size_t dimOffset = dc.first * tensorConfig.getValues().size();
      dimColumns[dc.first] = tensorConfig.getColumns().begin() + dimOffset;
   }

   Eigen::MatrixXd sliceMatrix(tensorConfig.getDims()[fixedDims[0]], tensorConfig.getDims()[fixedDims[1]]);
   for (std::size_t i = 0; i < tensorConfig.getValues().size(); i++)
   {
      bool dimCoordsMatchColumns =
         std::accumulate( dimCoords.begin()
                        , dimCoords.end()
                        , true
                        , [&](bool acc, const std::unordered_map<std::uint64_t, std::uint32_t>::value_type& dc)
                          {
                             return acc & (*(dimColumns[dc.first] + i) == dc.second);
                          }
                        );

      if (dimCoordsMatchColumns)
      {
         std::uint32_t d0_coord =
            tensorConfig.getColumns()[fixedDims[0] * tensorConfig.getValues().size() + i];
         std::uint32_t d1_coord =
            tensorConfig.getColumns()[fixedDims[1] * tensorConfig.getValues().size() + i];
         sliceMatrix(d0_coord, d1_coord) = tensorConfig.getValues()[i];
      }
   }
   return sliceMatrix;
}
