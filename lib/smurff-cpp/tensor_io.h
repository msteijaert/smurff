#include <string>
#include <iostream>

#include "TensorConfig.h"

namespace smurff { namespace tensor_io
{
   TensorConfig read_dense_float64(const std::string& filename);
   TensorConfig read_dense_float64(std::istream& in);

   TensorConfig read_sparse_float64(const std::string& filename);
   TensorConfig read_sparse_float64(std::istream& in);

   TensorConfig read_sparse_binary(const std::string& filename);
   TensorConfig read_sparse_binary(std::istream& in);

   void write_dense_float64(const std::string& filename, const TensorConfig& tensorConfig);
   void write_dense_float64(std::ostream& out, const TensorConfig& tensorConfig);

   void write_sparse_float64(const std::string& filename, const TensorConfig& tensorConfig);
   void write_sparse_float64(std::ostream& out, const TensorConfig& tensorConfig);

   void write_sparse_binary(const std::string& filename, const TensorConfig& tensorConfig);
   void write_sparse_binary(std::ostream& out, const TensorConfig& tensorConfig);
}}