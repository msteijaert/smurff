#include <string>
#include <iostream>

#include "TensorConfig.h"

namespace smurff { namespace tensor_io
{
   TensorConfig read_tensor(const std::string& filename);

   TensorConfig read_dense_float64_bin(std::istream& in);
   TensorConfig read_dense_float64_txt(std::istream& in);

   TensorConfig read_sparse_float64_bin(std::istream& in);
   TensorConfig read_sparse_float64_txt(std::istream& in);

   TensorConfig read_sparse_binary_bin(std::istream& in);
   TensorConfig read_sparse_binary_txt(std::istream& in);

   void write_tensor(const std::string& filename, const TensorConfig& tensorConfig);

   void write_dense_float64_bin(std::ostream& out, const TensorConfig& tensorConfig);
   void write_dense_float64_txt(std::ostream& out, const TensorConfig& tensorConfig);

   void write_sparse_float64_bin(std::ostream& out, const TensorConfig& tensorConfig);
   void write_sparse_float64_txt(std::ostream& out, const TensorConfig& tensorConfig);

   void write_sparse_binary_bin(std::ostream& out, const TensorConfig& tensorConfig);
   void write_sparse_binary_txt(std::ostream& out, const TensorConfig& tensorConfig);
}}