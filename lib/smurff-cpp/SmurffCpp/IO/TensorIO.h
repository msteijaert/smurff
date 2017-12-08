#include <string>
#include <iostream>
#include <memory>

#include <SmurffCpp/Configs/TensorConfig.h>

namespace smurff { namespace tensor_io
{
   enum class TensorType
   {
      //sparse types
      none,
      sdt,
      sbt,
      tns,

      //dense types
      csv,
      ddt
   };

   std::shared_ptr<TensorConfig> read_tensor(const std::string& filename, bool isScarce);

   std::shared_ptr<TensorConfig> read_dense_float64_bin(std::istream& in);
   std::shared_ptr<TensorConfig> read_dense_float64_csv(std::istream& in);

   std::shared_ptr<TensorConfig> read_sparse_float64_bin(std::istream& in, bool isScarce);
   std::shared_ptr<TensorConfig> read_sparse_float64_tns(std::istream& in, bool isScarce);

   std::shared_ptr<TensorConfig> read_sparse_binary_bin(std::istream& in, bool isScarce);

   // ===

   void write_tensor(const std::string& filename, std::shared_ptr<const TensorConfig> tensorConfig);

   void write_dense_float64_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);
   void write_dense_float64_csv(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);

   void write_sparse_float64_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);
   void write_sparse_float64_tns(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);

   void write_sparse_binary_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);
}}