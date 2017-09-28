#include "tensor_io.h"

#include <fstream>
#include <algorithm>

using namespace smurff;

TensorConfig tensor_io::read_dense_float64(const std::string& filename)
{
   std::ifstream fileStream(filename, std::ios_base::binary);
   return tensor_io::read_dense_float64(fileStream);
}

TensorConfig tensor_io::read_dense_float64(std::istream& in)
{
   std::uint64_t nmodes;
   in.read(reinterpret_cast<char*>(&nmodes), sizeof(std::uint64_t));

   std::vector<uint64_t> dims(nmodes);
   in.read(reinterpret_cast<char*>(dims.data()), dims.size() * sizeof(std::uint64_t));

   std::uint64_t nnz = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint64_t>());
   std::vector<double> values(nnz);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   return TensorConfig(std::move(dims), std::move(values), NoiseConfig());
}

TensorConfig tensor_io::read_sparse_float64(const std::string& filename)
{
   std::ifstream fileStream(filename, std::ios_base::binary);
   return tensor_io::read_sparse_float64(fileStream);
}

TensorConfig tensor_io::read_sparse_float64(std::istream& in)
{
   std::uint64_t nmodes;
   in.read(reinterpret_cast<char*>(&nmodes), sizeof(std::uint64_t));

   std::vector<uint64_t> dims(nmodes);
   in.read(reinterpret_cast<char*>(dims.data()), dims.size() * sizeof(std::uint64_t));

   std::uint64_t nnz;
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   std::vector<std::uint32_t> columns(nmodes * nnz);

   for (std::uint64_t i = 0; i < nmodes; i++)
   {
      std::uint64_t dataOffset = i * nnz;
      in.read(reinterpret_cast<char*>(columns.data() + dataOffset), nnz * sizeof(std::uint32_t));
   }

   std::for_each(columns.begin(), columns.end(), [](std::uint32_t& col){ col--; });

   std::vector<double> values(nnz);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   return TensorConfig(std::move(dims), std::move(columns), std::move(values), NoiseConfig());
}

TensorConfig tensor_io::read_sparse_binary(const std::string& filename)
{
   std::ifstream fileStream(filename, std::ios_base::binary);
   return tensor_io::read_sparse_binary(fileStream);
}

TensorConfig tensor_io::read_sparse_binary(std::istream& in)
{
   std::uint64_t nmodes;
   in.read(reinterpret_cast<char*>(&nmodes), sizeof(std::uint64_t));

   std::vector<std::uint64_t> dims(nmodes);
   in.read(reinterpret_cast<char*>(dims.data()), dims.size() * sizeof(std::uint64_t));

   std::uint64_t nnz;
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   std::vector<std::uint32_t> columns(nmodes * nnz);

   for (std::uint64_t i = 0; i < nmodes; i++)
   {
      std::uint64_t dataOffset = i * nnz;
      in.read(reinterpret_cast<char*>(columns.data() + dataOffset), nnz * sizeof(std::uint32_t));
   }

   std::for_each(columns.begin(), columns.end(), [](std::uint32_t& col){ col--; });

   return TensorConfig(std::move(dims), std::move(columns), NoiseConfig());
}

void tensor_io::write_dense_float64(const std::string& filename, const TensorConfig& tensorConfig)
{
   std::ofstream fileStream(filename, std::ios_base::binary);
   tensor_io::write_dense_float64(fileStream, tensorConfig);
}

void tensor_io::write_dense_float64(std::ostream& out, const TensorConfig& tensorConfig)
{
   std::uint64_t nmodes = tensorConfig.getNModes();
   const std::vector<std::uint64_t>& dims = tensorConfig.getDims();
   const std::vector<double>& values = tensorConfig.getValues();

   out.write(reinterpret_cast<const char*>(&nmodes), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void tensor_io::write_sparse_float64(const std::string& filename, const TensorConfig& tensorConfig)
{
   std::ofstream fileStream(filename, std::ios_base::binary);
   tensor_io::write_sparse_float64(fileStream, tensorConfig);
}

void tensor_io::write_sparse_float64(std::ostream& out, const TensorConfig& tensorConfig)
{
   std::uint64_t nmodes = tensorConfig.getNModes();
   std::uint64_t nnz = tensorConfig.getNNZ();
   const std::vector<std::uint64_t>& dims = tensorConfig.getDims();
   std::vector<std::uint32_t> columns = tensorConfig.getColumns();
   const std::vector<double>& values = tensorConfig.getValues();

   std::for_each(columns.begin(), columns.end(), [](std::uint32_t& col){ col++; });

   out.write(reinterpret_cast<const char*>(&nmodes), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(columns.data()), columns.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void tensor_io::write_sparse_binary(const std::string& filename, const TensorConfig& tensorConfig)
{
   std::ofstream fileStream(filename, std::ios_base::binary);
   tensor_io::write_sparse_binary(fileStream, tensorConfig);
}

void tensor_io::write_sparse_binary(std::ostream& out, const TensorConfig& tensorConfig)
{
   std::uint64_t nmodes = tensorConfig.getNModes();
   std::uint64_t nnz = tensorConfig.getNNZ();
   const std::vector<std::uint64_t>& dims = tensorConfig.getDims();
   std::vector<std::uint32_t> columns = tensorConfig.getColumns();

   std::for_each(columns.begin(), columns.end(), [](std::uint32_t& col){ col++; });

   out.write(reinterpret_cast<const char*>(&nmodes), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(columns.data()), columns.size() * sizeof(std::uint32_t));
}