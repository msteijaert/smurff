#include "TensorIO.h"

#include <fstream>
#include <algorithm>
#include <numeric>

#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/IO/GenericIO.h>

using namespace smurff;

#define EXTENSION_SDT ".sdt" //sparse double tensor (binary file)
#define EXTENSION_SBT ".sbt" //sparse binary tensor (binary file)
#define EXTENSION_TNS ".tns" //sparse tensor (txt file)
#define EXTENSION_CSV ".csv" //dense tensor (txt file)
#define EXTENSION_DDT ".ddt" //dense double tensor (binary file)

tensor_io::TensorType tensor_io::ExtensionToTensorType(const std::string& fname)
{
   std::size_t dotIndex = fname.find_last_of(".");
   if (dotIndex == std::string::npos)
   {
      THROWERROR("Extension is not specified in " + fname);
   }

   std::string extension = fname.substr(dotIndex);

   if (extension == EXTENSION_SDT)
   {
      return tensor_io::TensorType::sdt;
   }
   else if (extension == EXTENSION_SBT)
   {
      return tensor_io::TensorType::sbt;
   }
   else if (extension == EXTENSION_TNS)
   {
      return tensor_io::TensorType::tns;
   }
   else if (extension == EXTENSION_CSV)
   {
      return tensor_io::TensorType::csv;
   }
   else if (extension == EXTENSION_DDT)
   {
      return tensor_io::TensorType::ddt;
   }
   else
   {
      THROWERROR("Unknown file type: " + extension + " specified in " + fname);
   }
   return tensor_io::TensorType::none;
}

std::string TensorTypeToExtension(tensor_io::TensorType tensorType)
{
   switch (tensorType)
   {
   case tensor_io::TensorType::sdt:
      return EXTENSION_SDT;
   case tensor_io::TensorType::sbt:
      return EXTENSION_SBT;
   case tensor_io::TensorType::tns:
      return EXTENSION_TNS;
   case tensor_io::TensorType::csv:
       return EXTENSION_CSV;
   case tensor_io::TensorType::ddt:
      return EXTENSION_DDT;
   case tensor_io::TensorType::none:
      {
         THROWERROR("Unknown tensor type");
      }
   default:
      {
         THROWERROR("Unknown tensor type");
      }
   }
   return std::string();
}

std::shared_ptr<TensorConfig> tensor_io::read_tensor(const std::string& filename, bool isScarce)
{
   std::shared_ptr<TensorConfig> ret;

   TensorType tensorType = ExtensionToTensorType(filename);
   
   THROWERROR_FILE_NOT_EXIST(filename);

   switch (tensorType)
   {
   case tensor_io::TensorType::sdt:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         ret = tensor_io::read_sparse_float64_bin(fileStream, isScarce);
         break;
      }
   case tensor_io::TensorType::sbt:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         ret = tensor_io::read_sparse_binary_bin(fileStream, isScarce);
         break;
      }
   case tensor_io::TensorType::tns:
      {
         std::ifstream fileStream(filename);
         ret = tensor_io::read_sparse_float64_tns(fileStream, isScarce);
         break;
      }
   case tensor_io::TensorType::csv:
      {
         std::ifstream fileStream(filename);
         ret = tensor_io::read_dense_float64_csv(fileStream);
         break;
      }
   case tensor_io::TensorType::ddt:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         ret = tensor_io::read_dense_float64_bin(fileStream);
         break;
      }
   case tensor_io::TensorType::none:
      {
         THROWERROR("Unknown tensor type specified in " + filename);
      }
   default:
      {
         THROWERROR("Unknown tensor type specified in " + filename);
      }
   }

   ret->setFilename(filename);

   return ret;
}

std::shared_ptr<TensorConfig> tensor_io::read_dense_float64_bin(std::istream& in)
{
   std::uint64_t nmodes;
   in.read(reinterpret_cast<char*>(&nmodes), sizeof(std::uint64_t));

   std::vector<uint64_t> dims(nmodes);
   in.read(reinterpret_cast<char*>(dims.data()), dims.size() * sizeof(std::uint64_t));

   std::uint64_t nnz = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint64_t>());
   std::vector<double> values(nnz);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   return std::make_shared<TensorConfig>(std::move(dims), std::move(values), NoiseConfig());
}

std::shared_ptr<TensorConfig> tensor_io::read_dense_float64_csv(std::istream& in)
{
   std::stringstream ss;
   std::string line;
   std::string cell;

   // nmodes

   getline(in, line); 
   ss.clear();
   ss << line;
   std::uint64_t nmodes;
   ss >> nmodes;

   //dimentions

   getline(in, line);
   std::stringstream lineStream0(line);
   
   std::vector<uint64_t> dims(nmodes);

   std::uint64_t dim = 0;

   while (std::getline(lineStream0, cell, ',') && dim < nmodes)
   {
      ss.clear();
      ss << cell;
      std::uint64_t mode;
      ss >> mode;

      dims[dim++] = mode;
   }

   if(dim != nmodes)
   {
      THROWERROR("invalid number of dimensions");
   }

   //values

   std::getline(in, line);
   std::stringstream lineStream1(line);

   std::uint64_t nnz = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint64_t>());
   std::vector<double> values(nnz);

   std::uint64_t nval = 0;

   while (std::getline(lineStream1, cell, ',') && nval < nnz)
   {
      ss.clear();
      ss << cell;
      std::uint64_t value;
      ss >> value;

      values[nval++] = value;
   }

   if(nval != nnz)
   {
      THROWERROR("invalid number of values");
   }

   return std::make_shared<TensorConfig>(std::move(dims), std::move(values), NoiseConfig());
}

std::shared_ptr<TensorConfig> tensor_io::read_sparse_float64_bin(std::istream& in, bool isScarce)
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

   return std::make_shared<TensorConfig>(std::move(dims), std::move(columns), std::move(values), NoiseConfig(), isScarce);
}

std::shared_ptr<TensorConfig> tensor_io::read_sparse_float64_tns(std::istream& in, bool isScarce)
{
   std::stringstream ss;
   std::string line;
   std::string cell;

   // nmodes

   getline(in, line); 
   ss.clear();
   ss << line;
   std::uint64_t nmodes;
   ss >> nmodes;

   //dimentions

   getline(in, line);
   std::stringstream lineStream0(line);
   
   std::vector<uint64_t> dims(nmodes);

   std::uint64_t dim = 0;

   while (std::getline(lineStream0, cell, '\t') && dim < nmodes)
   {
      ss.clear();
      ss << cell;
      std::uint64_t mode;
      ss >> mode;

      dims[dim++] = mode;
   }

   if(dim != nmodes)
   {
      THROWERROR("invalid number of dimensions");
   }

   // nmodes

   getline(in, line); 
   ss.clear();
   ss << line;
   std::uint64_t nnz;
   ss >> nnz;

   //columns

   getline(in, line);
   std::stringstream lineStream1(line);

   std::vector<std::uint32_t> columns(nmodes * nnz);

   std::uint64_t col = 0;

   while (std::getline(lineStream1, cell, '\t') && col < (nmodes * nnz))
   {
      ss.clear();
      ss << cell;
      std::uint64_t column;
      ss >> column;

      columns[col++] = column;
   }

   if(col != nmodes * nnz)
   {
      THROWERROR("invalid number of coordinates");
   }
   
   std::for_each(columns.begin(), columns.end(), [](std::uint32_t& col){ col--; });

   //values

   getline(in, line);
   std::stringstream lineStream2(line);

   std::vector<double> values(nnz);

   std::uint64_t nval = 0;

   while (std::getline(lineStream2, cell, '\t') && nval < nnz)
   {
      ss.clear();
      ss << cell;
      std::uint64_t value;
      ss >> value;

      values[nval++] = value;
   }

   if(nval != nnz)
   {
      THROWERROR("invalid number of values");
   }

   return std::make_shared<TensorConfig>(std::move(dims), std::move(columns), std::move(values), NoiseConfig(), isScarce);
}

std::shared_ptr<TensorConfig> tensor_io::read_sparse_binary_bin(std::istream& in, bool isScarce)
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

   return std::make_shared<TensorConfig>(std::move(dims), std::move(columns), NoiseConfig(), isScarce);
}

// ======================================================================================================

void tensor_io::write_tensor(const std::string& filename, std::shared_ptr<const TensorConfig> tensorConfig)
{
   TensorType tensorType = ExtensionToTensorType(filename);
   switch (tensorType)
   {
   case tensor_io::TensorType::sdt:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         tensor_io::write_sparse_float64_bin(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::sbt:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         tensor_io::write_sparse_binary_bin(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::tns:
      {
         std::ofstream fileStream(filename);
         tensor_io::write_sparse_float64_tns(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::csv:
      {
         std::ofstream fileStream(filename);
         tensor_io::write_dense_float64_csv(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::ddt:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         tensor_io::write_dense_float64_bin(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::none:
      {
         THROWERROR("Unknown tensor type");
      }
   default:
      {
         THROWERROR("Unknown tensor type");
      }
   }
}

void tensor_io::write_dense_float64_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   std::uint64_t nmodes = tensorConfig->getNModes();
   const std::vector<std::uint64_t>& dims = tensorConfig->getDims();
   const std::vector<double>& values = tensorConfig->getValues();

   out.write(reinterpret_cast<const char*>(&nmodes), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void tensor_io::write_dense_float64_csv(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   std::uint64_t nmodes = tensorConfig->getNModes();

   out << nmodes << std::endl;

   const std::vector<std::uint64_t>& dims = tensorConfig->getDims();

   for(std::uint64_t i = 0; i < dims.size(); i++)
   {
      if(i == dims.size() - 1)
         out << dims[i];
      else
         out << dims[i] << ",";
   }

   out << std::endl;

   const std::vector<double>& values = tensorConfig->getValues();

   if(values.size() != tensorConfig->getNNZ())
   {
      THROWERROR("invalid number of values");
   }

   for(std::uint64_t i = 0; i < values.size(); i++)
   {
      if(i == values.size() - 1)
         out << values[i];
      else
         out << values[i] << ",";
   }

   out << std::endl;
}

void tensor_io::write_sparse_float64_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   std::uint64_t nmodes = tensorConfig->getNModes();
   std::uint64_t nnz = tensorConfig->getNNZ();
   const std::vector<std::uint64_t>& dims = tensorConfig->getDims();
   std::vector<std::uint32_t> columns = tensorConfig->getColumns(); //create copy of columns
   const std::vector<double>& values = tensorConfig->getValues();

   std::for_each(columns.begin(), columns.end(), [](std::uint32_t& col){ col++; });

   out.write(reinterpret_cast<const char*>(&nmodes), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(columns.data()), columns.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void tensor_io::write_sparse_float64_tns(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   std::uint64_t nmodes = tensorConfig->getNModes();
   std::uint64_t nnz = tensorConfig->getNNZ();
   const std::vector<std::uint64_t>& dims = tensorConfig->getDims();
   std::vector<std::uint32_t> columns = tensorConfig->getColumns(); //create copy of columns
   const std::vector<double>& values = tensorConfig->getValues();

   std::for_each(columns.begin(), columns.end(), [](std::uint32_t& col){ col++; });

   out << nmodes << std::endl;
   
   for(std::uint64_t i = 0; i < dims.size(); i++)
   {
      if(i == dims.size() - 1)
         out << dims[i];
      else
         out << dims[i] << "\t";
   }

   out << std::endl;

   out << nnz << std::endl;

   for(std::uint64_t i = 0; i < columns.size(); i++)
   {
      if(i == columns.size() - 1)
         out << columns[i];
      else
         out << columns[i] << "\t";
   }

   out << std::endl;

   for(std::uint64_t i = 0; i < values.size(); i++)
   {
      if(i == values.size() - 1)
         out << values[i];
      else
         out << values[i] << "\t";
   }

   out << std::endl;
}

void tensor_io::write_sparse_binary_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   std::uint64_t nmodes = tensorConfig->getNModes();
   std::uint64_t nnz = tensorConfig->getNNZ();
   const std::vector<std::uint64_t>& dims = tensorConfig->getDims();
   std::vector<std::uint32_t> columns = tensorConfig->getColumns(); //create copy of columns

   std::for_each(columns.begin(), columns.end(), [](std::uint32_t& col){ col++; });

   out.write(reinterpret_cast<const char*>(&nmodes), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(columns.data()), columns.size() * sizeof(std::uint32_t));
}
