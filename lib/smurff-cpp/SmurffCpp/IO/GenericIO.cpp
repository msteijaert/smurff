#include "GenericIO.h"

#include "MatrixIO.h"
#include "TensorIO.h"

using namespace smurff;

std::shared_ptr<TensorConfig> generic_io::read_data_config(const std::string& filename)
{
   try
   {
      //read will throw exception if file extension is not correct
      //for csv it will throw exception if mode != 2
      return matrix_io::read_matrix(filename);
   }
   catch(std::runtime_error& e)
   {
      try
      {
         //read will throw exception if file extension is not correct
         return tensor_io::read_tensor(filename);
      }
      catch(std::runtime_error& e)
      {
         throw std::runtime_error("Wrong file format " + filename);
      }
   }
}

void generic_io::write_data_config(const std::string& filename, std::shared_ptr<TensorConfig> tensorConfig)
{
   //call write_matrix or write_tensor
}