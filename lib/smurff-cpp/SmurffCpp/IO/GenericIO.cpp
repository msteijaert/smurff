#include "GenericIO.h"

#include <memory>

#include "MatrixIO.h"
#include "TensorIO.h"

#include "DataWriter.h"

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

std::shared_ptr<TensorConfig> generic_io::read_data_config(const std::string& filename, bool isScarce)
{
   THROWERROR_FILE_NOT_EXIST(filename);

   try
   {
      if(matrix_io::ExtensionToMatrixType(filename) == matrix_io::MatrixType::csv)
      {
         THROWERROR("csv input is not allowed for matrix data");
      }

      //read will throw exception if file extension is not correct
      return matrix_io::read_matrix(filename, isScarce);
   }
   catch(std::runtime_error& ex1)
   {
      try
      {
         if(tensor_io::ExtensionToTensorType(filename) == tensor_io::TensorType::csv)
         {
            THROWERROR("csv input is not allowed for tensor data");
         }

         //read will throw exception if file extension is not correct
         return tensor_io::read_tensor(filename, isScarce);
      }
      catch(std::runtime_error& ex2)
      {
         THROWERROR("Could not read tensor/matrix file: " + filename + 
                    "\n  Both Matrix and Tensor file formats tried. Both gave an error.\n  " +
                     ex2.what() + "\n  " +
                     ex1.what());
      }
   }
}

void generic_io::write_data_config(const std::string& filename, std::shared_ptr<TensorConfig> tensorConfig)
{
   tensorConfig->write(std::make_shared<DataWriter>(filename));
}

bool generic_io::file_exists(const std::string& filepath)
{
   std::ifstream infile(filepath.c_str());
   return infile.good();
}
