#include "DataWriter.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/TensorIO.h>

using namespace smurff;

void DataWriter::write(std::shared_ptr<const MatrixConfig> mc) const
{
   matrix_io::write_matrix(m_filename, mc);
}

void DataWriter::write(std::shared_ptr<const TensorConfig> tc) const
{
   tensor_io::write_tensor(m_filename, tc);
}
