#include "MatrixData.h"

using namespace smurff;

int MatrixData::nmode() const
{
   return 2;
}

std::ostream& MatrixData::info(std::ostream& os, std::string indent)
{
   Data::info(os, indent);
   double train_fill_rate = 100. * nnz() / size();
   os << indent << "Size: " << nnz() << " [" << nrow() << " x " << ncol() << "] (" << train_fill_rate << "%)\n";
   return os;
}