#include "MatrixData.h"

using namespace smurff;

std::uint64_t MatrixData::nmode() const
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

int MatrixData::nrow() const
{
   return dim(0);
}

int MatrixData::ncol() const
{
   return dim(1);
}


// return position of value at position 'n' in mode `mode'
// and at posisiton 'm' in the other mode
PVec<> MatrixData::pos(int mode, int n, int m) const
{
    if (mode == 0) return PVec<>({n,m});
    assert(mode == 1);
    return PVec<>({m,n});
}
