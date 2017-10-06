#include "SparseFeat.h"

std::unique_ptr<SparseFeat> load_bcsr(const char* filename)
{
   SparseBinaryMatrix* A = read_sbm(filename);
   SparseFeat* sf = new SparseFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols);
   free_sbm(A);
   free(A); //SparseFeat should create a copy of aall A fields
   std::unique_ptr<SparseFeat> sf_ptr(sf);
   return sf_ptr;
}