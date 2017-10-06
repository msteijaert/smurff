#include "matrix_io_old.h"

std::unique_ptr<SparseFeat> load_bcsr(const char* filename)
{
   SparseBinaryMatrix* A = read_sbm(filename);
   SparseFeat* sf = new SparseFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols);
   free_sbm(A);
   std::unique_ptr<SparseFeat> sf_ptr(sf);
   return sf_ptr;
}

std::unique_ptr<SparseDoubleFeat> load_csr(const char* filename)
{
    struct SparseDoubleMatrix* A = read_sdm(filename);
    SparseDoubleFeat* sf = new SparseDoubleFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols, A->vals);
    delete A;
    std::unique_ptr<SparseDoubleFeat> sf_ptr(sf);
    return sf_ptr;
}