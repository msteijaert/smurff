#include "SparseDoubleFeat.h"

std::unique_ptr<SparseDoubleFeat> load_csr(const char* filename)
{
    SparseDoubleMatrix* A = read_sdm(filename);
    SparseDoubleFeat* sf = new SparseDoubleFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols, A->vals);
    free_sdm(A);
    free(A); //SparseDoubleFeat should create a copy of aall A fields
    std::unique_ptr<SparseDoubleFeat> sf_ptr(sf);
    return sf_ptr;
}