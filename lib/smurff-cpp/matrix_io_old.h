#pragma once

std::unique_ptr<SparseFeat> load_bcsr(const char* filename);
std::unique_ptr<SparseDoubleFeat> load_csr(const char* filename);