from libc.stdint cimport *
from libcpp.vector cimport vector

cdef extern from "<SmurffCpp/Utils/PVec.hpp>" namespace "smurff":
    cdef cppclass PVec "smurff::PVec<>":
        vector[int64_t] as_vector() except +
 
