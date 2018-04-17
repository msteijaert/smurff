from libcpp.vector cimport vector

cdef extern from "<SmurffCpp/Utils/PVec.hpp>" namespace "smurff":
    cdef cppclass PVec "smurff::PVec<>":
        vector[int] as_vector() except +
 