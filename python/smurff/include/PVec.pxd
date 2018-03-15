cdef extern from "<SmurffCpp/Utils/PVec.hpp>" namespace "smurff":
    cdef cppclass PVec "smurff::PVec<>":
        PVec(size_t size) except +
        size_t size()
        const int& operator[](size_t p)
        int& at(size_t p)