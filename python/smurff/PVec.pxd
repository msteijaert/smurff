cdef extern from "<SmurffCpp/Utils/PVec.hpp>" namespace "smurff":
    cdef cppclass PVec:
        size_t size()
        const int& operator[](size_t p)