from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from MatrixConfig cimport MatrixConfig

cdef extern from "<SmurffCpp/Configs/MacauPriorConfig.h>" namespace "smurff":
    cdef cppclass MacauPriorConfigItem:
        MacauPriorConfigItem() except +
        void setSideInfo(shared_ptr[MatrixConfig] value)
        void setTol(double value)
        void setDirect(bool value)
