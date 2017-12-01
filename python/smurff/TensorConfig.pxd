from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from NoiseConfig cimport NoiseConfig

cdef extern from "<SmurffCpp/Configs/TensorConfig.h>" namespace "smurff":
    cdef cppclass TensorConfig:
        void setNoiseConfig(const NoiseConfig& value)