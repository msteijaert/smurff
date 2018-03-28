from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from MacauPriorConfigItem cimport MacauPriorConfigItem

cdef extern from "<SmurffCpp/Configs/MacauPriorConfig.h>" namespace "smurff":

    double BETA_PRECISION_DEFAULT_VALUE "smurff::MacauPriorConfig::BETA_PRECISION_DEFAULT_VALUE"
    double TOL_DEFAULT_VALUE "smurff::MacauPriorConfig::TOL_DEFAULT_VALUE"

    cdef cppclass MacauPriorConfig:
        MacauPriorConfig() except +
        vector[shared_ptr[MacauPriorConfigItem]]& getConfigItems()