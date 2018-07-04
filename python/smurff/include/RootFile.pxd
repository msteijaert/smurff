from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from StepFile cimport StepFile

cdef extern from "<SmurffCpp/Utils/RootFile.h>" namespace "smurff":
    cdef cppclass RootFile:
        string getFullPath() 
        string getOptionsFileName() 
        vector[shared_ptr[StepFile]] openSampleStepFiles()