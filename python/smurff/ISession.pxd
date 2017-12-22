from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from MatrixConfig cimport MatrixConfig
from ResultItem cimport ResultItem

cdef extern from "<SmurffCpp/Sessions/ISession.h>" namespace "smurff":
    cdef cppclass ISession:
        void run() except +
        void step() except +
        void init() except +
        shared_ptr[vector[ResultItem]] getResult()
        MatrixConfig getSample(int mode)
        double getRmseAvg()
