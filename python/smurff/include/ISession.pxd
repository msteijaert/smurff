from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from MatrixConfig cimport MatrixConfig
from ResultItem cimport ResultItem
from RootFile cimport RootFile

cdef extern from "<SmurffCpp/Sessions/ISession.h>" namespace "smurff":
    cdef cppclass ISession:
        void run() except +
        bool step() except +
        bool interrupted() except +
        void init() except +

        shared_ptr[vector[ResultItem]] getResult()
        MatrixConfig getSample(int mode)
        shared_ptr[RootFile] getRootFile()

        double getRmseAvg()
