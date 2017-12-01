from libcpp.vector cimport vector

from MatrixConfig cimport MatrixConfig
from ResultItem cimport ResultItem

cdef extern from "<SmurffCpp/Sessions/ISession.h>" namespace "smurff":
    cdef cppclass ISession:
        void run()
        void step()
        void init()
        vector[ResultItem] getResult()
        MatrixConfig getSample(int mode)
