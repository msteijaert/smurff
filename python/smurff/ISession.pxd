from MatrixConfig cimport MatrixConfig

cdef extern from "<SmurffCpp/Sessions/ISession.h>" namespace "smurff":
    cdef cppclass ISession:
        void run()
        void step()
        MatrixConfig getResult()
        MatrixConfig getSample(int dim)
