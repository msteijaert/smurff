from MatrixConfig cimport MatrixConfig

cdef extern from "<SmurffCpp/Sessions/ISession.h>" namespace "smurff":
    cdef cppclass ISession:
        void run()
        void step()
        void init()
        MatrixConfig getResult()
        MatrixConfig getSample(int mode)
