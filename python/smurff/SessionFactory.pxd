from libcpp.memory cimport shared_ptr
from Config cimport Config
from ISession cimport ISession

cdef extern from "<SmurffCpp/Sessions/SessionFactory.h>" namespace "smurff":
    cdef cppclass SessionFactory:
        @staticmethod
        shared_ptr[ISession] create_py_session(Config& cfg) except +
