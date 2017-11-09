from libcpp.memory cimport shared_ptr
from Config cimport Config
from Session cimport Session

cdef extern from "<SmurffCpp/Sessions/SessionFactory.h>" namespace "smurff":
   cdef cppclass SessionFactory:
      @staticmethod
      shared_ptr[Session] create_py_session(Config& cfg)
