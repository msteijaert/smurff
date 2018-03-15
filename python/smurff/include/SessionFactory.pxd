from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from Config cimport Config
from ISession cimport ISession

cdef extern from "<SmurffCpp/Sessions/SessionFactory.h>" namespace "smurff":
    cdef cppclass SessionFactory:
        @staticmethod
        shared_ptr[ISession] create_py_session_from_config "create_py_session"(Config& cfg) except +

        @staticmethod
        shared_ptr[ISession] create_py_session_from_root_path "create_py_session"(const string& rootPath) except +
