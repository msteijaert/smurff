cdef extern from "<SmurffCpp/Sessions/Session.h>" namespace "smurff":
   cdef cppclass Session:
      void run()
