from libcpp.vector cimport vector

from PVec cimport PVec
from RootFile cimport RootFile

cdef extern from "<SmurffCpp/ResultItem.h>" namespace "smurff":
    cdef cppclass ResultItem:
        PVec coords
        double val
        double pred_1sample
        double pred_avg
        double var
