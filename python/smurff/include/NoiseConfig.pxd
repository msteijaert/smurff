from libcpp.string cimport string

cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff":
    cdef cppclass NoiseConfig:

        NoiseConfig() except +
        void setPrecision(double value);
        void setSnInit(double value);
        void setSnMax(double value);
        void setThreshold(double value);
        void setNoiseType(string value)
