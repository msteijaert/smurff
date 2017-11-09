cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff":
    cdef cppclass NoiseTypes:
        pass

    cdef cppclass NoiseConfig:
        NoiseConfig()
