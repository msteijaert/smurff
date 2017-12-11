cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff":
    cdef cppclass NoiseTypes:
        pass

    cdef cppclass NoiseConfig:
        # for fixed gaussian noise
        double precision

        # for adaptive gausssian noise
        double sn_init
        double sn_max

        NoiseConfig() except +
        void setNoiseType(NoiseTypes value)

cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff::NoiseTypes":
    cdef NoiseTypes fixed
    cdef NoiseTypes adaptive
    cdef NoiseTypes probit
    cdef NoiseTypes noiseless
    cdef NoiseTypes unused