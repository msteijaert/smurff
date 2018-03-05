cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff":
    cdef cppclass NoiseTypes:
        pass

    cdef cppclass NoiseConfig:
        double getPrecision() const;
        void setPrecision(double value);

        double getSnInit() const;
        void setSnInit(double value);

        double getSnMax() const;
        void setSnMax(double value);

        double getThreshold() const;
        void setThreshold(double value);

        NoiseConfig() except +
        void setNoiseType(NoiseTypes value)

cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff::NoiseTypes":
    cdef NoiseTypes fixed
    cdef NoiseTypes adaptive
    cdef NoiseTypes probit
    cdef NoiseTypes noiseless
    cdef NoiseTypes unused