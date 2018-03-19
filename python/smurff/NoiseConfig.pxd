from libcpp.string cimport string

cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff":
    cdef cppclass NoiseTypes:
        pass

    NoiseTypes stringToNoiseType(string name)

    NoiseTypes NOISE_TYPE_DEFAULT_VALUE "smurff::NoiseConfig::NOISE_TYPE_DEFAULT_VALUE"

    cdef cppclass NoiseConfig:

        NoiseConfig() except +
        NoiseConfig(NoiseTypes nt) except +

        double getPrecision() const;
        void setPrecision(double value);

        double getSnInit() const;
        void setSnInit(double value);

        double getSnMax() const;
        void setSnMax(double value);

        double getThreshold() const;
        void setThreshold(double value);

        void setNoiseType(NoiseTypes value)

cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff::NoiseTypes":
    cdef NoiseTypes fixed
    cdef NoiseTypes adaptive
    cdef NoiseTypes probit
    cdef NoiseTypes noiseless
    cdef NoiseTypes unused