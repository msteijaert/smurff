from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from MatrixConfig cimport MatrixConfig
from TensorConfig cimport TensorConfig
from SideInfoConfig cimport SideInfoConfig

cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff":
    cdef cppclass PriorTypes:
        pass

    int BURNIN_DEFAULT_VALUE "smurff::Config::BURNIN_DEFAULT_VALUE"
    int NSAMPLES_DEFAULT_VALUE "smurff::Config::NSAMPLES_DEFAULT_VALUE"
    int NUM_LATENT_DEFAULT_VALUE "smurff::Config::NUM_LATENT_DEFAULT_VALUE"
    const char* SAVE_PREFIX_DEFAULT_VALUE "smurff::Config::SAVE_PREFIX_DEFAULT_VALUE"
    const char* SAVE_EXTENSION_DEFAULT_VALUE "smurff::Config::SAVE_EXTENSION_DEFAULT_VALUE"
    int SAVE_FREQ_DEFAULT_VALUE "smurff::Config::SAVE_FREQ_DEFAULT_VALUE"
    int CHECKPOINT_FREQ_DEFAULT_VALUE "smurff::Config::CHECKPOINT_FREQ_DEFAULT_VALUE"
    int VERBOSE_DEFAULT_VALUE "smurff::Config::VERBOSE_DEFAULT_VALUE"
    const char* STATUS_DEFAULT_VALUE "smurff::Config::STATUS_DEFAULT_VALUE"
    double BETA_PRECISION_DEFAULT_VALUE "smurff::Config::BETA_PRECISION_DEFAULT_VALUE"
    bool ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE "smurff::Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE"
    double THRESHOLD_DEFAULT_VALUE "smurff::Config::THRESHOLD_DEFAULT_VALUE"
    int RANDOM_SEED_DEFAULT_VALUE "smurff::Config::RANDOM_SEED_DEFAULT_VALUE"

    double TOL_DEFAULT_VALUE "smurff::SideInfoConfig::TOL_DEFAULT_VALUE"

    cdef cppclass Config:
        #-- train and test
        shared_ptr[TensorConfig] getTrain()
        void setTrain(shared_ptr[TensorConfig] value)

        shared_ptr[TensorConfig] getTest()
        void setTest(shared_ptr[TensorConfig] value)

        #-- sideinfo
        vector[shared_ptr[SideInfoConfig]]& addSideInfoConfig(int mode, shared_ptr[SideInfoConfig] config)

        #-- aux data
        vector[shared_ptr[TensorConfig]]& addAuxData(shared_ptr[TensorConfig])

        #-- priors
        vector[PriorTypes]& setPriorTypes(vector[string])

        #-- init model
        void setModelInitType(string)

        #-- save
        void setSavePrefix(string value)
        void setSaveExtension(string value)
        void setSaveFreq(int value)

        #-- general
        void setRandomSeedSet(bool value)
        void setRandomSeed(int value)
        void setVerbose(int value)
        void setCsvStatus(string value)

        int getBurnin()
        void setBurnin(int value)

        int getNSamples()
        void setNSamples(int value)

        void setNumLatent(int value)

        #-- binary classification
        void setClassify(bool value)
        void setThreshold(double value)

        void save(string fname)
