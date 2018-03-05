from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from MatrixConfig cimport MatrixConfig
from TensorConfig cimport TensorConfig

cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff":
    cdef cppclass PriorTypes:
        pass

    cdef cppclass ModelInitTypes:
        pass

    PriorTypes stringToPriorType(string name)
    ModelInitTypes stringToModelInitType(string name)

    cdef cppclass Config:
        #-- train and test
        shared_ptr[TensorConfig] getTrain()
        void setTrain(shared_ptr[TensorConfig] value)

        shared_ptr[TensorConfig] getTest()
        void setTest(shared_ptr[TensorConfig] value)

        #-- sideinfo
        vector[shared_ptr[MatrixConfig]]& getSideInfo()

        #-- aux data
        vector[shared_ptr[TensorConfig]]& getAuxData()

        #-- priors
        vector[PriorTypes]& getPriorTypes()

        #-- init model
        void setModelInitType(ModelInitTypes value)

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

        #-- for macau priors
        void setLambdaBeta(double value)
        void setTol(double value)
        void setDirect(bool value)

        #-- binary classification
        void setClassify(bool value)
        void setThreshold(double value)
