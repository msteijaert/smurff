#SETUP PROJECT
set (PROJECT smurff-cpp)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES "../Version.h"
                        "../Model.h"
                        "../result.h"
                        "../SparseFeat.h"
                        "../SparseDoubleFeat.h"
                        "../VMatrixIterator.hpp"
                        "../ConstVMatrixIterator.hpp"
                        "../VMatrixExprIterator.hpp"
                        "../ConstVMatrixExprIterator.hpp"

                        "../Utils/TruncNorm.h"
                        "../Utils/InvNormCdf.h"
                        "../Utils/Distribution.h"
                        "../Utils/MatrixUtils.h"
                        "../Utils/TensorUtils.h"
                        "../Utils/PVec.hpp"
                        "../Utils/chol.h"
                        "../Utils/counters.h"
                        "../Utils/linop.h"
                        "../Utils/omp_util.h"
                        "../Utils/utils.h"
                        "../Utils/Error.h"
                        "../ThreadVector.hpp"

                        "../IO/ini.h"
                        "../IO/INIReader.h"
                        "../IO/GenericIO.h"
                        "../IO/MatrixIO.h"
                        "../IO/TensorIO.h"
                        "../IO/IDataWriter.h"
                        "../IO/DataWriter.h"

                        "../Configs/Config.h"
                        "../Configs/MatrixConfig.h"
                        "../Configs/TensorConfig.h"
                        "../Configs/NoiseConfig.h"

                        "../Noises/AdaptiveGaussianNoise.h"
                        "../Noises/FixedGaussianNoise.h"
                        "../Noises/ProbitNoise.h"
                        "../Noises/Noiseless.h"
                        "../Noises/UnusedNoise.h"
                        "../Noises/INoiseModel.h"
                        "../Noises/NoiseFactory.h"

                        "../Priors/GlobalPrior.h"
                        "../Priors/ILatentPrior.h"
                        "../Priors/NormalPrior.h"
                        "../Priors/SpikeAndSlabPrior.h"
                        "../Priors/MacauPrior.hpp"
                        "../Priors/MacauOnePrior.hpp"
                        "../Priors/PriorFactory.h"

                        "../DataMatrices/Data.h"
                        "../DataMatrices/DenseMatrixData.h"
                        "../DataMatrices/FullMatrixData.hpp"
                        "../DataMatrices/MatricesData.h"
                        "../DataMatrices/MatrixData.h"
                        "../DataMatrices/MatrixDataTempl.hpp"
                        "../DataMatrices/ScarceBinaryMatrixData.h"
                        "../DataMatrices/ScarceMatrixData.h"
                        "../DataMatrices/SparseMatrixData.h"
                        "../DataMatrices/MatrixDataFactory.h"
                        "../DataMatrices/IDataCreator.h"
                        "../DataMatrices/DataCreator.h"

                        "../DataTensors/TensorData.h"
                        "../DataTensors/SparseMode.h"
                        "../DataTensors/TensorDataFactory.h"

                        "../Sessions/ISession.h"
                        "../Sessions/BaseSession.h"
                        "../Sessions/Session.h"
                        "../Sessions/CmdSession.h"
                        "../Sessions/PythonSession.h"
                        "../Sessions/SessionFactory.h")
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../Version.cpp"
                        "../Model.cpp"
                        "../result.cpp"

                        "../Utils/TruncNorm.cpp"
                        "../Utils/InvNormCdf.cpp"
                        "../Utils/Distribution.cpp"
                        "../Utils/MatrixUtils.cpp"
                        "../Utils/TensorUtils.cpp"
                        "../Utils/chol.cpp"
                        "../Utils/counters.cpp"
                        "../Utils/linop.cpp"
                        "../Utils/omp_util.cpp"

                        "../IO/ini.c"
                        "../IO/INIReader.cpp"
                        "../IO/GenericIO.cpp"
                        "../IO/MatrixIO.cpp"
                        "../IO/TensorIO.cpp"
                        "../IO/DataWriter.cpp"

                        "../Configs/Config.cpp"
                        "../Configs/MatrixConfig.cpp"
                        "../Configs/TensorConfig.cpp"
                        "../Configs/NoiseConfig.cpp"

                        "../Noises/AdaptiveGaussianNoise.cpp"
                        "../Noises/FixedGaussianNoise.cpp"
                        "../Noises/ProbitNoise.cpp"
                        "../Noises/Noiseless.cpp"
                        "../Noises/UnusedNoise.cpp"
                        "../Noises/NoiseFactory.cpp"

                        "../Priors/GlobalPrior.cpp"
                        "../Priors/ILatentPrior.cpp"
                        "../Priors/NormalPrior.cpp"
                        "../Priors/SpikeAndSlabPrior.cpp"
                        "../Priors/MacauPrior.cpp"
                        "../Priors/PriorFactory.cpp"

                        "../DataMatrices/Data.cpp"
                        "../DataMatrices/DenseMatrixData.cpp"
                        "../DataMatrices/MatricesData.cpp"
                        "../DataMatrices/MatrixData.cpp"
                        "../DataMatrices/ScarceBinaryMatrixData.cpp"
                        "../DataMatrices/ScarceMatrixData.cpp"
                        "../DataMatrices/SparseMatrixData.cpp"
                        "../DataMatrices/MatrixDataFactory.cpp"
                        "../DataMatrices/DataCreator.cpp"

                        "../DataTensors/TensorData.cpp"
                        "../DataTensors/SparseMode.cpp"
                        "../DataTensors/TensorDataFactory.cpp"

                        "../Sessions/BaseSession.cpp"
                        "../Sessions/Session.cpp"
                        "../Sessions/CmdSession.cpp"
                        "../Sessions/PythonSession.cpp"
                        "../Sessions/SessionFactory.cpp")
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_library (${PROJECT} STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Libraries")

#SETUP INCLUDES
include_directories(../)
include_directories(../..)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${ARGP_INCLUDE_PATH})

#INSTALL
install (TARGETS ${PROJECT} ARCHIVE DESTINATION lib)
install (FILES ../ResultItem.h DESTINATION include/SmurffCpp)
install (FILES ../Utils/Error.h DESTINATION include/SmurffCpp/Utils)
install (FILES ../Utils/PVec.hpp DESTINATION include/SmurffCpp/Utils)
install (FILES ../Configs/Config.h DESTINATION include/SmurffCpp/Configs)
install (FILES ../Configs/MatrixConfig.h DESTINATION include/SmurffCpp/Configs)
install (FILES ../Configs/TensorConfig.h DESTINATION include/SmurffCpp/Configs)
install (FILES ../Configs/NoiseConfig.h DESTINATION include/SmurffCpp/Configs)
install (FILES ../Sessions/SessionFactory.h DESTINATION include/SmurffCpp/Sessions)
install (FILES ../Sessions/ISession.h DESTINATION include/SmurffCpp/Sessions)
