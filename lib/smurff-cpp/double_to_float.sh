#!/bin/sh

# FILES="
# Smurff/smurff.cpp
# SmurffMPI/MPISession.h
# SmurffMPI/MPIMacauPrior.cpp
# SmurffMPI/MPIMacauPrior.h
# SmurffMPI/mpi_smurff.cpp
# SmurffMPI/MPIPriorFactory.cpp
# SmurffMPI/MPISession.cpp
# SmurffMPI/MPIPriorFactory.h
# Tests/TestsLinop.cpp
# Tests/TestsSmurff.cpp
# Tests/TestsMatrixUtils.cpp
# Tests/TestsTensor.cpp
# Tests/TestsPVec.cpp
# Tests/tests.cpp
# Tests/TestsSparseDoubleFeatSideInfo.cpp
# SmurffCpp/DataMatrices/DataCreatorBase.cpp
# SmurffCpp/DataMatrices/DataCreator.cpp
# SmurffCpp/DataMatrices/Data.h
# SmurffCpp/DataMatrices/ScarceMatrixData.h
# SmurffCpp/DataMatrices/DenseMatrixData.cpp
# SmurffCpp/DataMatrices/Data.cpp
# SmurffCpp/DataMatrices/SparseMatrixData.h
# SmurffCpp/DataMatrices/MatrixData.h
# SmurffCpp/DataMatrices/DataCreator.h
# SmurffCpp/DataMatrices/SparseMatrixData.cpp
# SmurffCpp/DataMatrices/DataCreatorBase.h
# SmurffCpp/DataMatrices/MatrixDataFactory.h
# SmurffCpp/DataMatrices/MatrixData.cpp
# SmurffCpp/DataMatrices/ScarceMatrixData.cpp
# SmurffCpp/DataMatrices/MatricesData.cpp
# SmurffCpp/DataMatrices/MatrixDataFactory.cpp
# SmurffCpp/DataMatrices/DenseMatrixData.h
# SmurffCpp/DataMatrices/MatricesData.h
# SmurffCpp/DataMatrices/IDataCreator.h
# SmurffCpp/Noises/FixedGaussianNoise.cpp
# SmurffCpp/Noises/AdaptiveGaussianNoise.cpp
# SmurffCpp/Noises/GaussianNoise.h
# SmurffCpp/Noises/ProbitNoise.cpp
# SmurffCpp/Noises/INoiseModel.cpp
# SmurffCpp/Noises/ProbitNoise.h
# SmurffCpp/Noises/SampledGaussianNoise.h
# SmurffCpp/Noises/GaussianNoise.cpp
# SmurffCpp/Noises/FixedGaussianNoise.h
# SmurffCpp/Noises/NoiseFactory.h
# SmurffCpp/Noises/INoiseModel.h
# SmurffCpp/Noises/UnusedNoise.cpp
# SmurffCpp/Noises/SampledGaussianNoise.cpp
# SmurffCpp/Noises/UnusedNoise.h
# SmurffCpp/Noises/AdaptiveGaussianNoise.h
# SmurffCpp/Noises/NoiseFactory.cpp
# SmurffCpp/Version.h
# SmurffCpp/Priors/SpikeAndSlabPrior.cpp
# SmurffCpp/Priors/NormalPrior.h
# SmurffCpp/Priors/MacauPrior.h
# SmurffCpp/Priors/PriorFactory.cpp
# SmurffCpp/Priors/NormalPrior.cpp
# SmurffCpp/Priors/PriorFactory.h
# SmurffCpp/Priors/ILatentPrior.h
# SmurffCpp/Priors/NormalOnePrior.cpp
# SmurffCpp/Priors/NormalOnePrior.h
# SmurffCpp/Priors/IPriorFactory.h
# SmurffCpp/Priors/SpikeAndSlabPrior.h
# SmurffCpp/Priors/MacauOnePrior.h
# SmurffCpp/Priors/MacauPrior.cpp
# SmurffCpp/Priors/MacauOnePrior.cpp
# SmurffCpp/Priors/ILatentPrior.cpp
# SmurffCpp/Version.cpp
# SmurffCpp/result.h
# SmurffCpp/Utils/Error.h
# SmurffCpp/Utils/Distribution.cpp
# SmurffCpp/Utils/omp_util.h
# SmurffCpp/Utils/linop.h
# SmurffCpp/Utils/TensorUtils.h
# SmurffCpp/Utils/omp_util.cpp
# SmurffCpp/Utils/StringUtils.h
# SmurffCpp/Utils/TruncNorm.cpp
# SmurffCpp/Utils/RootFile.h
# SmurffCpp/Utils/StringUtils.cpp
# SmurffCpp/Utils/RootFile.cpp
# SmurffCpp/Utils/TensorUtils.cpp
# SmurffCpp/Utils/TruncNorm.h
# SmurffCpp/Utils/MatrixUtils.cpp
# SmurffCpp/Utils/MatrixUtils.h
# SmurffCpp/Utils/Distribution.h
# SmurffCpp/Utils/StepFile.h
# SmurffCpp/Utils/InvNormCdf.h
# SmurffCpp/Utils/StepFile.cpp
# SmurffCpp/Utils/InvNormCdf.cpp
# SmurffCpp/ResultItem.h
# SmurffCpp/SideInfo/DenseSideInfo.cpp
# SmurffCpp/SideInfo/DenseSideInfo.h
# SmurffCpp/SideInfo/ISideInfo.h
# SmurffCpp/SideInfo/SparseSideInfo.cpp
# SmurffCpp/SideInfo/SparseSideInfo.h
# SmurffCpp/StatusItem.cpp
# SmurffCpp/Predict/PredictSession.h
# SmurffCpp/Predict/PredictSession.cpp
# SmurffCpp/Sessions/SessionFactory.cpp
# SmurffCpp/Sessions/SessionFactory.h
# SmurffCpp/Sessions/CmdSession.h
# SmurffCpp/Sessions/PythonSession.cpp
# SmurffCpp/Sessions/Session.h
# SmurffCpp/Sessions/Session.cpp
# SmurffCpp/Sessions/ISession.h
# SmurffCpp/Sessions/PythonSession.h
# SmurffCpp/Sessions/CmdSession.cpp
# SmurffCpp/Sessions/ISession.cpp
# SmurffCpp/Model.h
# SmurffCpp/Model.cpp
# SmurffCpp/result.cpp
# SmurffCpp/DataTensors/TensorData.h
# SmurffCpp/DataTensors/TensorDataFactory.h
# SmurffCpp/DataTensors/TensorDataFactory.cpp
# SmurffCpp/DataTensors/TensorData.cpp
# SmurffCpp/DataTensors/SparseMode.cpp
# SmurffCpp/DataTensors/SparseMode.h
# SmurffCpp/StatusItem.h
# SmurffCpp/IO/*.cpp
# SmurffCpp/IO/*.h
# Tests/expected_results/*.h
# "



FILES="SmurffCpp/DataMatrices/FullMatrixData.hpp
SmurffCpp/DataMatrices/MatrixDataTempl.hpp
SmurffCpp/ConstVMatrixExprIterator.hpp
SmurffCpp/Utils/PVec.hpp
SmurffCpp/Utils/ThreadVector.hpp
SmurffCpp/ConstVMatrixIterator.hpp
SmurffCpp/VMatrixIterator.hpp
SmurffCpp/VMatrixExprIterator.hpp
Tests/catch.hpp"

for F in $FILES
do
	gsed -i -e "
	s/\bdouble\b/float/g;
	s/Eigen::MatrixXd/Eigen::MatrixXf/g;
	s/Eigen::VectorXd/Eigen::VectorXf/g;
	s/Eigen::ArrayXd/Eigen::ArrayXf/g;
	s/Eigen::ArrayXXd/Eigen::ArrayXXf/g;" $F

done

