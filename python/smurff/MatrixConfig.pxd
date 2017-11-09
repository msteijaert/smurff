from libc.stdint cimport *
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from NoiseConfig cimport NoiseConfig

cdef extern from "<SmurffCpp/Configs/MatrixConfig.h>" namespace "smurff":
    cdef cppclass MatrixConfig:
        #
        # Dense double matrix constructos
        #
        #MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
        #             const std::vector<double>& values,
        #             const NoiseConfig& noiseConfig);
        #
        #MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
        #             std::vector<double>&& values,
        #             const NoiseConfig& noiseConfig);
        #
        #MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
        #             std::shared_ptr<std::vector<double> > values,
        #             const NoiseConfig& noiseConfig);

        #
        # Sparse double matrix constructors
        #
        MatrixConfig(uint64_t nrow, uint64_t ncol,
                     const vector[uint32_t]& rows, const vector[uint32_t]& cols, const vector[double]& values,
                     const NoiseConfig& noiseConfig) except +

        MatrixConfig(uint64_t nrow, uint64_t ncol,
                     vector[uint32_t]&& rows, vector[uint32_t]&& cols, vector[double]&& values,
                     const NoiseConfig& noiseConfig) except +

        MatrixConfig(uint64_t nrow, uint64_t ncol,
                     shared_ptr[vector[uint32_t]] rows, shared_ptr[vector[uint32_t]] cols, shared_ptr[vector[double]] values,
                     const NoiseConfig& noiseConfig) except +

        #
        # Sparse binary matrix constructors
        #
        #MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
        #             const std::vector<std::uint32_t>& rows, const std::vector<std::uint32_t>& cols,
        #             const NoiseConfig& noiseConfig);
        #
        #MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
        #             std::vector<std::uint32_t>&& rows, std::vector<std::uint32_t>&& cols,
        #             const NoiseConfig& noiseConfig);
        #
        #MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
        #             std::shared_ptr<std::vector<std::uint32_t> > rows, std::shared_ptr<std::vector<std::uint32_t> > cols,
        #             const NoiseConfig& noiseConfig);

        #
        # Constructors for constructing matrix as a tensor
        #
        #MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
        #             const std::vector<std::uint32_t>& columns, const std::vector<double>& values,
        #             const NoiseConfig& noiseConfig);
        #
        #MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
        #             std::vector<std::uint32_t>&& columns, std::vector<double>&& values,
        #             const NoiseConfig& noiseConfig);
        #
        #MatrixConfig(std::uint64_t nrow, std::uint64_t ncol,
        #             std::shared_ptr<std::vector<std::uint32_t> > columns, std::shared_ptr<std::vector<double> > values,
        #             const NoiseConfig& noiseConfig);
