cmake_minimum_required (VERSION 3.6.0)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

set (PROJECT smurff)
project (${PROJECT})

MESSAGE("smurff")

# OPTIONS

OPTION(ENABLE_BLAS "Enable BLAS library" OFF)
OPTION(ENABLE_LAPACK "Enable LAPACK library" OFF)
OPTION(ENABLE_OPENBLAS "Enable OPENBLAS library" OFF)

# INIT CMAKE

message("Initializing cmake ...")
MESSAGE( STATUS "CMAKE_VERSION: " ${CMAKE_VERSION} )
# the short system name, e.g. "Linux", "FreeBSD" or "Windows"
MESSAGE( STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME} )
# only the version part of CMAKE_SYSTEM
MESSAGE( STATUS "CMAKE_SYSTEM_VERSION: " ${CMAKE_SYSTEM_VERSION} )
# the processor name (e.g. "Intel(R) Pentium(R) M processor 2.00GHz")
MESSAGE( STATUS "CMAKE_SYSTEM_PROCESSOR: " ${CMAKE_SYSTEM_PROCESSOR} )

# FETCH OS

message("Fetching os and compiler ...")

#Fetch OS state and mixture thereof, like cygwin(unix+windows), OSX(unix+apple)
if (UNIX)
  MESSAGE( STATUS "OS: Unix system")
endif()
if (WIN32)
  MESSAGE( STATUS "OS: Windows system" )
endif()
if (APPLE)
  MESSAGE( STATUS "OS: Apple system")
endif()
if (MINGW)
  MESSAGE( STATUS "Compiler is a MINGW variant")
endif()
if (CYGWIN)
  MESSAGE( STATUS "Compiler is a CYGWIN variant")
endif()
if (BORLAND)
  MESSAGE( STATUS "Compiler is a BORLAND variant")
endif()
if (CMAKE_COMPILER_IS_GNUCC)
  MESSAGE( STATUS "Compiler is a gcc variant")
endif()

# FETCH COMPILER

if (MSVC OR MSVC_IDE OR MSVC_VERSION OR MSVC60 OR MSVC70 OR MSVC71 OR MSVC80 OR MSVC90 OR MSVC10 OR MSVC11 OR MSVC12)
   message( STATUS "Compiler is a Microsoft variant")
   message( STATUS "MSVC: " ${MSVC} )
   message( STATUS "MSVC_IDE: " ${MSVC_IDE} )
   message( STATUS "MSVC_VERSION: " ${MSVC_VERSION} )
   message( STATUS "MSVC60: " ${MSVC60} )
   message( STATUS "MSVC70: " ${MSVC70} )
   message( STATUS "MSVC71: " ${MSVC71} )
   message( STATUS "MSVC80: " ${MSVC80} )
   message( STATUS "MSVC90: " ${MSVC90} )
   message( STATUS "MSVC10: " ${MSVC10} )
   message( STATUS "MSVC11: " ${MSVC11} )
   message( STATUS "MSVC12: " ${MSVC12} )

   add_definitions (/W3)

   #hide some irritating c++ unsafe function warnings
   add_definitions(-D_SCL_SECURE_NO_WARNINGS)

   set (CMAKE_EXE_LINKER_FLAGS "/SAFESEH:NO")

elseif (CMAKE_COMPILER_IS_GNUCXX OR APPLE)
   if(APPLE)
      # Set developer directory
      execute_process(COMMAND /usr/bin/xcode-select -print-path
      OUTPUT_VARIABLE XCODE_DEVELOPER_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)

      # Set the CMAKE_OSX_SYSROOT to the latest SDK found
      execute_process(COMMAND /usr/bin/xcrun -sdk macosx --show-sdk-path
      OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

      # Locate gcc
      #execute_process(COMMAND /usr/bin/xcrun -sdk macosx -find gcc
      #OUTPUT_VARIABLE CMAKE_C_COMPILER
      #OUTPUT_STRIP_TRAILING_WHITESPACE)

      # Locate g++
      #execute_process(COMMAND /usr/bin/xcrun -sdk macosx -find g++
      #OUTPUT_VARIABLE CMAKE_CXX_COMPILER
      #OUTPUT_STRIP_TRAILING_WHITESPACE)

      message(STATUS "   -- OSX: Using iOS SDK: ${CMAKE_OSX_SYSROOT}")
      message(STATUS "   -- OSX: System root directory found at: ${CMAKE_OSX_SYSROOT}")
      message(STATUS "   -- OSX: Xcode developer directory found at: ${XCODE_DEVELOPER_DIR}")
      message(STATUS "   -- OSX: gcc found at: ${CMAKE_C_COMPILER}")
      message(STATUS "   -- OSX: g++ found at: ${CMAKE_CXX_COMPILER}")

      add_definitions(-DMAC_OSX)
   endif()

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-unknown-pragmas -Wnon-virtual-dtor -fPIC")

else()
   message (FATAL_ERROR "Using untested/unconfigured compiler, only Microsoft and g++ toolchain has been tested/established.")
   return()
endif()

# INIT VERSIONING

message("Initializing versioning...")

#get GIT commit count we can later use as patch level
execute_process(
  COMMAND git rev-list HEAD --count
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_COUNT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "GIT commit count ${GIT_COMMIT_COUNT}")

#get GIT commit SHA
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "GIT commit SHA ${GIT_COMMIT}")

#get GIT branch
execute_process(
  COMMAND git name-rev --name-only HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "GIT branch ${GIT_BRANCH}")

#get GIT describe
execute_process(
  COMMAND git describe
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE SMURFF_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "GIT describe ${SMURFF_VERSION}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../Version.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/../Version.cpp)

# CONFIGURE BUILD TYPE

message(STATUS "Analyzing build type...")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

if(NOT MSVC)
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build.")
    add_definitions(-D_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -g")
  elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Release build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -march=native")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -march=native -flto")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -g -fstrict-aliasing -flto")
  else()
    message(STATUS "Unknown build type...")
  endif()
endif()

# CONFIGURE LOGGING

message(STATUS "Initializing logging options...")

set (CV ON)

if (UNIX)
    #set verbose output of cmake
    if (CV)
        set(CMAKE_VERBOSE_MAKEFILE ON)
    endif()

    #enable make verbose linker log
    if (LV)
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -###" )
    endif()
endif()

# CONFIGURE DEPENDENCIES

message("Checking build dependencies...")

#set additional search paths for cmake scripts
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/")

include(scripts/DependenciesConfig.cmake)
configure_pthreads()
configure_mpi()
configure_eigen()

set(ALGEBRA_LIB_FOUND OFF)

if(${ENABLE_BLAS})
  if(${ENABLE_LAPACK} OR ${ENABLE_OPENBLAS})
  message(FATAL_ERROR "Multiple algebra libraries specified")
  endif()
  configure_blas()
  if(${BLAS_FOUND})
    set(ALGEBRA_LIBS ${BLAS_LIBRARIES})
    set(ALGEBRA_LIB_NAME BLAS)
    set(ALGEBRA_LIB_FOUND ON)
  endif()
endif()

if(${ENABLE_LAPACK})
  if(${ENABLE_BLAS} OR ${ENABLE_OPENBLAS})
  message(FATAL_ERROR "Multiple algebra libraries specified")
  endif()
  configure_lapack()
  if(${LAPACK_FOUND})
    set(ALGEBRA_LIBS ${LAPACK_LIBRARIES})
    set(ALGEBRA_LIB_NAME LAPACK)
    set(ALGEBRA_LIB_FOUND ON)
  endif()
endif()

if(${ENABLE_OPENBLAS})
  if(${ENABLE_BLAS} OR ${ENABLE_LAPACK})
  message(FATAL_ERROR "Multiple algebra libraries specified")
  endif()
  configure_openblas()
  if(${BLAS_FOUND})
    set(ALGEBRA_LIBS ${BLAS_LIBRARIES})
    set(ALGEBRA_LIB_NAME OPENBLAS)
    set(ALGEBRA_LIB_FOUND ON)
  endif()
endif()

if(${ALGEBRA_LIB_FOUND})
  message(STATUS "Algebra lib name: " ${ALGEBRA_LIB_NAME})
  message(STATUS ${ALGEBRA_LIBS})
else()
  message(FATAL_ERROR "Algebra library not specified")
endif()


# CONFIGURE TARGETS

message(STATUS "Creating folder structure...")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

############## setup smurff-cpp ##############

#SETUP PROJECT
set (PROJECT smurff-cpp)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES "../Version.h"
                        "../truncnorm.h"
                        "../model.h"
                        "../result.h"
			"../SparseFeat.h"
			"../SparseDoubleFeat.h"

                        "../Utils/InvNormCdf.h"
                        "../Utils/Distribution.h"
                        "../Utils/MatrixUtils.h"
                        "../Utils/PVec.hpp"
                        "../Utils/chol.h"
                        "../Utils/counters.h"
                        "../Utils/linop.h"
			"../Utils/omp_util.h"
			"../Utils/utils.h"

                        "../IO/ini.h"
                        "../IO/INIReader.h"
                        "../IO/MatrixIO.h"
                        "../IO/TensorIO.h"

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

                        "../Priors/GlobalPrior.h"
                        "../Priors/ILatentPrior.h"
                        "../Priors/NormalPrior.h"
                        "../Priors/SpikeAndSlabPrior.h"
                        "../Priors/MacauPrior.hpp"
                        "../Priors/MacauOnePrior.hpp"
                        
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

			"../Sessions/BaseSession.h"
			"../Sessions/Session.h")
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../Version.cpp"
                        "../truncnorm.cpp"
                        "../model.cpp"
                        "../result.cpp"
			"../SparseFeat.cpp"
			"../SparseDoubleFeat.cpp"

                        "../Utils/InvNormCdf.cpp"
                        "../Utils/Distribution.cpp"
                        "../Utils/MatrixUtils.cpp"
                        "../Utils/chol.cpp"
                        "../Utils/counters.cpp"
                        "../Utils/linop.cpp"

                        "../IO/ini.c"
                        "../IO/INIReader.cpp"
                        "../IO/MatrixIO.cpp"
                        "../IO/TensorIO.cpp"

			"../Configs/Config.cpp"
                        "../Configs/MatrixConfig.cpp"
                        "../Configs/TensorConfig.cpp"
                        "../Configs/NoiseConfig.cpp"

                        "../Noises/AdaptiveGaussianNoise.cpp"
                        "../Noises/FixedGaussianNoise.cpp"
                        "../Noises/ProbitNoise.cpp"
                        "../Noises/Noiseless.cpp"
                        "../Noises/UnusedNoise.cpp"

                        "../Priors/GlobalPrior.cpp"
                        "../Priors/ILatentPrior.cpp"
                        "../Priors/NormalPrior.cpp"
                        "../Priors/SpikeAndSlabPrior.cpp"
                        "../Priors/MacauPrior.cpp"

                        "../DataMatrices/Data.cpp"
                        "../DataMatrices/DenseMatrixData.cpp"
                        "../DataMatrices/MatricesData.cpp"
                        "../DataMatrices/MatrixData.cpp"
                        "../DataMatrices/MatrixDataTempl.cpp"
                        "../DataMatrices/ScarceBinaryMatrixData.cpp"
                        "../DataMatrices/ScarceMatrixData.cpp"
                        "../DataMatrices/SparseMatrixData.cpp"
			"../DataMatrices/MatrixDataFactory.cpp"

			"../Sessions/BaseSession.cpp"
			"../Sessions/Session.cpp")
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_library (${PROJECT} STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Libraries")

#SETUP INCLUDES
include_directories(../)
include_directories(${EIGEN3_INCLUDE_DIR})

############## setup smurff ##############

#SETUP PROJECT
set (PROJECT smurff)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES "../Sessions/CmdSession.h"
                        )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../Sessions/CmdSession.cpp"
                        "../smurff.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Executables")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_output")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
                                  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(${EIGEN3_INCLUDE_DIR})

############## setup tests ##############

#SETUP PROJECT
set (PROJECT tests)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES "../Tests/catch.hpp"
                        )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../Tests/tests.cpp"
			"../Tests/TestsCentring.cpp"
                        "../Tests/TestsMatrixIO.cpp"
                        "../Tests/TestsTensorIO.cpp"
                        "../Tests/TestsMatrixConfig.cpp"
                        "../Tests/TestsTensorConfig.cpp"
     	  	   	"../Tests/TestsPVec.cpp"
                        "../Tests/TestsTensor.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Executables")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_output")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
                                  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(${EIGEN3_INCLUDE_DIR})

############## setup mpi_smurff ##############

if(${MPI_CXX_FOUND})

#SETUP PROJECT
set (PROJECT mpi_smurff)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES "../Sessions/MPISession.h"
                        "../Priors/MPIMacauPrior.hpp"
                        )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../Sessions/MPISession.cpp"
                        "../mpi_smurff.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Executables")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_output")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
                                  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(${EIGEN3_INCLUDE_DIR})

endif()

############## setup bench_nrandn ##############

#SETUP PROJECT
set (PROJECT bench_nrandn)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES ""
                        )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../bench_nrandn.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Executables")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_output")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

set_target_properties(${PROJECT} PROPERTIES
		COMPILE_DEFINITIONS "${DEFINITIONS}")

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
				  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(${EIGEN3_INCLUDE_DIR})

############## setup bench_wishart ##############

#SETUP PROJECT
set (PROJECT bench_wishart)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES ""
                        )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../bench_wishart.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Executables")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_output")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

set_target_properties(${PROJECT} PROPERTIES
		COMPILE_DEFINITIONS "${DEFINITIONS}")

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
				  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(${EIGEN3_INCLUDE_DIR})

############## setup bench_mvnormal ##############

#SETUP PROJECT
set (PROJECT bench_mvnormal)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES ""
                        )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../bench_mvnormal.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Executables")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_output")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

set_target_properties(${PROJECT} PROPERTIES
		COMPILE_DEFINITIONS "${DEFINITIONS}")

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
				  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(${EIGEN3_INCLUDE_DIR})

