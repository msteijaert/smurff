cmake_minimum_required (VERSION 3.6.0)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

set (PROJECT smurff)
project (${PROJECT})

MESSAGE("smurff")

# OPTIONS

OPTION(ENABLE_BLAS "Enable BLAS library" OFF)
OPTION(ENABLE_LAPACK "Enable LAPACK library" OFF)
OPTION(ENABLE_OPENBLAS "Enable OPENBLAS library" OFF)

OPTION(ENABLE_PROFILING "Enable performance counters" OFF)

OPTION(ENABLE_VERBOSE_COMPILER_LOG "Enable verbose compilation log" OFF)
OPTION(ENABLE_VERBOSE_LINKER_LOG "Enable verbose linker log" OFF)

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
      #execute_process(COMMAND /usr/bin/xcode-select -print-path
      #OUTPUT_VARIABLE XCODE_DEVELOPER_DIR
      #OUTPUT_STRIP_TRAILING_WHITESPACE)

      # Set the CMAKE_OSX_SYSROOT to the latest SDK found
      #execute_process(COMMAND /usr/bin/xcrun -sdk macosx --show-sdk-path
      #OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
      #OUTPUT_STRIP_TRAILING_WHITESPACE)

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
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas -fPIC")

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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../SmurffCpp/Version.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/../SmurffCpp/Version.cpp)

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
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

    #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1")
    #set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1")
  elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Release build.")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -g -fstrict-aliasing")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffast-math -g -fstrict-aliasing")
  else()
    message(STATUS "Unknown build type...")
  endif()
endif()

# CONFIGURE LOGGING

message(STATUS "Initializing logging options...")

if (UNIX)
    #set verbose output of cmake
    if (${ENABLE_VERBOSE_COMPILER_LOG})
        set(CMAKE_VERBOSE_MAKEFILE ON)
    endif()

    #enable make verbose linker log
    if (${ENABLE_VERBOSE_LINKER_LOG})
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
configure_openmp()
configure_eigen()

configure_argp()

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

if(${ENABLE_PROFILING})
    add_definitions(-DPROFILING)
endif()


# support for running "make test" (or alike)

enable_testing()

# unittests
add_test( unittests tests )

if(${EXTERNAL_DATA})
    file(DOWNLOAD  http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm 
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-346targets.mm
            EXPECTED_HASH SHA256=10c3e1f989a7a415a585a175ed59eeaa33eff66272d47580374f26342cddaa88
        )

    file(DOWNLOAD http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-compound-feat.mm
            EXPECTED_HASH SHA256=f9fe0d296272ef26872409be6991200dbf4884b0cf6c96af8892abfd2b55e3bc
        )

    file(DOWNLOAD http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compounds.csv
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-compounds.csv
            EXPECTED_HASH SHA256=e8f045a67ee149c6100684e07920036de72583366596eb5748a79be6e3b96f7c
        )

    file(DOWNLOAD http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-proteins-uniprot.csv
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-proteins-uniprot.csv
            EXPECTED_HASH SHA256=224b1b44abcab8448b023874f4676af30d64fe651754144f9cbdc67853b76ea8
        )

    add_test(chembl_bmpf smurff --num-latent=4 --burnin=2 --nsamples=2
        --train ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-346targets.mm
        )

    add_test(chembl_macau smurff --num-latent=4 --burnin=2 --nsamples=2
        --train ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-346targets.mm
        --row-features ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-compound-feat.mm
        )
endif()



# CONFIGURE TARGETS

message(STATUS "Creating folder structure...")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# libraries

add_subdirectory (../SmurffCpp/cmake libraries/SmurffCpp)

# utils

add_subdirectory (../Smurff/cmake utils/Smurff)
add_subdirectory (../SmurffMPI/cmake utils/SmurffMPI)

# tests

add_subdirectory (../Tests/cmake tests/Tests)
add_subdirectory (../BenchMvnormal/cmake tests/BenchMvnormal)
add_subdirectory (../BenchNrandn/cmake tests/BenchNrandn)
add_subdirectory (../BenchWishart/cmake tests/BenchWishart)

