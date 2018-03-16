macro(configure_pthreads)
    message ("Dependency check for pthreads multi-threading support...")

    if(UNIX)
        find_package(Threads REQUIRED)
        if(Threads_FOUND)
            message(STATUS "Found threading library")
            if(CMAKE_USE_PTHREADS_INIT)
                message(STATUS "Found pthreads " ${CMAKE_THREAD_LIBS_INIT})
            else()
                message(STATUS "Pthreads not found")
            endif()
        else()
            message(STATUS "Threading library not found")
        endif()
    else()
       message(STATUS "Not required on windows")
    endif()
endmacro(configure_pthreads)

macro(configure_mpi)
  message ("Dependency check for mpi...")

  find_package(MPI)
  if(${MPI_C_FOUND})
    message(STATUS "MPI found")
  else()
    message(STATUS "MPI not found")
  endif()
   
endmacro(configure_mpi)

macro(configure_openmp)
  message ("Dependency check for OpenMP")

  find_package(OpenMP)
  if(${OPENMP_FOUND})
      message(STATUS "OpenMP found")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
      message(STATUS "OpenMP_CXX_LIB_NAMES ${OpenMP_CXX_LIB_NAMES}")
      message(STATUS "OpenMP_CXX_LIBRARY ${OpenMP_CXX_LIBRARY}")
      message(STATUS "OpenMP_CXX_LIBRARIES ${OpenMP_CXX_LIBRARIES}")
  else()
      message(STATUS "OpenMP not found")
  endif()
   
endmacro(configure_openmp)

macro(configure_lapack)
  message ("Dependency check for lapack...")
  find_package(LAPACK REQUIRED)
  message(STATUS LAPACK: ${LAPACK_LIBRARIES})
endmacro(configure_lapack)

macro(configure_openblas)
  message ("Dependency check for openblas...")
  
  if(MSVC)
  set(BLAS_LIBRARIES  $ENV{BLAS_LIBRARIES})
  set(BLAS_INCLUDES $ENV{BLAS_INCLUDES})
  set(BLAS_FOUND ON)
  else()
  set(BLA_VENDOR "OpenBLAS")
  find_package( BLAS REQUIRED )
  endif()
  
  message(STATUS BLAS: ${BLAS_LIBRARIES} )
 
endmacro(configure_openblas)

macro(configure_mkl)
  message ("Dependency check for MKL...")
  set(BLA_VENDOR "Intel")
  find_package( LAPACK REQUIRED )

  # since we mix OpenMP and mkl we need to link this
  if(${OPENMP_FOUND})
      list(FIND OpenMP_CXX_LIB_NAMES "gomp" GNU_OPENMP)
      list(FIND OpenMP_CXX_LIB_NAMES "iomp5" INTEL_OPENMP)
      if(NOT GNU_OPENMP EQUAL -1)
          find_library(INTEL_THREAD_LIBRARY "mkl_gnu_thread"  HINTS ENV LD_LIBRARY_PATH)
      elseif(NOT INTEL_OPENMP EQUAL -1)
          find_library(INTEL_THREAD_LIBRARY "mkl_intel_thread"  HINTS ENV LD_LIBRARY_PATH)
      else()
          message(ERROR "Unknown threading library ${OpenMP_CXX_LIB_NAMES}")
      endif()
      set(LAPACK_LIBRARIES ${LAPACK_LIBRARIES} ${INTEL_THREAD_LIBRARY})
  endif()
  
  message(STATUS MKL: ${LAPACK_LIBRARIES} )
endmacro(configure_mkl)

macro(configure_eigen)
  message ("Dependency check for eigen...")
  
  if(DEFINED ENV{EIGEN3_INCLUDE_DIR})
  SET(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
  else()
  find_package(Eigen3 REQUIRED)
  endif()
  
  message(STATUS EIGEN3: ${EIGEN3_INCLUDE_DIR})

  add_definitions(-DEIGEN_DONT_PARALLELIZE)
endmacro(configure_eigen)

macro(configure_boost)
  if(${ENABLE_BOOST})
      message ("Dependency check for boost...")
      
      set (Boost_USE_STATIC_LIBS ON)
      set (Boost_USE_MULTITHREADED ON)

      # find boost random library - optional

      if(${ENABLE_BOOST_RANDOM})
        set (BOOST_COMPONENTS random)

        FIND_PACKAGE(Boost COMPONENTS ${BOOST_COMPONENTS})

        if(Boost_FOUND)
            message("Found Boost random library")
            message("Found Boost_VERSION: ${Boost_VERSION}")
            message("Found Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
            message("Found Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")

            set(BOOST_RANDOM_LIBRARIES ${Boost_LIBRARIES})
            set(BOOST_RANDOM_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})

            add_definitions(-DUSE_BOOST_RANDOM)
            message("Boost minor: ${Boost_MINOR_VERSION}")
            if ((${Boost_MINOR_VERSION} LESS "60") AND (${Boost_MINOR_VERSION} GREATER_EQUAL "54"))
                add_definitions(-DTEST_RANDOM)
                message("Enabling test cases that depend on Boost random 1.5x.y")
            endif()
        else()
            message("Boost random library is not found")
        endif()
      endif()

      #find boost program_options library - required

      set (BOOST_COMPONENTS system 
                            program_options)

      FIND_PACKAGE(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)
  endif()

  if(Boost_FOUND)
      message("Found Boost_VERSION: ${Boost_VERSION}")
      message("Found Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
      message("Found Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
      add_definitions(-DHAVE_BOOST)
  else()
      message("Boost library is not found")
  endif()
endmacro(configure_boost)

macro(configure_python)
    find_package(PythonInterp) 

    if(PYTHONINTERP_FOUND)
        find_package(NumPy REQUIRED)
        find_package( Cython REQUIRED )
   endif()
endmacro(configure_python)
