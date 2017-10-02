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
  if(${MPI_CXX_FOUND})
    message(STATUS "MPI found")
  else()
    message(STATUS "MPI not found")
  endif()
   
endmacro(configure_mpi)

macro(configure_blas)
  message ("Dependency check for blas...")
  set(BLA_VENDOR "Generic")
  find_package( BLAS REQUIRED )
  message(STATUS BLAS: ${BLAS_LIBRARIES} )

  find_path(BLAS_INCLUDE_DIRS cblas.h
  /usr/include
  /usr/local/include
  $ENV{BLAS_HOME}/include)
  message(STATUS ${BLAS_INCLUDE_DIRS})

endmacro(configure_blas)

macro(configure_lapack)
  message ("Dependency check for lapack...")
  set(BLA_VENDOR "Generic")
  find_package(LAPACK REQUIRED)
  message(STATUS LAPACK: ${LAPACK_LIBRARIES})

endmacro(configure_lapack)

macro(configure_openblas)
  message ("Dependency check for openblas...")
  set(BLA_VENDOR "OpenBLAS")
  find_package( BLAS REQUIRED )
  message(STATUS BLAS: ${BLAS_LIBRARIES} )
 
endmacro(configure_openblas)

macro(configure_libfastsparse)
  message ("Dependency check for libfastsparse...")

  #SUBMODULE INSTALLATION
  set(LIBFASTSPARSE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../libfastsparse)
  message(STATUS LIBFASTSPARSE: ${LIBFASTSPARSE_INCLUDE_DIR})
   
endmacro(configure_libfastsparse)

macro(configure_eigen)
  message ("Dependency check for eigen...")

  # EXTERNAL INSTALLATION
  #find_package(Eigen3 REQUIRED)
  #message(STATUS EIGEN3: ${EIGEN3_INCLUDE_DIR})

  #SUBMODULE INSTALLATION
  set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../eigen3)
  message(STATUS EIGEN3: ${EIGEN3_INCLUDE_DIR})

endmacro(configure_eigen)




