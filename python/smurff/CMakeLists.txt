#SETUP PROJECT
set (PROJECT cy_smurff)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include( UseCython )

find_package(NumPy REQUIRED)
message(STATUS ${NUMPY_INCLUDE_DIR})

include_directories(${NUMPY_INCLUDE_DIR})
include_directories(../../lib/smurff-cpp)
include_directories(.)

# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
    ${PROJECT}.pyx
    PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( ${PROJECT} ${PROJECT}.pyx )

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
                            ${ALGEBRA_LIBS}
                            ${OpenMP_CXX_LIBRARIES}
                            ${CMAKE_THREAD_LIBS_INIT})

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${SCRIPT_DIR}/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

