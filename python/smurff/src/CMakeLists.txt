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

