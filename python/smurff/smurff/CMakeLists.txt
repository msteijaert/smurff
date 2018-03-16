# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
    wrapper.pyx
    PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( wrapper wrapper.pyx )

#LINK LIBRARIES
target_link_libraries (wrapper smurff-cpp
                            ${ALGEBRA_LIBS}
                            ${OpenMP_CXX_LIBRARIES}
                            ${CMAKE_THREAD_LIBS_INIT})

