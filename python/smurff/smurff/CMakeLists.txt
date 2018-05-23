set (CYTHON_FLAGS "--directive" "embedsignature=True")

add_cython_target(wrapper CXX)
add_library(wrapper MODULE ${wrapper})
python_extension_module(wrapper)

#LINK LIBRARIES
target_link_libraries (wrapper smurff-cpp
                            ${ALGEBRA_LIBS}
                            ${OpenMP_CXX_LIBRARIES}
                            ${CMAKE_THREAD_LIBS_INIT})

