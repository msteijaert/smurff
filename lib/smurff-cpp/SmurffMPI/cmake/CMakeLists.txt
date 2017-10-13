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
