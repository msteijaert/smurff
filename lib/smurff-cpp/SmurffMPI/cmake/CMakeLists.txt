if(${MPI_C_FOUND})

#SETUP PROJECT
set (PROJECT mpi_smurff)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES "../MPISession.h"
                        "../MPIMacauPrior.hpp"
                        "../MPIPriorFactory.h"
                        "../../SmurffCpp/Sessions/CmdSession.h")
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../mpi_smurff.cpp"
                        "../MPISession.cpp"
                        "../MPIPriorFactory.cpp"
                        "../../SmurffCpp/Sessions/CmdSession.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Utils")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
                                  ${Boost_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT}
                                  ${MPI_LIBRARIES})

set_target_properties(${PROJECT} PROPERTIES COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
set_target_properties(${PROJECT} PROPERTIES LINK_FLAGS "${MPI_C_LINK_FLAGS}")

#SETUP INCLUDES
include_directories(..)
include_directories(../..)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${MPI_C_INCLUDE_PATH})
include_directories(${Boost_INCLUDE_DIRS})

endif()
