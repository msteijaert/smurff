#SETUP PROJECT
set (PROJECT tests)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES "../catch.hpp"
                        )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../tests.cpp"
			"../TestsCentring.cpp"
                        "../TestsMatrixIO.cpp"
                        "../TestsTensorIO.cpp"
                        "../TestsMatrixConfig.cpp"
                        "../TestsTensorConfig.cpp"
     	  	   	         "../TestsPVec.cpp"
                        "../TestsTensor.cpp"
                        "../TestsMatrixUtils.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Tests")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
install(TARGETS ${PROJECT} RUNTIME DESTINATION libexec)

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
                                  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(../..)
include_directories(${EIGEN3_INCLUDE_DIR})
