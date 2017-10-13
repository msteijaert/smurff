#SETUP PROJECT
set (PROJECT tests)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES "../Tests/catch.hpp"
                        )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../Tests/tests.cpp"
			"../Tests/TestsCentring.cpp"
                        "../Tests/TestsMatrixIO.cpp"
                        "../Tests/TestsTensorIO.cpp"
                        "../Tests/TestsMatrixConfig.cpp"
                        "../Tests/TestsTensorConfig.cpp"
     	  	   	"../Tests/TestsPVec.cpp"
                        "../Tests/TestsTensor.cpp"
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
