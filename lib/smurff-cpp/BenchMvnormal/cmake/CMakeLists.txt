#SETUP PROJECT
set (PROJECT bench_mvnormal)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES ""
                        )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../bench_mvnormal.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Executables")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_output")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

set_target_properties(${PROJECT} PROPERTIES
		COMPILE_DEFINITIONS "${DEFINITIONS}")

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
				  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(../../SmurffCpp) # this is a temporarely solution. headers in SmurffCpp should be fixed first to start with SmurffCpp
include_directories(${EIGEN3_INCLUDE_DIR})
