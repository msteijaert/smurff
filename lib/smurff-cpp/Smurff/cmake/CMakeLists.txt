#SETUP PROJECT
set (PROJECT smurff)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB SOURCE_FILES "../smurff.cpp")

source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Utils")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
                                  ${Boost_LIBRARIES}
                                  ${BOOST_RANDOM_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(../..)
include_directories(${SMURFF_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${BOOST_RANDOM_INCLUDE_DIRS})
