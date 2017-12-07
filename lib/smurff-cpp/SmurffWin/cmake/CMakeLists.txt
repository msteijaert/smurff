#SETUP PROJECT
set (PROJECT win_smurff)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB HEADER_FILES "")
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../win_smurff.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#FIND BOOST
set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_MULTITHREADED ON)

set (BOOST_COMPONENTS system 
                      program_options)

FIND_PACKAGE(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)

if(Boost_FOUND)
message("Found Boost_VERSION: ${Boost_VERSION}")
message("Found Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message("Found Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
else()
message("Boost library is not found")
endif()

#SETUP OUTPUT
add_executable (${PROJECT} ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Utils")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
install(TARGETS ${PROJECT} RUNTIME DESTINATION bin)

#LINK LIBRARIES
target_link_libraries (${PROJECT} smurff-cpp
                                  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT}
                                  ${Boost_LIBRARIES})

#SETUP INCLUDES
include_directories(../)
include_directories(../..)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
