cmake_minimum_required(VERSION 3.22)

#if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
#	message(FATAL_ERROR "In-source builds are not allowed. Please make a new build directory and run cmake from there. You may want to remove CMakeCache.txt")
#endif()

project("Nexus Renderer")

add_subdirectory(Nexus)

file(COPY ${CMAKE_SOURCE_DIR}/Nexus/assets DESTINATION ${CMAKE_BINARY_DIR}/Nexus/)

