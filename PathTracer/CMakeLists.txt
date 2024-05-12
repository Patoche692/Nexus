set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
set(ASSIMP_BUILD_SAMPLES OFF)
set(ASSIMP_BUILD_TESTS OFF)
set(ASSIMP_BUILD_DOCS OFF)
set(ASSIMP_INSTALL OFF)
add_subdirectory(vendor/assimp)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)
add_subdirectory(vendor/GLFW)

set(BUILD_UTILS OFF)
set(GLEW_INSTALL OFF)
add_subdirectory(vendor/GLEW/build/cmake)

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)
set(GLM_BUILD_TESTS OFF)
add_subdirectory(vendor/glm)

add_subdirectory(vendor/imgui)

file(GLOB_RECURSE src_files src/*.h src/*.cpp src/*.cuh src/*.cu)

add_executable(Nexus ${src_files})

target_link_libraries(Nexus
	glfw
	glew
	assimp
	imgui
)

target_include_directories( Nexus
	PUBLIC
	src
	vendor/assimp/include
	vendor/GLFW/include
	vendor/GLEW/include
	vendor/glm
	vendor/imgui
	vendor/imgui/backends
	vendor/stb
)
