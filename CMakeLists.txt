cmake_minimum_required(VERSION 3.0)

project(wildpose)


add_executable(wildpose src/main.cpp src/lvx_file.cpp include/xiAPI/xiAPI_tiff.cpp)

target_link_libraries(wildpose 
	m3api
	tiff
	wildpose_static)

include_directories(include
	include/sdk_core
	include/xiAPI
	include/libtiff)

add_subdirectory(include/libtiff)
add_subdirectory(include/sdk_core)

install(TARGETS wildpose RUNTIME DESTINATION bin)
