cmake_minimum_required(VERSION 2.8)

project(kfusion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_subdirectory(dep/TooN)
add_subdirectory(dep/libcvd)

include_directories(
	${CMAKE_BINARY_DIR}/include
	dep/libcvd
	)


find_package(CUDA)
find_package(Freenect)
find_package(OpenGL)

if(CUDA_FOUND AND FREENECT_FOUND AND OPENGL_FOUND)
	
	include_directories(${FREENECT_INCLUDE_PATHS} ${OPENGL_INCLUDE_DIRS})

	set(kfusion_cuda_srcs
		helpers.cu
		helpers.h
		kfusion.cu
		kfusion.h
		cutil_math.h
		)
	
	CUDA_ADD_EXECUTABLE(kfusion_test
		${kfusion_cuda_srcs}
		kinect.cpp
		perfstats.h
		)
endif()
	