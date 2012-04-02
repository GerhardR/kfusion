cmake_minimum_required(VERSION 2.8)

project(kfusion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include_directories(
	${CMAKE_BINARY_DIR}/include
	thirdparty
)

find_package(CUDA REQUIRED)
find_package(OpenGL)
find_package(Freenect)
find_package(GLUT)

cuda_add_executable(devicequery
	devicequery.cpp
	)

set(kfusion_cuda_srcs
	helpers.cu
	helpers.h
	kfusion.cu
	kfusion.h
	cutil_math.h
	)
	
cuda_add_library(kfusion SHARED
	${kfusion_cuda_srcs}
	perfstats.h
	OPTIONS
	-O3
	)

if (GLUT_FOUND AND FREENECT_FOUND)		
	
	include_directories(${FREENECT_INCLUDE_PATHS})

	add_executable(kfusion_kinect
		kinect.cpp
		)
	target_link_libraries(kfusion_kinect kfusion ${FREENECT_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ) 

endif()


if(GLUT_FOUND)

	add_executable(kfusion_test
		test.cpp
		)
	
	target_link_libraries(kfusion_test kfusion ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}) 

endif()
	
