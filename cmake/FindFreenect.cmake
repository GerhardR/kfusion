
set(LIBRARY_PATHS
	~/usr/lib
	~/usr/local/lib
	/usr/lib
	/usr/local/lib
	)

find_library(FREENECT_LIBRARY 
	NAMES freenect
	PATHS ${LIBRARY_PATHS}
	)
	
find_path(FREENECT_INCLUDE_PATH libfreenect/libfreenect.h
#    PATH_SUFFIXES libfreenect
 	~/usr/include
	~/usr/local/include
	/usr/include
	/usr/local/include
	)
	
find_path(LIBUSB1_INCLUDE_PATH libusb-1.0/libusb.h
#    PATH_SUFFIXES libusb-1.0
 	~/usr/include
	~/usr/local/include
	/usr/include
	/usr/local/include
	)
	
if(FREENECT_LIBRARY AND FREENECT_INCLUDE_PATH)
	set(FREENECT_FOUND TRUE)
	set(FREENECT_INCLUDE_PATHS ${LIBUSB1_INCLUDE_PATH} ${FREENECT_INCLUDE_PATH} CACHE STRING "The include paths needed to use freenect")
    set(FREENECT_LIBRARIES ${FREENECT_LIBRARY} CACHE STRING "The libraries needed to use freenect")
endif()

mark_as_advanced(
    FREENECT_INCLUDE_PATHS
    FREENECT_LIBRARIES
	)