find_path(TOON_INCLUDE_PATH TooN/TooN.h
	~/usr/include
	~/usr/local/include
	/usr/include
	/usr/local/include
	thirdparty
)

if(TOON_INCLUDE_PATH)
	set(TOON_FOUND TRUE)
	set(TOON_INCLUDE_PATHS ${TOON_INCLUDE_PATH} CACHE STRING "The include paths needed to use TooN")
endif()

mark_as_advanced(
	TOON_INCLUDE_PATHS
)
