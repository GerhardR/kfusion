cmake_minimum_required(VERSION 2.8)

project(TooN)


find_program(AWK awk)

set(TOON_HDR_PATH ${CMAKE_BINARY_DIR}/include/TooN CACHE STRING "TooN Header Path")


file(MAKE_DIRECTORY ${TOON_HDR_PATH}/optimization)
file(MAKE_DIRECTORY ${TOON_HDR_PATH}/internal)
file(MAKE_DIRECTORY ${TOON_HDR_PATH}/functions)


add_custom_command(
	OUTPUT ${TOON_HDR_PATH}/TooN.h
	COMMAND cp
	ARGS  ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${TOON_HDR_PATH}
	COMMAND cp
	ARGS -r ${CMAKE_CURRENT_SOURCE_DIR}/optimization ${TOON_HDR_PATH}
	COMMAND cp
	ARGS -r ${CMAKE_CURRENT_SOURCE_DIR}/internal ${TOON_HDR_PATH}	
	COMMAND cp
	ARGS -r ${CMAKE_CURRENT_SOURCE_DIR}/functions ${TOON_HDR_PATH}	
)

#	cp *.h $(hdr)
#	cp -r optimization $(hdr)/
#	cp -r internal $(hdr)/
#	cp -r functions $(hdr)/

add_custom_command(
	OUTPUT ${TOON_HDR_PATH}/internal/data_functions.hh
	COMMAND ${AWK}
	ARGS -f ${CMAKE_CURRENT_SOURCE_DIR}/make_data_functions.awk > ${TOON_HDR_PATH}/internal/data_functions.hh
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/make_data_functions.awk
	)

add_custom_command(
	OUTPUT ${TOON_HDR_PATH}/internal/make_vector.hh
	COMMAND ${AWK}
	ARGS -f ${CMAKE_CURRENT_SOURCE_DIR}/make_make_vector.awk > ${TOON_HDR_PATH}/internal/make_vector.hh
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/make_make_vector.awk
	)

add_custom_command(
	OUTPUT ${TOON_HDR_PATH}/internal/builtin_typeof.h
	COMMAND ${AWK}
	ARGS -f ${CMAKE_CURRENT_SOURCE_DIR}/make_typeof.awk > ${TOON_HDR_PATH}/internal/builtin_typeof.h
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/make_typeof.awk
	)

add_custom_target(TooN ALL 
	DEPENDS
	${TOON_HDR_PATH}/TooN.h
	${TOON_HDR_PATH}/internal/builtin_typeof.h
	${TOON_HDR_PATH}/internal/make_vector.hh	
	${TOON_HDR_PATH}/internal/data_functions.hh
)

