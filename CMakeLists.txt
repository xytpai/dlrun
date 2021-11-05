cmake_minimum_required(VERSION 3.0)
project(dlrun)

set(ROOTDIR ${PROJECT_SOURCE_DIR})

aux_source_directory(${ROOTDIR}/examples EXAMPLES_DIR)
aux_source_directory(${ROOTDIR}/core DLRUN_DIR)
include_directories(${ROOTDIR}/core/include)
message("Example files: ${EXAMPLES_DIR}")

foreach(EXAMPLE ${EXAMPLES_DIR})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_EXAMPLE ${EXAMPLE})
    add_executable(test_${FILE_EXAMPLE} ${EXAMPLE} ${DIR_SRCS})
endforeach()
