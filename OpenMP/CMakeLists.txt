cmake_minimum_required(VERSION 3.16)
project(Final_parallel C)

set(CMAKE_C_STANDARD 99)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(Final_parallel main.c sparseKMeans.c structs.h sparseKMeans.h updateCenters.c updateCenters.h updateWeights.c minMaxKMeans.c read_write_routines.c array_routines.c read_write_routines.h array_routines.h minMaxKMeans.h updateWeights.h)