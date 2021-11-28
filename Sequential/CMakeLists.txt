cmake_minimum_required(VERSION 3.16)
project(Final)

set(CMAKE_CXX_STANDARD 14)

add_executable(Final main.c sparseKMeans.c sparseKMeans.h structs.h minMaxKMeans.c minMaxKMeans.h updateWeights.c updateWeights.h updateCenters.c updateCenters.h matrix_array_routines.c matrix_array_routines.h read_write_routines.c read_write_routines.h)