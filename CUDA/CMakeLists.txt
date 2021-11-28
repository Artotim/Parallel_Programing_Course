cmake_minimum_required(VERSION 3.16)
project(CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(final_parallel_cuda main.cu sparseKMeans.cu sparseKMeans.h updateCenters.cu updateCenters.h array_routines.cu array_routines.h read_write_routines.cu read_write_routines.h updateWeights.cu updateWeights.h minMaxKMeans.cu minMaxKMeans.h cuda_one_file.cu)

set_target_properties(
	CUDA
	PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON)