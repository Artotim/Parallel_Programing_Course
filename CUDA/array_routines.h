#ifndef FINAL_PARALLEL_ARRAY_ROUTINES_H
#define FINAL_PARALLEL_ARRAY_ROUTINES_H

void printArray(const float arr[], int size);

void printIntArray(const int arr[], int size);

float *floatArrayAlloc(long size, int populate);

int *intArrayAlloc(long size, int populate);

float *cudaFloatAlloc(long arrSize, int populate);

int *cudaIntAlloc(long arrSize, int populate);

void debugFloat(float *target, long size);

void debugInt(int *target, long size);

#endif //FINAL_PARALLEL_ARRAY_ROUTINES_H
