//
// Created by Tango on 01/09/2021.
//

#ifndef FINAL_PARALLEL_ARRAY_ROUTINES_H
#define FINAL_PARALLEL_ARRAY_ROUTINES_H

void printArray(const float arr[], int size);

void printIntArray(const int arr[], int size);

float *floatArrayAlloc(long size, int populate);

int *intArrayAlloc(long size, int populate);

#endif //FINAL_PARALLEL_ARRAY_ROUTINES_H
