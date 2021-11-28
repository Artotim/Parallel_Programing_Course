#include <stdio.h>
#include <stdlib.h>
#include "array_routines.h"


float *floatArrayAlloc(long size, int populate) {
    /* Alloc float array */
    float *array;
    if (populate) {
        array = (float *) calloc(1, sizeof(float) * size);
    } else {
        array = (float *) malloc(sizeof(float) * size);
    }
    if (array == NULL) exit(EXIT_FAILURE);

    return array;
}


int *intArrayAlloc(long size, int populate) {
    /* Alloc int array */

    int *array;
    if (populate) {
        array = (int *) calloc(1, sizeof(int) * size);
    } else {
        array = (int *) malloc(sizeof(int) * size);
    }
    if (array == NULL) exit(EXIT_FAILURE);

    return array;
}


float *cudaFloatAlloc(long arrSize, int populate) {
    /* Alloc float array in GPU*/

    float *array;

    cudaMalloc(&array, sizeof(float) * arrSize);

    if (populate) {
        cudaMemset(array, 0, sizeof(float) * arrSize);
    }

    return array;
}


int *cudaIntAlloc(long arrSize, int populate) {
    /* Alloc int array in GPU*/

    int *array;

    cudaMalloc(&array, sizeof(int) * arrSize);

    if (populate) {
        cudaMemset(array, 0, sizeof(int) * arrSize);
    }

    return array;
}


void printArray(const float arr[], int size) {
    /* Display float array */

    int arrayIndex = 0;
    while (arrayIndex <= size - 1) {
        if (arrayIndex > 4 && arrayIndex < size - 5 && size > 30) {
            printf(". . . . ");
            arrayIndex = size - 5;
        }

        printf("%f ", arr[arrayIndex]);
        arrayIndex++;
    }
    printf("\n");
}


void printIntArray(const int arr[], int size) {
    /* Display int array */

    int arrayIndex = 0;
    while (arrayIndex <= size - 1) {
        if (arrayIndex > 4 && arrayIndex < size - 5 && size > 15) {
            printf(". . . . ");
            arrayIndex = size - 5;
        }

        printf("%d ", arr[arrayIndex]);
        arrayIndex++;
    }
    printf("\n");
}


void debugFloat(float *target, long size) {
    /* Print float array and exit */

    float *debugArr = floatArrayAlloc(size, 0);
    cudaMemcpy(debugArr, target, sizeof(float) * size, cudaMemcpyDefault);
    printArray(debugArr, size);
    exit(1);
}


void debugInt(int *target, long size) {
    /* Print float array and exit */

    int *debugArr = intArrayAlloc(size, 0);
    cudaMemcpy(debugArr, target, sizeof(int) * size, cudaMemcpyDefault);
    printIntArray(debugArr, size);
    exit(1);
}
