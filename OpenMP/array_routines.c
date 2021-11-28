#include <stdio.h>
#include <stdlib.h>
#include "array_routines.h"

float *floatArrayAlloc(long size, int populate) {
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
    int *array;
    if (populate) {
        array = (int *) calloc(1, sizeof(int) * size);
    } else {
        array = (int *) malloc(sizeof(int) * size);
    }
    if (array == NULL) exit(EXIT_FAILURE);

    return array;
}


void printArray(const float arr[], int size) {
    /* Display float array */

    int arrayIndex = 0;
    while (arrayIndex <= size - 1) {
//        if (arrayIndex > 4 && arrayIndex < size - 5 && size > 30) {
//            printf(". . . . ");
//            arrayIndex = size - 5;
//        }

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
