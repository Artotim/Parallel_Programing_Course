#include <stdio.h>
#include <stdlib.h>
#include "matrix_array_routines.h"


void printArray(float arr[], int size) {
    /* Display float array */

    int arrayIndex = 0;
    while (arrayIndex <= size - 1) {
        if (arrayIndex > 4 && arrayIndex < size - 5 && size > 15) {
            printf(". . . . ");
            arrayIndex = size - 5;
        }

        printf("%f ", arr[arrayIndex]);
        arrayIndex++;
    }
    printf("\n");
}


void printIntArray(int arr[], int size) {
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


void printMatrix(float** matrix, int rows, int columns) {
    /* Prints matrix */

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            printf("%f ",matrix[i][j]);
        }
        putchar('\n');
    }
}


float *floatArrayAlloc(long size, int populate) {
    float *array;
    if (populate) {
        array = calloc(1, sizeof(float) * size);
    } else {
        array = malloc(sizeof(float) * size);
    }
    if(array == NULL) exit(EXIT_FAILURE);

    return array;
}


int *intArrayAlloc(long size, int populate) {
    int *array;
    if (populate) {
        array = calloc(1, sizeof(int) * size);
    } else {
        array = malloc(sizeof(int) * size);
    }
    if(array == NULL) exit(EXIT_FAILURE);

    return array;
}


float **matrixAlloc(int rows, int columns) {
    /* Allocate memory for matrix */

    float **matrix = malloc(sizeof(float) * rows * columns);

    if(matrix == NULL) exit(EXIT_FAILURE);

    for(int i = 0; i < rows; i++) {
        matrix[i] = calloc(1, sizeof(float) * columns);
        if(matrix[i] == NULL) exit(EXIT_FAILURE);
    }

    return matrix;
}


void matrixFree(float **matrix, int rows) {
    /* Free matrix memory */

    for(int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}
