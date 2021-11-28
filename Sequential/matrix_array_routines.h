#ifndef FINAL_MATRIX_ARRAY_ROUTINES_H
#define FINAL_MATRIX_ARRAY_ROUTINES_H

void printArray(float arr[], int size);

void printIntArray(int arr[], int size);

void printMatrix(float** matrix, int rows, int columns);

float *floatArrayAlloc(long size, int populate);

int *intArrayAlloc(long size, int populate);

float **matrixAlloc(int rows, int columns);

void matrixFree(float **matrix, int rows);

#endif //FINAL_MATRIX_ARRAY_ROUTINES_H
