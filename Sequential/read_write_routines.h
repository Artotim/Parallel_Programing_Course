#ifndef FINAL_READ_WRITE_ROUTINES_H
#define FINAL_READ_WRITE_ROUTINES_H

#include <stdio.h>
#include <time.h>
#include "structs.h"

FILE *openMatrixFile(char *fileName);

void getMatrixDimensions(FILE *mtxFile, long *rows, long *columns);

void populateMatrix(FILE *argFile, float **matrix, float *sum);

void normalizeCPM(float **matrix, const float *sum, long numRows, long numColumns);

void getJobTimes(struct timespec init, struct timespec reading, struct timespec total);

void writeOutput(float **centers, int *membership, mtxDimensions job, const float *columnsWeights);

#endif //FINAL_READ_WRITE_ROUTINES_H
