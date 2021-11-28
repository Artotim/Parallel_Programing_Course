#include <stdlib.h>
#include <string.h>

#include "read_write_routines.h"


FILE *openMatrixFile(char *fileName) {
    /* Reads matrix from mtx file */


    FILE *mtxFile = fopen(fileName, "r");
    if (mtxFile == NULL) {
        fprintf(stderr, "Fatal: failed to open arg file: %s!\n", fileName);
        exit(EXIT_FAILURE);
    }

    return mtxFile;
}


void getMatrixDimensions(FILE *mtxFile, long *rows, long *columns) {
    /* Get matrix dimensions from file */

    char *line = NULL;
    char *stringNumber;
    char *ptr;
    size_t len = 0;

    getline(&line, &len, mtxFile);  // Discard first line
    getline(&line, &len, mtxFile);  // Read line with dimensions

    stringNumber = strtok(line, " ");
    *columns = strtol(stringNumber, &ptr, 10);

    stringNumber = strtok(NULL, " ");
    *rows = strtol(stringNumber, &ptr, 10);

    printf("\nNumber of cells: %ld\nNumber of genes: %ld\n", *rows, *columns);
    free(line);
}


void populateMatrix(FILE *argFile, float **matrix, float *sum) {
    /* Assign values to the matrix */

    char *line = NULL;
    char *stringNumber;
    size_t len = 0;

    long row, column;
    float value;
    char *ptr;

    printf("\nPopulating matrix...\n");
    while (getline(&line, &len, argFile) != -1) {

        stringNumber = strtok(line, " ");
        column = strtol(stringNumber, &ptr, 10) - 1;

        stringNumber = strtok(NULL, " ");
        row = strtol(stringNumber, &ptr, 10) - 1;

        stringNumber = strtok(NULL, " ");
        value = strtof(stringNumber, &ptr);

        matrix[row][column] = value;
        sum[row] += value;
    }
    free(line);
}


void normalizeCPM(float **matrix, const float *sum, long numRows, long numColumns) {
    /* Perform CPM normalization */

    printf("\nNormalizing...\n");

    for (int objIdx = 0; objIdx < numRows; objIdx++) {
        if (sum[objIdx] != 0) {
            for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
                matrix[objIdx][columnIdx] = (matrix[objIdx][columnIdx] / sum[objIdx]) * 1e6f;
            }
        }
    }
}


void getJobTimes(struct timespec init, struct timespec reading, struct timespec total) {
    /* Get times spent */

    float readingTime = (reading.tv_sec - init.tv_sec);
    readingTime += (reading.tv_sec - init.tv_sec) / 1000000000.0;
    printf("\nReading Time: %f \n", readingTime);

    float executionTime = (total.tv_sec - init.tv_sec);
    executionTime += (total.tv_nsec - init.tv_nsec) / 1000000000.0;
    printf("Total execution time is: %f\n", executionTime);
}


void writeOutput(float **centers, int *membership, const mtxDimensions job, const float *columnsWeights) {
    /* Write results to output file */

    FILE *centFiles = fopen("center.txt", "w+");
    if (centFiles == NULL) {
        fprintf(stderr, "Fatal: failed to open output file.\n");
        exit(EXIT_FAILURE);
    }

    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        long centersCol = 0;

        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            if (columnsWeights[columnIdx]) {
                fprintf(centFiles, "%f ", centers[clusterIdx][centersCol]);
                centersCol++;
            } else {
                fprintf(centFiles, "%f ", 0.0f);
            }
        }
        fprintf(centFiles, "\n");
    }
    fclose(centFiles);


    FILE *outFile = fopen("members.txt", "w+");
    if (outFile == NULL) {
        fprintf(stderr, "Fatal: failed to open output file.\n");
        exit(EXIT_FAILURE);
    }

    for (int arrayIndex = 0; arrayIndex < job.numObjs; arrayIndex++) {
        fprintf(outFile, "%d\n", membership[arrayIndex]);
    }


    fclose(outFile);
}
