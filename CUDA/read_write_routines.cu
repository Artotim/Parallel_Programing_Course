#include <stdlib.h>
#include <string.h>

#include "read_write_routines.h"


void getJobTimes(struct timespec init, struct timespec reading, struct timespec total) {
    /* Get times spent */

    float readingTime = (reading.tv_sec - init.tv_sec);
    readingTime += (reading.tv_sec - init.tv_sec) / 1000000000.0;
    printf("\nReading Time: %f \n", readingTime);

    float executionTime = (total.tv_sec - init.tv_sec);
    executionTime += (total.tv_nsec - init.tv_nsec) / 1000000000.0;
    printf("Total execution time is: %f\n", executionTime);
}



FILE *openMatrixFile(char *fileName) {
    /* Reads matrix from mtx file */

    //fileName = (char *) "C:\\Users\\Tango\\Final\\data\\teste.txt";

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


void populateMatrix(FILE *argFile, float *matrix, long numColumns, float *sum) {
    /* Assign values to the matrix */

    char *line = NULL;
    char *stringNumber;
    size_t len = 0;

    long row, column;
    float value;
    char *ptr;

    printf("\nPopulating matrix... ");
    while (getline(&line, &len, argFile) != -1) {

        stringNumber = strtok(line, " ");
        column = strtol(stringNumber, &ptr, 10) - 1;

        stringNumber = strtok(NULL, " ");
        row = strtol(stringNumber, &ptr, 10) - 1;

        stringNumber = strtok(NULL, " ");
        value = strtof(stringNumber, &ptr);

        matrix[(row * numColumns) + column] = value;
        sum[row] += value;
    }
    free(line);

    printf("Done!\n");
}


void writeOutput(float *centers, int *membership, const mtxDimensions job, const float *columnsWeights, const int nonZeroWeights) {
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
                fprintf(centFiles, "%f ", centers[clusterIdx * nonZeroWeights + centersCol]);
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


void normalizeCPM(float *matrix, const float *sum, long numRows, long numColumns) {
    /* Perform CPM normalization */

    printf("\nNormalizing...\n");

    #pragma omp parallel for collapse(2)
    for (int objIdx = 0; objIdx < numRows; objIdx++) {
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            matrix[(objIdx * numColumns) + columnIdx] = (matrix[(objIdx * numColumns) + columnIdx] / sum[objIdx]) * 1e6f;
        }
    }
}


void checkGPUMem(const long numObjs, const long numColumns, long *d_numObjs, long *d_arrSize, int *fit) {
    /* Check GPU memory and decide if matrix will fit in there */

    const long arrSize = numObjs * numColumns;

    size_t free_t, total_t;
    cudaMemGetInfo(&free_t, &total_t);

    double free_m, total_m, used_m;
    free_m = free_t / 1048576.0;
    total_m = total_t / 1048576.0;
    used_m = total_m - free_m;

    printf("GPU mem free %.2f MB. Mem total %.2f MB. Mem used %.2f MB\n", free_m, total_m, used_m);

    if ((double) (sizeof(float) * arrSize) > 0.9f * free_t) {
        printf("\nMatrix doesn't fit in GPU memory\n");

        double total_percentage = (double) ((sizeof(float) * arrSize) * 100) / free_t;

        *d_numObjs = (long) (.9 * numObjs);

        *d_arrSize = numColumns * *d_numObjs;
        *fit = 0;


        printf("Will fit %ld from %f \n\n", *d_numObjs, total_percentage);
    } else {
        printf("\nMatrix fit in memory!\n");
        *d_arrSize = arrSize;
        *d_numObjs = numObjs;
        *fit = 1;
    }
}
