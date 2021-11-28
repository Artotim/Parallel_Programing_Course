// #define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparseKMeans.h"
#include "matrix_array_routines.h"
#include "read_write_routines.h"


int main(int argc, char *argv[]) {
    /* Main routine */

    struct timespec initTime, readingTime, totalTime;
    clock_gettime(CLOCK_MONOTONIC, &initTime);


    // Open matrix file
    FILE *mtxFile = openMatrixFile(argv[1]);


    // Reads number of rows and columns from file
    long numObjs, numColumns;
    getMatrixDimensions(mtxFile, &numObjs, &numColumns);


    // Populate matrix and expression sum for normalization
    float *rowSum = floatArrayAlloc(numObjs, 1);
    float **matrix = matrixAlloc(numObjs, numColumns);
    populateMatrix(mtxFile, matrix, rowSum);
    clock_gettime(CLOCK_MONOTONIC, &readingTime);


    // Normalize matrix using CPM
    if (!strcmp(argv[2], "yes")) {
        normalizeCPM(matrix, rowSum, numObjs, numColumns);
    }
    free(rowSum);


    // Initialize variables for KMeans
    const long numClusters = 2;
    const int maxLoops = 10;
    const mtxDimensions jobInfo = {numObjs, numColumns, numClusters};
    float *columnsWeights = floatArrayAlloc(numColumns, 0);



    // Perform Sparse Min Max KMeans
    puts("\nInitializing Sparse KMeans");
    kMeansOutput kMeansResult = sparseKMeans((const float **)matrix, jobInfo, columnsWeights, maxLoops);
    clock_gettime(CLOCK_MONOTONIC, &totalTime);
    puts("\nDone!");


    // Writes output
    writeOutput(kMeansResult.centers, kMeansResult.cluster, jobInfo, columnsWeights);


    // Get job time
    getJobTimes(initTime, readingTime, totalTime);


    matrixFree(matrix, numObjs);
    matrixFree(kMeansResult.centers, numClusters);
    free(kMeansResult.cluster);
    free(kMeansResult.clusterWeights);
    free(columnsWeights);

    return 0;
}
