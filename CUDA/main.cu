#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>

#include "read_write_routines.h"
#include "array_routines.h"
#include "sparseKMeans.h"


int main(int argc, char *argv[]) {

    // Initialize times
    struct timespec initTime, readingTime, totalTime;
    clock_gettime(CLOCK_MONOTONIC, &initTime);

    // Open matrix file
    FILE *mtxFile = openMatrixFile(argv[1]);

    // Reads number of rows and columns from file
    long numObjs, numColumns;
    getMatrixDimensions(mtxFile, &numObjs, &numColumns);
    long arrSize = numColumns * numObjs;


    // Populate matrix and expression sum for normalization
    float *rowSum = floatArrayAlloc(numObjs, 1);
    float *matrix = floatArrayAlloc(arrSize, 1);
    populateMatrix(mtxFile, matrix, numColumns, rowSum);
    clock_gettime(CLOCK_MONOTONIC, &readingTime);


    // Normalize matrix using CPM
    if (!strcmp(argv[3], "yes")) {
        double kMeansStart = omp_get_wtime();
        normalizeCPM(matrix, rowSum, numObjs, numColumns);
        printf("Normalization Time = %f\n", omp_get_wtime() - kMeansStart);
    }
    free(rowSum);


    // Check GPU memory
    long d_numObjs, d_arrSize;
    int fit;
    checkGPUMem(numObjs, numColumns, &d_numObjs, &d_arrSize, &fit);


    // Initalize values for run
    long numClusters = 7;
    if (numObjs < 10) numClusters = 2;
    const int maxLoops = 10;
    const int numThreads = atoi(argv[2]);
    printf("Num Threads = %d\n", numThreads);
    omp_set_num_threads(numThreads);

    const mtxDimensions jobInfo = {numObjs, numObjs - d_numObjs, d_numObjs, numColumns, numClusters, fit};
    float *columnsWeights = floatArrayAlloc(numColumns, 1);


    // Perform Sparse Min Max KMeans
    puts("\nInitializing Sparse KMeans");
    kMeansOutput kMeansResult = sparseKMeans((const float *) matrix, jobInfo, columnsWeights, maxLoops);
    clock_gettime(CLOCK_MONOTONIC, &totalTime);
    puts("\nDone!");


    // Writes output
    writeOutput(kMeansResult.centers, kMeansResult.cluster, jobInfo, columnsWeights, getNonZeroWeights(columnsWeights, numColumns));


    // Get job time
    getJobTimes(initTime, readingTime, totalTime);


    free(matrix);
    free(kMeansResult.centers);
    free(kMeansResult.cluster);
    free(kMeansResult.clusterWeights);
    free(columnsWeights);

    return 0;
}

//nvcc  -Xcompiler " -fopenmp" -o parallel main.cu