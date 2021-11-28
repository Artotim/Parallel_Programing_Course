#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "sparseKMeans.h"
#include "minMaxKMeans.h"
#include "updateWeights.h"
#include "updateCenters.h"
#include "array_routines.h"


void initializeWeights(float *columnsWeights, const long numColumns) {
    /* initialize weights array to 1/sqrtf(columns) */

    float initWeight = 1.f / sqrtf((float) numColumns);
#pragma omp parallel for
    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++) {
        columnsWeights[columnIndex] = initWeight;
    }
}


float absoluteSubtract(const float *weights, const float *weights_old, const long numColumns) {
    /* Gets the sum of the difference between two arrays */

    float result = 0;
#pragma omp parallel for reduction(+:result)
    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++) {
        result += fabsf(weights[columnIndex] - weights_old[columnIndex]);
    }

    return result;
}


int getNonZeroWeights(const float *columnWeights, const long numColumns) {
    /* Get number of non zero weight values */

    int nonZero = 0;
#pragma omp parallel for reduction(+:nonZero)
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (columnWeights[columnIdx] != 0) nonZero++;
    }

    return nonZero;
}


kMeansOutput sparseKMeans(const float *matrix, const mtxDimensions jobInfo, float *columnsWeights, const int maxLoops) {
    /* Main routine for sparse KMeans */

    // Initialize values
    const long numColumns = jobInfo.numColumns;
    float boundWeight = 2.f;

    float *d_matrix = cudaFloatAlloc(jobInfo.d_numObjs * numColumns, 0);
    cudaMemcpy(d_matrix, matrix, sizeof(float) * jobInfo.d_numObjs * numColumns, cudaMemcpyDefault);


    // Initialize columns weights
    float *oldColumnsWeights = floatArrayAlloc(numColumns, 0);
    initializeWeights(columnsWeights, numColumns);


    // Perform MinMaxKMeans once
    float *centroids = floatArrayAlloc(jobInfo.numClusters * numColumns, 1);

    double kMeansStart = omp_get_wtime();
    kMeansOutput kMeansResult = getKMeansResult(matrix, d_matrix, centroids, jobInfo, 1);
    double kMeansEnd = omp_get_wtime();

    // Variables for loop
    int nonZeroWeights = numColumns, nIter = 1;
    float condition = 1;
    const float threshold = 0.0001f;

    double totalCenters = 0, totalWeigths = 0, elapsedStart;

    // Loop while condition not reached
    while (condition > threshold && nIter < maxLoops) {
        printf("\nRunning loop %d\n", nIter);
        printf("Weights convergence: %f\n ", condition);
        printf("\n");


        memcpy(oldColumnsWeights, columnsWeights, sizeof(float) * numColumns);

        // Updates centers according to cluster weights
        if (nIter > 1) {

            free(kMeansResult.clusterWeights);
            free(kMeansResult.centers);
            elapsedStart = omp_get_wtime();
            kMeansResult = updateCenters(matrix, jobInfo, columnsWeights, kMeansResult.cluster, nonZeroWeights);
            totalCenters += omp_get_wtime() - elapsedStart;

        }

        // Updates columns weights
        elapsedStart = omp_get_wtime();
        updatesWeights(matrix, d_matrix, jobInfo, kMeansResult.cluster, boundWeight, kMeansResult.clusterWeights,
                       kMeansResult.pValue, columnsWeights);
        totalWeigths += omp_get_wtime() - elapsedStart;
        nonZeroWeights = getNonZeroWeights(columnsWeights, numColumns);


        // Get condition for next loop
        condition = absoluteSubtract(columnsWeights, oldColumnsWeights, numColumns) /
                    absoluteSum(oldColumnsWeights, numColumns);
        nIter++;
    }

    free(oldColumnsWeights);

    printf("Kmeans Time = %f\n", kMeansEnd - kMeansStart);
    printf("Centers Time = %f\n", totalCenters);
    printf("Weights Time = %f\n", totalWeigths);

    return kMeansResult;
}
