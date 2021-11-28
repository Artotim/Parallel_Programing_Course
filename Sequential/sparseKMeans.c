#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sparseKMeans.h"
#include "minMaxKMeans.h"
#include "updateWeights.h"
#include "updateCenters.h"
#include "matrix_array_routines.h"


float absoluteSubtract(const float *weights, const float *weights_old, const long numColumns) {
    /* Gets the sum of the difference between two arrays */

    float result = 0;
    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++) {
        result += fabsf(weights[columnIndex] - weights_old[columnIndex]);
    }

    return result;
}


void initializeWeights(float *columnsWeights, const long numColumns) {
    /* initialize weights array to 1/sqrtf(columns) */

    float initWeight = 1.f / sqrtf((float) numColumns);
    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++) {
        columnsWeights[columnIndex] = initWeight;
    }
}


int getNonZeroWeights(const float *columnWeights, const long numColumns) {
    /* Get number of non zero weight values */

    int nonZero = 0;
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (columnWeights[columnIdx] != 0) nonZero++;
    }

    return nonZero;
}


kMeansOutput sparseKMeans(const float **matrix, const mtxDimensions jobInfo, float *columnsWeights, const int maxLoops) {
    /* Main routine for sparse KMeans */

    // Initialize values
    const long numColumns = jobInfo.numColumns, numObjs = jobInfo.numObjs;
    float boundWeight = 2.f; // should be a list from 1.1 to sqrtf numColumns


    // Initialize columns weights
    float *oldColumnsWeights = floatArrayAlloc(numColumns, 0);
    initializeWeights(columnsWeights, numColumns);


    // Perform MinMaxKMeans once
    int *membershipArray = intArrayAlloc(numObjs, 0);
    float **centroids = matrixAlloc(jobInfo.numClusters, numColumns);

    kMeansOutput kMeansResult = getKMeansResult(matrix, centroids, jobInfo, 1);
    memcpy(membershipArray, kMeansResult.cluster, sizeof(int) * numObjs);


    // Variables for loop
    int nonZeroWeights = numColumns, nIter = 1;
    float condition = 1;
    const float threshold = 0.0001f;

    // Loop while condition not reached
    while (condition > threshold && nIter < maxLoops) {
        printf("\nRunning loop %d\n", nIter);
        printf("Weights convergence: %f\n ", condition);
        printf("\n");

        memcpy(oldColumnsWeights, columnsWeights, sizeof(float) * numColumns);

        // Updates centers according to cluster weights
        if (nIter > 1) {
            cleanOldKMeans(kMeansResult);
            matrixFree(kMeansResult.centers, jobInfo.numClusters);
            kMeansResult = updateCenters(matrix, jobInfo, columnsWeights, membershipArray, nonZeroWeights);
            memcpy(membershipArray, kMeansResult.cluster, sizeof(int) * numObjs);
        }

        // Updates columns weights
        updatesWeights(matrix, jobInfo, membershipArray, boundWeight, kMeansResult.clusterWeights,
                       kMeansResult.pValue, columnsWeights);
        nonZeroWeights = getNonZeroWeights(columnsWeights, numColumns);

        // Get condition for next loop
        condition = absoluteSubtract(columnsWeights, oldColumnsWeights, numColumns) /
                    absoluteSum(oldColumnsWeights, numColumns);

        nIter++;
    }


    free(oldColumnsWeights);
    free(membershipArray);

    return kMeansResult;
}
