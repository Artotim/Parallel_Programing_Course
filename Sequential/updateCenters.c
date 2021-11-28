#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "structs.h"
#include "updateCenters.h"
#include "matrix_array_routines.h"
#include "minMaxKMeans.h"


void getWeightedObjects(const float **matrix, float **weightedObj, long numObjs, long numColumns, float *weights) {
    /* Calculates weight for every column in every object */

    long numRows = 0;

    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (weights[columnIdx] != 0) {
            for (int objIdx = 0; objIdx < numObjs; objIdx++) {
                weightedObj[objIdx][numRows] = matrix[objIdx][columnIdx] * sqrtf(weights[columnIdx]);
            }
            numRows++;
        }
    }
}


void moveCenterMeanBalanced(float **weightedObjs, float **balancedCenters, const mtxDimensions job,
                            long nonZeroWeights, const int *centroids) {
    /* Moves centers according to calculated weights */

    int cluster, columnIdx;

    int *sumClusterPoints = intArrayAlloc(job.numClusters, 1);

    // Get sum of points in cluster considering weights
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        cluster = centroids[objIdx];
        sumClusterPoints[cluster]++;
        for (columnIdx = 0; columnIdx < nonZeroWeights; columnIdx++) {
            balancedCenters[cluster][columnIdx] += weightedObjs[objIdx][columnIdx];
        }
    }

    // Calculate mean for each cluster and move center to it
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (columnIdx = 0; columnIdx < nonZeroWeights; columnIdx++) {
            balancedCenters[clusterIdx][columnIdx] =
                    balancedCenters[clusterIdx][columnIdx] / (float) sumClusterPoints[clusterIdx];
        }
    }

    free(sumClusterPoints);
}


int countUniques(const int *uniqueNearest, int numClusters) {
    // Return count of unique values in array

    int ans = 0;

    for (int i = 0; i < numClusters; i++) {
        if (uniqueNearest[i] > 0) {
            ans++;
        }
    }

    return ans;
}


kMeansOutput updateCenters(const float **matrix, mtxDimensions jobInfo, float *columnsWeights,
                           int *membership, int nonZeroWeights) {
    /* Main routine for updating centers according to objects weights */

    printf("Performing centers update...");
    const long numObjs = jobInfo.numObjs, numColumns = jobInfo.numColumns, numClusters = jobInfo.numClusters;

    float **weightedObjs = matrixAlloc(numObjs, nonZeroWeights);
    float **balancedCenters = matrixAlloc(numClusters, nonZeroWeights);

    // Weights every object
    getWeightedObjects(matrix, weightedObjs, numObjs, numColumns, columnsWeights);

    // Move centers according to weights
    moveCenterMeanBalanced(weightedObjs, balancedCenters, jobInfo, nonZeroWeights, membership);

    mtxDimensions updateJob = {numObjs, nonZeroWeights, numClusters};
    // Get euclidean distances
    float **euclideanDistances = matrixAlloc(numObjs, numClusters);
    getEuclideanDist((const float **) weightedObjs, balancedCenters, euclideanDistances, updateJob);

    // Get nearest cluster for each point
    int *clustersMembership = intArrayAlloc(numClusters, 1);
    int nearestCluster;
    for (int objIdx = 0; objIdx < numObjs; objIdx++) {
        nearestCluster = getNearestCluster(euclideanDistances[objIdx], numClusters);
        clustersMembership[nearestCluster]++;
    }
    matrixFree(euclideanDistances, numObjs);

    // Perform MinMaxKMeans with defined centers if every cluster is populated else random

    kMeansOutput kMeansResult;
    if (countUniques(clustersMembership, numClusters) == numClusters) {
        kMeansResult = getKMeansResult((const float **) weightedObjs, balancedCenters, updateJob, 0);
    } else {
        kMeansResult = getKMeansResult((const float **) weightedObjs, balancedCenters, updateJob, 1);
    }

    matrixFree(weightedObjs, numObjs);
    free(clustersMembership);

    printf("Done\n");
    return kMeansResult;
}
