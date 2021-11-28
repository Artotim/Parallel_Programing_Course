#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "array_routines.h"
#include "minMaxKMeans.h"


void getWeightedObjects(const float *matrix, float *weightedObj, long numObjs, long numColumns, const float *weights,
                        const long nonZeroWeights) {
    /* Calculates weight for every column in every object */

    long numRows = 0;

    #pragma omp for
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (weights[columnIdx] != 0) {
            for (int objIdx = 0; objIdx < numObjs; objIdx++) {
                weightedObj[(objIdx * nonZeroWeights) + numRows] =
                        matrix[(objIdx * numColumns) + columnIdx] * sqrtf(weights[columnIdx]);
            }
            numRows++;
        }
    }
}


void moveCenterMeanBalanced(const float *weightedObjs, float *balancedCenters, const mtxDimensions job,
                            long nonZeroWeights, const int *centroids) {
    /* Moves centers according to calculated weights */

    int cluster, columnIdx;

    int *sumClusterPoints = intArrayAlloc(job.numClusters, 1);



    // Get sum of points in cluster considering weights
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        cluster = centroids[objIdx];
        sumClusterPoints[cluster]++;

    #pragma omp parallel for firstprivate(objIdx, cluster)
        for (columnIdx = 0; columnIdx < nonZeroWeights; columnIdx++) {
            balancedCenters[(cluster * nonZeroWeights) + columnIdx] += weightedObjs[(objIdx * nonZeroWeights) +
                                                                                    columnIdx];
        }
    }

    // Calculate mean for each cluster and move center to it
    #pragma omp parallel for collapse(2)
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (columnIdx = 0; columnIdx < nonZeroWeights; columnIdx++) {
            balancedCenters[(clusterIdx * nonZeroWeights) + columnIdx] =
                    balancedCenters[(clusterIdx * nonZeroWeights) + columnIdx] / (float) sumClusterPoints[clusterIdx];
        }
    }

    free(sumClusterPoints);
}


int countUniques(const int *uniqueNearest, int numClusters) {
    // Return count of unique values in array

    int ans = 0;
    #pragma omp parallel for reduction(+:ans)
    for (int i = 0; i < numClusters; i++) {
        if (uniqueNearest[i] > 0) {
            ans++;
        }
    }

    return ans;
}


void cpuFindNearestCluster(int *clustersMembership, const float *euclideanDistances, const mtxDimensions job) {
    /* Returns the  nearest cluster from the point */
    const long numClusters = job.numClusters;


    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {

        int nearestCluster = 0;
        float min_dist = FLT_MAX;

        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (min_dist > euclideanDistances[(objIdx * numClusters) + clusterIdx]) {
                min_dist = euclideanDistances[(objIdx * numClusters) + clusterIdx];
                nearestCluster = clusterIdx;
            }
        }
        clustersMembership[nearestCluster]++;
    }
}


kMeansOutput updateCenters(const float *matrix, mtxDimensions jobInfo, float *columnsWeights,
                           int *membership, int nonZeroWeights) {
    /* Main routine for updating centers according to objects weights */

    printf("Performing centers update...");
    const long numObjs = jobInfo.numObjs, numColumns = jobInfo.numColumns, numClusters = jobInfo.numClusters;

    float *weightedObjs = floatArrayAlloc(numObjs * nonZeroWeights, 0);
    float *balancedCenters = floatArrayAlloc(numClusters * nonZeroWeights, 1);

    // Weights every object
    getWeightedObjects(matrix, weightedObjs, numObjs, numColumns, columnsWeights, nonZeroWeights);

    // Move centers according to weights
    moveCenterMeanBalanced(weightedObjs, balancedCenters, jobInfo, nonZeroWeights, membership);


    mtxDimensions updateJob = {numObjs, numObjs, 0, nonZeroWeights, numClusters, 0};

    // Get euclidean distances
    float *euclideanDistances = floatArrayAlloc(numObjs * numClusters, 0);
    cpuGetEuclideanDist((const float *) weightedObjs, balancedCenters, euclideanDistances, updateJob);

    // Get nearest cluster for each point
    int *clustersMembership = intArrayAlloc(numClusters, 1);
    cpuFindNearestCluster(clustersMembership, euclideanDistances, updateJob);

    free(euclideanDistances);

    // Perform MinMaxKMeans with defined centers if every cluster is populated else random
    kMeansOutput kMeansResult;
    if (countUniques(clustersMembership, numClusters) == numClusters) {
        kMeansResult = getKMeansResult(weightedObjs, NULL, balancedCenters, updateJob, 0);
    } else {
        kMeansResult = getKMeansResult(weightedObjs, NULL, balancedCenters, updateJob, 1);
    }

    free(weightedObjs);
    free(clustersMembership);

    printf("Done\n");
    return kMeansResult;

}
