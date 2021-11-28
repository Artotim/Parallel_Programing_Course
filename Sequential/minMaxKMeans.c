#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "structs.h"
#include "minMaxKMeans.h"
#include "matrix_array_routines.h"


void cleanOldKMeans(kMeansOutput oldKMeans) {
    /* Free unused kMeans results */

    free(oldKMeans.cluster);
    free(oldKMeans.clusterWeights);
}


void initializeRandomCenters(const float **matrix, float **centroids, const mtxDimensions job) {
    /* Initialize centroids to random points */

    int randomPoint;

    srand(time(NULL) * job.numObjs);

    printf("\nSetting random centers to");
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        randomPoint = rand() % job.numObjs;
        printf(" %d", randomPoint);

        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            centroids[clusterIdx][columnIdx] = matrix[randomPoint][columnIdx];
        }
    }

    printf(".\n\n");
}


void initializeClusterWeights(float *clusterWeights, float *oldClusterWeights, const long numClusters) {
    /* Initialize cluster weights */

    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        clusterWeights[clusterIdx] = 1 / (float) numClusters;
        oldClusterWeights[clusterIdx] = 1 / (float) numClusters;
    }
}


void getEuclideanDist(const float **matrix, float **centroids, float **euclideanDistances, const mtxDimensions job) {
    /* Find euclidean distances for every attribute in every object to each clusters */

    float distance;

    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (int objsIdx = 0; objsIdx < job.numObjs; objsIdx++) {
            distance = 0;

            for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
                distance += powf((matrix[objsIdx][columnIdx] - centroids[clusterIdx][columnIdx]), 2);
            }

            euclideanDistances[objsIdx][clusterIdx] = sqrtf(distance);
        }
    }
}


void getClusterMeans(const float **matrix, float **centroids, const int *clusterSize, const int *objsMembership,
                     const mtxDimensions job) {
    /* Move centroids to mean between every point in cluster */

    const long numClusters = job.numClusters;
    const long numColumns = job.numColumns;

    float **sumDistances = matrixAlloc(numClusters, numColumns);

    int cluster;

    // Loop over matrix to populate sumDistances
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        cluster = objsMembership[objIdx];
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            sumDistances[cluster][columnIdx] += matrix[objIdx][columnIdx];
        }
    }

    // Loop over sumDistances to get means and move centroids
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            centroids[clusterIdx][columnIdx] = sumDistances[clusterIdx][columnIdx] / (float) clusterSize[clusterIdx];
        }
    }

    matrixFree(sumDistances, numClusters);
}


int getNearestCluster(const float *distances, const int numClusters) {
    /* Returns the  nearest cluster from the point */

    int nearestCluster = 0;
    float min_dist = FLT_MAX;

    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        if (min_dist > distances[clusterIdx]) {
            min_dist = distances[clusterIdx];
            nearestCluster = clusterIdx;
        }
    }

    return nearestCluster;
}


void updateClusterWeights(float *clusterWeights, float **objsMtxMembership, float **euclideanDistances,
                          const mtxDimensions job, const float pValue) {
    /* Updates cluster weights according to distances and size */

    const long numClusters = job.numClusters;
    float sumWeightedDistances = 0;
    float *clustersDistance = floatArrayAlloc(numClusters, 1);

    // Loop over clusters to get sum of distances for each obj
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
            clustersDistance[clusterIdx] +=
                    objsMtxMembership[objIdx][clusterIdx] * euclideanDistances[objIdx][clusterIdx];
        }
        sumWeightedDistances += powf(clustersDistance[clusterIdx], (1 / (1 - pValue)));
    }

    // Loop over sum of clusters to update weights
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        clusterWeights[clusterIdx] = powf(clustersDistance[clusterIdx], (1 / (1 - pValue))) / sumWeightedDistances;
    }

    free(clustersDistance);
}


float updateSumWeights(float **weightedDistances, float **euclideanDistances, float *clusterWeights,
                       const float pValue, const mtxDimensions job) {
    /* Get sum of weights to check condition */

    float sumWeights = 0;
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
            weightedDistances[objIdx][clusterIdx] =
                    powf(clusterWeights[clusterIdx], pValue) * euclideanDistances[objIdx][clusterIdx];
            sumWeights += weightedDistances[objIdx][clusterIdx];
        }
    }
    return sumWeights;
}


void copyMatrix(float **dest, float** source, mtxDimensions jobInfo) {
    /* Copy one matrix into another */

    for (int objIdx = 0; objIdx < jobInfo.numObjs; objIdx++) {
        memcpy(dest[objIdx], source[objIdx], sizeof(float) * jobInfo.numClusters);
    }
}


kMeansOutput minMaxKMeans(const float **matrix, float **centroids, const mtxDimensions jobInfo, const int getRandomPoints) {
    /* Main routine for MinMaxKMeans */

    // Initialize variables
    const long numObjs = jobInfo.numObjs, numClusters = jobInfo.numClusters;
    int objIdx, clusterIdx;
    kMeansOutput kMeansResult;
    float pValue = 0;
    const float pStep = 0.01f, pMax = 0.5f;
    int flag = 1;


    // If random is true, initialize centers with random points
    if (getRandomPoints == 1) {
        initializeRandomCenters(matrix, centroids, jobInfo);
    }


    // DELETAR PAR A RANDOMIZAR
    for (int j = 0; j < jobInfo.numColumns; j++) {
        centroids[0][j] = matrix[0][j];
        centroids[1][j] = matrix[2][j];
    }
    printf("Performing MinMaxKMeans...");

    // Initialize matrices and arrays to be used
    float **oldObjsMembership = matrixAlloc(numObjs, numClusters);
    float **weightedDistances = matrixAlloc(numObjs, numClusters);
    float **objsMtxMembership = matrixAlloc(numObjs, numClusters);

    float *clusterWeights = floatArrayAlloc(numClusters, 0);
    float *oldClusterWeights = floatArrayAlloc(numClusters, 0);
    int *objsMembership = intArrayAlloc(numObjs, 0);


    // Initialize cluster weights
    initializeClusterWeights(clusterWeights, oldClusterWeights, numClusters);


    // Initialize euclidean distances
    float **euclideanDistances = matrixAlloc(numObjs, numClusters);
    getEuclideanDist(matrix, centroids, euclideanDistances, jobInfo);


    // Initialize matrix weights
    for (objIdx = 0; objIdx < numObjs; objIdx++) {
        for (clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            weightedDistances[objIdx][clusterIdx] =
                    powf(clusterWeights[clusterIdx], pValue) * euclideanDistances[objIdx][clusterIdx];
        }
    }


    // Variables for loop
    int empty = 0, nIter = 0;
    float oldSumWeights, sumWeights = 0;


    // Loop until weights converge to 0 or hit max loops
    while (1) {
        nIter++;


        // Classify each object and get cluster sizes
        int *clusterSize = intArrayAlloc(numClusters, 1);
        int nearestCluster;
        for (objIdx = 0; objIdx < numObjs; objIdx++) {
            nearestCluster = getNearestCluster(weightedDistances[objIdx], numClusters);
            objsMtxMembership[objIdx][nearestCluster] = 1;
            objsMembership[objIdx] = nearestCluster;
            clusterSize[nearestCluster]++;
        }

        // Lower pValue if some cluster is empty
        for (clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (clusterSize[clusterIdx] < 1) {
                empty = 1;
                pValue -= pStep;

                copyMatrix(objsMtxMembership, oldObjsMembership, jobInfo);
                memcpy(clusterWeights, oldClusterWeights, sizeof(float) * numClusters);
                break;
            }
        }

        // Break loop if pValue lesser than 0
        if (pValue < 0) {
            pValue = 0;
            flag = 0;
            free(clusterSize);
            break;
        }

        // Update centroids with distances means
        getClusterMeans(matrix, centroids, clusterSize, objsMembership, jobInfo);

        // Increase pValue if no cluster is empty
        if ((pValue < pMax) && (empty == 0)) {
            copyMatrix(oldObjsMembership, objsMtxMembership, jobInfo);
            memcpy(oldClusterWeights, clusterWeights, sizeof(float) * numClusters);
            pValue += pStep;
        }

        // Update euclidean distances to new centroids
        getEuclideanDist(matrix, centroids, euclideanDistances, jobInfo);

        // Update clusters weights according to distances
        updateClusterWeights(clusterWeights, objsMtxMembership, euclideanDistances, jobInfo, pValue);


        // Update sum of weights to verify condition
        oldSumWeights = sumWeights;
        sumWeights = updateSumWeights(weightedDistances, euclideanDistances, clusterWeights, pValue, jobInfo);

        free(clusterSize);

        // Break loop when weights convert to 0
        if (nIter > 1) {
            if (fabsf(sumWeights - oldSumWeights) <= 1e-04 || nIter >= 10) {
                break;
            }
        }
    }
    printf("Done\n");

    // Set return results
    kMeansResult.centers = centroids;  // here
    kMeansResult.flag = flag;
    kMeansResult.cluster = objsMembership;  // here
    kMeansResult.clusterWeights = clusterWeights; // here
    kMeansResult.pValue = pValue;


    matrixFree(weightedDistances, numObjs);
    matrixFree(objsMtxMembership, numObjs);
    matrixFree(oldObjsMembership, numObjs);
    matrixFree(euclideanDistances, numObjs);
    free(oldClusterWeights);

    return kMeansResult;
}


kMeansOutput getKMeansResult(const float **matrix, float **centroids, const mtxDimensions jobInfo, int getRandom) {
    /* Perform MinMax KMeans until all clusters are populated */
    int nIter = 0;

    kMeansOutput kMeansResult = minMaxKMeans(matrix, centroids, jobInfo, getRandom);

    while (kMeansResult.flag == 0 && nIter < 30) {
        getRandom = 1;
        nIter++;
        cleanOldKMeans(kMeansResult);
        kMeansResult = minMaxKMeans(matrix, centroids, jobInfo, getRandom);
    }

    return kMeansResult;
}
