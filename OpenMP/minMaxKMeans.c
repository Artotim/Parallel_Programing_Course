#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>

#include "structs.h"
#include "minMaxKMeans.h"
#include "array_routines.h"

void initializeRandomCenters(const float *matrix, float *centroids, const mtxDimensions job) {
    /* Initialize centroids to random points */

    int randomPoint;

    srand(time(NULL) * job.numObjs);

    printf("\nSetting random centers to");

    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        randomPoint = rand() % job.numObjs;
        printf(" %d", randomPoint);

#pragma omp parallel for firstprivate(randomPoint, clusterIdx)
        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            centroids[clusterIdx * job.numColumns + columnIdx] = matrix[randomPoint * job.numColumns +
                                                                        columnIdx];
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



void cpuGetEuclideanDist(const float *matrix, float *centroids, float *euclideanDistances, const mtxDimensions job) {
    /* Find euclidean distances for every attribute in every object to each clusters */

    const long numColumns = job.numColumns;


    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {

#pragma omp parallel for
        for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
            float distance = 0;
            for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
                distance += powf((matrix[(objIdx * numColumns) + columnIdx] -
                                  centroids[(clusterIdx * numColumns) + columnIdx]),
                                 2);
            }

            euclideanDistances[(objIdx  * job.numClusters) + clusterIdx] = sqrtf(distance);
        }
    }

}

void resolveEuclideanDistances(const float *matrix,float *centroids, float *euclideanDistances, const mtxDimensions job) {

    cpuGetEuclideanDist(matrix, centroids, euclideanDistances, job);

}

void initializeDistanceWeights (float *weightedDistances, const float *clusterWeights, const  float *euclideanDistances,
                                const mtxDimensions job, const float pValue) {
    // Initialize matrix weights
    const long numObjs = job.numObjs, numClusters = job.numClusters;

#pragma omp parallel for collapse(2)
    for (int objIdx = 0; objIdx < numObjs; objIdx++) {
        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            weightedDistances[objIdx + clusterIdx * numObjs] =
                    powf(clusterWeights[clusterIdx], pValue) * euclideanDistances[objIdx + clusterIdx * numObjs];
        }
    }
}

int *getObjectsDistribution(const float *weightedDistances, float *objsMtxMembership, int *objsMembership, const mtxDimensions job) {
    const long numObjs = job.numObjs, numClusters = job.numClusters;
    int *clusterSize = intArrayAlloc(numClusters, 1);

#pragma omp parallel for
    for (int objIdx = 0; objIdx < numObjs; objIdx++) {
        int nearestCluster;
        float min_dist = FLT_MAX;

        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (min_dist > weightedDistances[objIdx * numClusters + clusterIdx]) {
                min_dist = weightedDistances[objIdx * numClusters + clusterIdx];
                nearestCluster = clusterIdx;
            }
        }

        objsMtxMembership[(objIdx * numClusters) + nearestCluster] = 1;
        objsMembership[objIdx] = nearestCluster;

    }


    //#pragma omp for
    for (int cluster = 0; cluster < numClusters; cluster++) {
        for (int objIdx = 0; objIdx < numObjs; objIdx++) {
            if (objsMembership[objIdx] == cluster)
                clusterSize[cluster]++;
        }
    }

    return clusterSize;
}



void cpuClusterMeans(const float *matrix, float *centroids,const int *objsMembership, const mtxDimensions job) {
    /* Move centroids to mean between every point in cluster */

    const long numColumns = job.numColumns;



    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        int clusterIdx = objsMembership[objIdx];

#pragma omp parallel for firstprivate(objIdx, clusterIdx)
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            centroids[(clusterIdx * numColumns) + columnIdx] += matrix[(objIdx * numColumns) + columnIdx];
        }
    }
}



void resolveClusterMeans(const float *matrix, float *centroids, const int *clusterSize,
                         int *objsMembership, const mtxDimensions job) {

#pragma omp parallel for collapse(2)
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            centroids[(clusterIdx * job.numColumns) + columnIdx] = 0;
        }
    }

    cpuClusterMeans(matrix, centroids, objsMembership, job);


#pragma omp parallel for collapse(2)
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            centroids[(clusterIdx * job.numColumns) + columnIdx] /= (float) clusterSize[clusterIdx];
        }
    }

}


void updateClusterWeights(float *clusterWeights, const float *objsMtxMembership, const float *euclideanDistances,
                          const mtxDimensions job, const float pValue) {
    /* Updates cluster weights according to distances and size */

    const long numClusters = job.numClusters;
    float sumWeightedDistances = 0;
    float *clustersDistance = floatArrayAlloc(numClusters, 1);

    // Loop over clusters to get sum of distances for each obj
#pragma omp parallel for reduction(+:sumWeightedDistances)
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
            clustersDistance[clusterIdx] +=
                    objsMtxMembership[(objIdx * numClusters) + clusterIdx] * euclideanDistances[(objIdx * numClusters) + clusterIdx];
        }
        sumWeightedDistances += powf(clustersDistance[clusterIdx], (1 / (1 - pValue)));
    }

    // Loop over sum of clusters to update weights
#pragma omp parallel for
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        clusterWeights[clusterIdx] = powf(clustersDistance[clusterIdx], (1 / (1 - pValue))) / sumWeightedDistances;
    }


    free(clustersDistance);
}


float updateSumWeights(float *weightedDistances, const float *euclideanDistances, float *clusterWeights,
                       const float pValue, const mtxDimensions job) {
    /* Get sum of weights to check condition */

    const long numClusters = job.numClusters;
    float sumWeights = 0;

#pragma omp parallel for reduction(+:sumWeights)
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            weightedDistances[objIdx * numClusters + clusterIdx] =
                    powf(clusterWeights[clusterIdx], pValue) *
                    euclideanDistances[(objIdx * numClusters) + clusterIdx];
            sumWeights+= weightedDistances[objIdx * numClusters + clusterIdx];
        }
    }

    return sumWeights;
}

kMeansOutput minMaxKMeans(const float *matrix, float *centroids, const mtxDimensions jobInfo,
                          const int getRandomPoints) {
    /* Main routine for MinMaxKMeans */

    // Initialize variables
    const long numObjs = jobInfo.numObjs, numClusters = jobInfo.numClusters;
    kMeansOutput kMeansResult;
    float pValue = 0;
    const float pStep = 0.01f, pMax = 0.5f;
    int flag = 1;

    // If random is true, initialize centers with random points
    if (getRandomPoints == 1) {
        initializeRandomCenters(matrix, centroids, jobInfo);
    }

    // DELETAR PAR A RANDOMIZAR
    if(jobInfo.numClusters == 2) {
        for (int j = 0; j < jobInfo.numColumns; j++) {
            centroids[0 + j] = matrix[0 + j];
            centroids[1 * jobInfo.numColumns + j] = matrix[2 * jobInfo.numColumns + j];
        }
    }

    printf("Performing MinMaxKMeans...");

    // Initialize matrices and arrays to be used
    float *oldObjsMembership = floatArrayAlloc(numObjs * numClusters, 1);
    float *weightedDistances = floatArrayAlloc(numObjs * numClusters, 1);
    float *objsMtxMembership = floatArrayAlloc(numObjs * numClusters, 1);

    float *clusterWeights = floatArrayAlloc(numClusters, 0);
    float *oldClusterWeights = floatArrayAlloc(numClusters, 0);
    int *objsMembership = intArrayAlloc(numObjs, 0);


    // Initialize cluster weights
    initializeClusterWeights(clusterWeights, oldClusterWeights, numClusters);

    // Initialize euclidean distances
    float *euclideanDistances = floatArrayAlloc(numObjs * numClusters, 1);
    resolveEuclideanDistances(matrix, centroids, euclideanDistances, jobInfo);


    // Initialize matrix weights
    initializeDistanceWeights(weightedDistances, clusterWeights, euclideanDistances, jobInfo, pValue); // ok


    // Variables for loop
    int empty = 0, nIter = 0;
    float oldSumWeights, sumWeights = 0;


    // Loop until weights converge to 0 or hit max loops
    while (1) {
        nIter++;

        // Classify each object and get cluster sizes
        int *clusterSize = getObjectsDistribution(weightedDistances, objsMtxMembership, objsMembership, jobInfo);

        // Lower pValue if some cluster is empty
        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (clusterSize[clusterIdx] < 1) {
                empty = 1;
                pValue -= pStep;

                memcpy(objsMtxMembership, oldObjsMembership, sizeof(float) * numObjs * numClusters);
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
        resolveClusterMeans(matrix, centroids, clusterSize, objsMembership, jobInfo);


        // Increase pValue if no cluster is empty
        if ((pValue < pMax) && (empty == 0)) {
            memcpy(oldObjsMembership, objsMtxMembership, sizeof(float) * numObjs * numClusters);
            memcpy(oldClusterWeights, clusterWeights, sizeof(float) * numClusters);
            pValue += pStep;
        }

        // Update euclidean distances to new centroids
        resolveEuclideanDistances(matrix, centroids, euclideanDistances, jobInfo);

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

    free(weightedDistances);
    free(objsMtxMembership);
    free(oldObjsMembership);
    free(euclideanDistances);
    free(oldClusterWeights);

    return kMeansResult;
}


void cleanOldKMeans(kMeansOutput oldKMeans) {
    /* Free unused kMeans results */

    free(oldKMeans.cluster);
    free(oldKMeans.clusterWeights);
}


kMeansOutput getKMeansResult(const float *matrix, float *centroids, const mtxDimensions jobInfo,
                             int getRandom) {
    /* Perform MinMax KMeans until all clusters are populated */
    int nIter = 0;

    kMeansOutput kMeansResult = minMaxKMeans(matrix, centroids, jobInfo, getRandom);

    while (kMeansResult.flag == 0 && nIter < 6) {
        getRandom = 1;
        nIter++;
        cleanOldKMeans(kMeansResult);
        kMeansResult = minMaxKMeans(matrix, centroids, jobInfo, getRandom);
    }

    return kMeansResult;
}

